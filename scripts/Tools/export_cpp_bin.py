"""将 HuggingFace 格式的 MiniMind2 (如 resource/MiniMind2) 导出为纯 C++ 推理的二进制 .bin 格式。

导出内容：
1. 超参数 Header (7 个 int32: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
2. 词表字符串与词表打分 (用于纯 C++ Tokenizer)
3. 按照 Transformer 执行顺序紧凑排列的所有 fp32 权重 (可在 C++ 中直接 mmap 或 fread 映射为指针)

用法：
    python scripts/Tools/export_cpp_bin.py --model_dir resource/MiniMind2 --output models/minimind2.bin
"""
import argparse
import json
import os
import struct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_bin(model_dir: str, output_path: str, max_seq_len: int = 1024):
    print(f"正在从 {model_dir} 加载模型与分词器...")
    with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    state = model.state_dict()

    dim = cfg['hidden_size']                      # 768
    hidden_dim = cfg['intermediate_size']         # 2048
    n_layers = cfg['num_hidden_layers']           # 16
    n_heads = cfg['num_attention_heads']          # 8
    n_kv_heads = cfg.get('num_key_value_heads', n_heads) # 2
    vocab_size = cfg['vocab_size']                # 6400

    print(f"模型参数: dim={dim}, hidden_dim={hidden_dim}, layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}, vocab={vocab_size}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'wb') as f:
        # 1. 写入 Header (7 个 int32)
        header = struct.pack('7i', dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len)
        f.write(header)

        # 2. 写入 Tokenizer 词表
        # 每个 token 写入真实的原始字节序列 (还原 Byte-level BPE 映射，彻底消除中文多字节切分乱码)
        print("正在写入词表...")
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

        for i in range(vocab_size):
            t = tokenizer.convert_ids_to_tokens(i)
            if t is None:
                b_bytes = f"<unk_{i}>".encode('utf-8')
            elif t.startswith('<|') and t.endswith('|>'):
                b_bytes = t.encode('utf-8')
            else:
                try:
                    b_bytes = bytes([byte_decoder.get(c, ord(c)) for c in t])
                except Exception:
                    b_bytes = t.encode('utf-8')

            f.write(struct.pack('H', len(b_bytes)))
            f.write(b_bytes)

        # 辅助写入 tensor 函数
        def write_tensor(tensor: torch.Tensor, name: str):
            arr = tensor.detach().cpu().to(torch.float32).numpy()
            f.write(arr.tobytes())

        print("正在写入权重参数...")
        # 3.1 Token Embedding: [vocab_size, dim]
        write_tensor(state['model.embed_tokens.weight'], 'embed_tokens')

        # 3.2 各层 Transformer Block 权重
        for i in range(n_layers):
            # Attention RMSNorm: [dim]
            write_tensor(state[f'model.layers.{i}.input_layernorm.weight'], f'layer.{i}.input_layernorm')
            # Q projection: [n_heads * (dim/n_heads), dim] -> [dim, dim]
            write_tensor(state[f'model.layers.{i}.self_attn.q_proj.weight'], f'layer.{i}.q_proj')
            # K projection: [n_kv_heads * (dim/n_heads), dim]
            write_tensor(state[f'model.layers.{i}.self_attn.k_proj.weight'], f'layer.{i}.k_proj')
            # V projection: [n_kv_heads * (dim/n_heads), dim]
            write_tensor(state[f'model.layers.{i}.self_attn.v_proj.weight'], f'layer.{i}.v_proj')
            # Out projection: [dim, dim]
            write_tensor(state[f'model.layers.{i}.self_attn.o_proj.weight'], f'layer.{i}.o_proj')

            # FFN RMSNorm: [dim]
            write_tensor(state[f'model.layers.{i}.post_attention_layernorm.weight'], f'layer.{i}.post_attn_layernorm')
            # Gate proj (SwiGLU): [hidden_dim, dim]
            write_tensor(state[f'model.layers.{i}.mlp.gate_proj.weight'], f'layer.{i}.gate_proj')
            # Up proj (SwiGLU): [hidden_dim, dim]
            write_tensor(state[f'model.layers.{i}.mlp.up_proj.weight'], f'layer.{i}.up_proj')
            # Down proj (SwiGLU): [dim, hidden_dim]
            write_tensor(state[f'model.layers.{i}.mlp.down_proj.weight'], f'layer.{i}.down_proj')

        # 3.3 最终 RMSNorm: [dim]
        write_tensor(state['model.norm.weight'], 'final_norm')

        # 3.4 输出 LM Head: 如果 tie_word_embeddings 为 true，直接共享或者写一份
        if 'lm_head.weight' in state:
            write_tensor(state['lm_head.weight'], 'lm_head')
        else:
            write_tensor(state['model.embed_tokens.weight'], 'lm_head')

    print(f"🎉 导出成功！二进制模型文件已生成至: {output_path} (文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="将 MiniMind2 导出为 C++ 推理专用 .bin 文件")
    parser.add_argument("--model_dir", type=str, default="resource/MiniMind2", help="HuggingFace 模型目录")
    parser.add_argument("--output", type=str, default="models/minimind2.bin", help="导出的 .bin 文件路径")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="支持的最大推理序列长度")
    args = parser.parse_args()

    export_bin(args.model_dir, args.output, args.max_seq_len)


if __name__ == '__main__':
    main()
