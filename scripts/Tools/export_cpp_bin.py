"""将 HuggingFace 格式的 MiniMind2 (如 resource/MiniMind2) 导出为纯 C++ 推理的 Transformer 核心权重 .bin 格式。

说明：
- 分词器 (.vocab.bin) 请使用 scripts/Tools/export_tokenizer_bin.py 导出。
- Embedding 矩阵 (.embedding) 请使用 scripts/Tools/export_embedding.py 导出（支持 14 种量化方案）。
- 本脚本仅导出 Transformer 核心超参数 Header 与各层计算权重。

用法：
    python scripts/Tools/export_cpp_bin.py --model_dir resource/MiniMind2 --output models/minimind2.bin
"""
import argparse
import json
import os
import struct
import torch
from transformers import AutoModelForCausalLM


def export_bin(model_dir: str, output_path: str, max_seq_len: int = 1024):
    print(f"正在从 {model_dir} 加载模型结构与权重...")
    with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
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

        # 辅助写入 tensor 函数
        def write_tensor(tensor: torch.Tensor, name: str):
            arr = tensor.detach().cpu().to(torch.float32).numpy()
            f.write(arr.tobytes())

        print("正在写入 Transformer 各层权重参数 (不含 Embedding)...")
        # 2. 各层 Transformer Block 权重
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

        # 3. 最终 RMSNorm: [dim]
        write_tensor(state['model.norm.weight'], 'final_norm')

        # 4. 输出 LM Head: [vocab_size, dim]
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
