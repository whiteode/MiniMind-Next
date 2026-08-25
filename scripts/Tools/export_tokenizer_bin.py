import os
import sys
import json
import struct
import argparse

def get_unicode_to_byte():
    """
    构建 HuggingFace Byte-Level BPE 的 Unicode 字符到真实字节 (0~255) 的反向映射字典
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}

def load_bpe_ranks(merges_path, vocab):
    """
    从 merges.txt 或 merges 规则列表加载 BPE 合并规则，返回 {(left_id, right_id): rank}
    """
    bpe_ranks = {}
    if not merges_path or not os.path.exists(merges_path):
        return bpe_ranks

    with open(merges_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过第一行注释或版本号（如果存在）
    start_idx = 1 if lines and (lines[0].startswith("#") or "version" in lines[0]) else 0
    for rank, line in enumerate(lines[start_idx:]):
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ')
        if len(parts) != 2:
            continue
        left_token, right_token = parts[0], parts[1]
        if left_token in vocab and right_token in vocab:
            left_id = vocab[left_token]
            right_id = vocab[right_token]
            bpe_ranks[(left_id, right_id)] = rank

    return bpe_ranks

def compile_vocab_bin(tokenizer_dir_or_json, output_bin_path):
    """
    将 tokenizer.json 或包含 tokenizer 文件的目录编译为紧凑二进制词表 (.vocab.bin)
    """
    print(f"\n>>> 开始编译二进制词表: {tokenizer_dir_or_json} -> {output_bin_path}")

    vocab = {}
    merges_list = []

    # 1. 尝试从 tokenizer.json 或 vocab.json 读取
    if os.path.isdir(tokenizer_dir_or_json):
        tokenizer_json_path = os.path.join(tokenizer_dir_or_json, "tokenizer.json")
        vocab_json_path = os.path.join(tokenizer_dir_or_json, "vocab.json")
        merges_txt_path = os.path.join(tokenizer_dir_or_json, "merges.txt")
        added_tokens_path = os.path.join(tokenizer_dir_or_json, "added_tokens.json")
    else:
        tokenizer_json_path = tokenizer_dir_or_json
        vocab_json_path = None
        merges_txt_path = None
        added_tokens_path = None

    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tk_data = json.load(f)
            model_info = tk_data.get("model", {})
            vocab = model_info.get("vocab", {})
            merges_list = model_info.get("merges", [])
            # 处理 added_tokens
            for item in tk_data.get("added_tokens", []):
                content = item.get("content")
                token_id = item.get("id")
                if content is not None and token_id is not None:
                    vocab[content] = token_id
    elif vocab_json_path and os.path.exists(vocab_json_path):
        with open(vocab_json_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        if os.path.exists(added_tokens_path):
            with open(added_tokens_path, 'r', encoding='utf-8') as f:
                vocab.update(json.load(f))
    else:
        raise FileNotFoundError(f"未找到有效的 tokenizer 文件: {tokenizer_dir_or_json}")

    if not vocab:
        raise ValueError("词表为空！")

    max_id = max(vocab.values())
    n_tokens = max_id + 1
    id_to_str = [b""] * n_tokens
    unicode_to_byte = get_unicode_to_byte()

    for token_str, token_id in vocab.items():
        try:
            token_bytes = bytes([unicode_to_byte[c] for c in token_str])
        except KeyError:
            token_bytes = token_str.encode('utf-8')
        id_to_str[token_id] = token_bytes

    string_data_block = bytearray()
    info_array = []

    for token_id in range(n_tokens):
        token_bytes = id_to_str[token_id]
        offset = len(string_data_block)
        length = len(token_bytes)
        info_array.append((offset, length))
        string_data_block.extend(token_bytes)

    # 提取有实质内容的 Token 进行严格字典序排列
    valid_items = [(id_to_str[tid], tid) for tid in range(n_tokens) if len(id_to_str[tid]) > 0]
    valid_items.sort(key=lambda x: x[0])
    sorted_ids = [item[1] for item in valid_items]
    n_sorted = len(sorted_ids)

    # 加载 BPE 合并规则
    bpe_ranks = {}
    if merges_list:
        for rank, merge_str in enumerate(merges_list):
            parts = merge_str.split(' ') if isinstance(merge_str, str) else merge_str
            if len(parts) == 2:
                l_str, r_str = parts[0], parts[1]
                if l_str in vocab and r_str in vocab:
                    bpe_ranks[(vocab[l_str], vocab[r_str])] = rank
    elif merges_txt_path and os.path.exists(merges_txt_path):
        bpe_ranks = load_bpe_ranks(merges_txt_path, vocab)

    # 构建 256 字节到 Token ID 映射表
    byte_to_token = [0] * 256
    byte_to_unicode = {v: k for k, v in unicode_to_byte.items()}
    mapped_count = 0
    for byte_val in range(256):
        if byte_val in byte_to_unicode:
            uchar = byte_to_unicode[byte_val]
            if uchar in vocab:
                byte_to_token[byte_val] = vocab[uchar]
                mapped_count += 1

    print(f"    - 总 Token 数 (n_tokens): {n_tokens}")
    print(f"    - 字典序索引数 (n_sorted): {n_sorted}")
    print(f"    - BPE 合并规则数: {len(bpe_ranks)}")
    print(f"    - 基础字节映射数: {mapped_count}/256")

    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    with open(output_bin_path, 'wb') as f:
        # Binary header: n_tokens (4B), n_sorted (4B)
        f.write(struct.pack('<II', n_tokens, n_sorted))

        # Token Info Array
        for offset, length in info_array:
            f.write(struct.pack('<II', offset, length))

        # Sorted IDs Array
        for tid in sorted_ids:
            f.write(struct.pack('<I', tid))

        # Raw Bytes Pool
        f.write(string_data_block)

        # BPE Merges
        f.write(struct.pack('<I', len(bpe_ranks)))
        for (left_id, right_id), rank in bpe_ranks.items():
            f.write(struct.pack('<III', left_id, right_id, rank))

        # 256 Byte-to-Token table
        for token_id in byte_to_token:
            f.write(struct.pack('<I', token_id))

    print(f"✅ 成功生成词表二进制文件: {output_bin_path} ({os.path.getsize(output_bin_path) / 1024:.2f} KB)\n")

def main():
    parser = argparse.ArgumentParser(description="Export Tokenizer into high-performance binary format (.vocab.bin)")
    parser.add_argument("--tokenizer_path", type=str, default="resource/MiniMind2",
                        help="Path to tokenizer directory or tokenizer.json")
    parser.add_argument("--output", type=str, default="models/minimind2.vocab.bin",
                        help="Output path for .vocab.bin binary file")
    args = parser.parse_args()

    compile_vocab_bin(args.tokenizer_path, args.output)

if __name__ == "__main__":
    main()
