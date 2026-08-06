"""MiniMind 终端对话：支持跨轮 KV cache（--enable_kv）。"""
import os
import sys
import time
import argparse
import warnings
import torch
from transformers import TextStreamer

sys.path.insert(0, os.getcwd())
from scripts.Deploy.model_loader import add_model_args, config_from_args, init_model
from scripts.Deploy.kv_generate import SamplingParams, generate_kv
from scripts.Trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')


def window_chunk_text(tokenizer, prompt, is_first, last_is_eos):
    """方案B：构造增量 user 片段文本（首轮走完整模板，后续按模板分隔符追加）。"""
    if is_first:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    sep = '\n' if last_is_eos else f'{tokenizer.eos_token}\n'
    return f'{sep}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'


def trim_window(all_ids, cache, max_tokens, pos_offset):
    """把 KV 缓存裁剪到最近 max_tokens 个 token（前段淘汰），返回 (裁剪后ids, 裁剪后cache, 新偏移)。
    注意：缓存 K/V 布局为 [batch, seq, kv_heads, head_dim]，seq 在 dim 1，沿 dim 1 裁前段。"""
    if cache is None or all_ids is None or all_ids.shape[1] <= max_tokens:
        return all_ids, cache, pos_offset
    drop = all_ids.shape[1] - max_tokens
    all_ids = all_ids[:, drop:]
    cache = [(k[:, drop:, :, :], v[:, drop:, :, :]) for k, v in cache]
    return all_ids, cache, pos_offset + drop


class ChatSession:
    """终端多轮会话：维护对话历史与可选跨轮 KV cache。"""

    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = args.device
        self.weight = args.weight
        self.enable_kv = args.enable_kv
        self.historys = args.historys
        self.max_cache_tokens = args.max_cache_tokens
        self.pos_offset = 0          # 已从缓存前段淘汰的 token 数（全局 RoPE 位置偏移）
        self.params = SamplingParams(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        self.streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.conversation = []
        self.kv_cache = None     # 跨轮 KV cache：每层一个 (k, v) 元组构成的列表
        self.kv_all_ids = None   # 已进入 cache 的全部 token id（前缀求差 & 重复惩罚用）

    def turn(self, prompt):
        """处理一轮输入，返回 (回复文本, 生成token数, 耗时秒)。"""
        setup_seed(2026)
        st = time.time()
        if self.enable_kv:
            if self.max_cache_tokens:
                response, gen_tokens = self._turn_kv_windowed(prompt)
            else:
                response, gen_tokens = self._turn_kv(prompt)
        else:
            response, gen_tokens = self._turn_full(prompt)
        return response, gen_tokens, time.time() - st

    def _turn_kv(self, prompt):
        # ---------- 多轮 KV cache 路径 ----------
        self.conversation.append({"role": "user", "content": prompt})

        if self.weight == 'pretrain':
            # raw 续写：不走模板，直接续接上一轮生成的末尾
            text = prompt if self.kv_all_ids is not None else self.tokenizer.bos_token + prompt
            chunk_ids = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device).input_ids
            self.kv_all_ids = chunk_ids if self.kv_all_ids is None else torch.cat([self.kv_all_ids, chunk_ids], dim=1)
        else:
            # 全量渲染当前对话，再与已缓存 token 求差，只把增量喂给模型
            templates = {"conversation": self.conversation, "tokenize": False, "add_generation_prompt": True}
            if self.weight == 'reason':
                templates["enable_thinking"] = True
            full_ids = self.tokenizer(self.tokenizer.apply_chat_template(**templates), truncation=True).input_ids

            cached_ids = self.kv_all_ids[0].tolist() if self.kv_all_ids is not None else None
            if cached_ids is not None and full_ids[:len(cached_ids)] == cached_ids:
                chunk_ids = torch.tensor(full_ids[len(cached_ids):], dtype=torch.long, device=self.device).unsqueeze(0)
            else:
                # 前缀不匹配（历史被裁剪 / 重编码漂移）→ 丢弃缓存，全量重算
                self.kv_cache = None
                chunk_ids = torch.tensor(full_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            self.kv_all_ids = torch.tensor(full_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        print('🤖: ', end='')
        generated_ids, self.kv_cache, self.kv_all_ids = generate_kv(
            self.model, chunk_ids, self.kv_all_ids, self.kv_cache,
            eos_token_id=self.tokenizer.eos_token_id,
            params=self.params,
            streamer=self.streamer,
        )
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.conversation.append({"role": "assistant", "content": response})
        return response, generated_ids.shape[1]

    def _turn_kv_windowed(self, prompt):
        # ---------- 方案B：滑动窗口 KV（token 追加 + 前段淘汰 + 全局位置偏移） ----------
        eos = self.tokenizer.eos_token_id
        if self.weight == 'pretrain':
            text = prompt if self.kv_all_ids is not None else self.tokenizer.bos_token + prompt
        else:
            is_first = self.kv_all_ids is None
            last_is_eos = (not is_first) and self.kv_all_ids[0, -1].item() == eos
            text = window_chunk_text(self.tokenizer, prompt, is_first, last_is_eos)

        chunk_ids = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device).input_ids
        self.kv_all_ids = chunk_ids if self.kv_all_ids is None else torch.cat([self.kv_all_ids, chunk_ids], dim=1)

        print('🤖: ', end='')
        generated_ids, self.kv_cache, self.kv_all_ids = generate_kv(
            self.model, chunk_ids, self.kv_all_ids, self.kv_cache,
            eos_token_id=eos,
            params=self.params,
            streamer=self.streamer,
            position_offset=self.pos_offset,
        )
        self.kv_all_ids, self.kv_cache, self.pos_offset = trim_window(
            self.kv_all_ids, self.kv_cache, self.max_cache_tokens, self.pos_offset)

        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.conversation.append({"role": "assistant", "content": response})
        return response, generated_ids.shape[1]

    def _turn_full(self, prompt):
        # ---------- 原路径：每轮全量重编码 ----------
        self.conversation = self.conversation[-self.historys:] if self.historys else []
        self.conversation.append({"role": "user", "content": prompt})

        if self.weight == 'pretrain':
            inputs = self.tokenizer(self.tokenizer.bos_token + prompt, return_tensors='pt', truncation=True).to(self.device)
        else:
            templates = {"conversation": self.conversation, "tokenize": False, "add_generation_prompt": True}
            if self.weight == 'reason':
                templates["enable_thinking"] = True
            text = self.tokenizer.apply_chat_template(**templates)
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device)

        print('🤖: ', end='')
        generated_ids = self.model.generate(
            inputs=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=self.params.max_new_tokens,
            do_sample=True,
            streamer=self.streamer,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            top_p=self.params.top_p,
            temperature=self.params.temperature,
            repetition_penalty=self.params.repetition_penalty,
        )

        response = self.tokenizer.decode(generated_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        self.conversation.append({"role": "assistant", "content": response})
        return response, len(generated_ids[0]) - len(inputs['input_ids'][0])


def main():
    parser = argparse.ArgumentParser(description='MiniMind 终端对话')
    add_model_args(parser)
    parser.add_argument('--max_new_tokens', default=8192, type=int)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--historys', default=0, type=int, help='携带历史对话轮数（需为偶数，0=不带历史）')
    parser.add_argument('--enable_kv', default=False, action='store_true', help='启用跨轮 KV cache（多轮只计算新增 token，与 --historys 互斥）')
    parser.add_argument('--max_cache_tokens', default=0, type=int, help='KV 缓存窗口上限（token 数；0=不设上限保留全部历史；需配合 --enable_kv，实现滑动窗口方案B）')
    parser.add_argument('--repetition_penalty', default=1.0, type=float)
    parser.add_argument('--show_speed', default=1, type=int)
    args = parser.parse_args()

    if args.enable_kv and args.historys != 0:
        raise SystemExit('--enable_kv 与 --historys 互斥：KV 模式会维护全部对话历史')
    if args.max_cache_tokens and not args.enable_kv:
        raise SystemExit('--max_cache_tokens 需要配合 --enable_kv 使用')

    model, tokenizer = init_model(config_from_args(args))
    session = ChatSession(model, tokenizer, args)

    while True:
        prompt = input('💬: ').strip()
        if not prompt:
            break
        _, gen_tokens, elapsed = session.turn(prompt)
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / elapsed:.2f} tokens/s\n')
        else:
            print()


if __name__ == '__main__':
    main()
