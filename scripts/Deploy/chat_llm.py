import os
import sys
import time
import argparse
import warnings
import torch
from transformers import TextStreamer

sys.path.insert(0, os.getcwd())
from scripts.Deploy.model_loader import init_model
from scripts.Deploy.kv_generate import generate_kv
from scripts.Trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='MiniMind 终端对话')
    parser.add_argument('--load_from', default='scripts/Model', type=str)
    parser.add_argument('--save_dir', default='models', type=str)
    parser.add_argument('--weight', default='full_sft', type=str,
                        help='权重前缀，如 full_sft / pretrain / reason')
    parser.add_argument('--lora_weight', default='None', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true')
    parser.add_argument('--max_new_tokens', default=8192, type=int)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--historys', default=0, type=int, help='携带历史对话轮数（需为偶数，0=不带历史）')
    parser.add_argument('--enable_kv', default=False, action='store_true', help='启用跨轮 KV cache（多轮只计算新增 token，与 --historys 互斥）')
    parser.add_argument('--repetition_penalty', default=1.0, type=float)
    parser.add_argument('--show_speed', default=1, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.enable_kv and args.historys != 0:
        raise SystemExit('--enable_kv 与 --historys 互斥：KV 模式会维护全部对话历史')

    model, tokenizer = init_model(args)

    conversation = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    kv_cache = None    # 跨轮 KV cache：每层一个 (k, v) 元组构成的列表
    kv_all_ids = None  # 已进入 cache 的全部 token id（前缀求差 & 重复惩罚用）

    while True:
        prompt = input('💬: ').strip()
        if not prompt:
            break

        setup_seed(2026)

        if args.enable_kv:
            # ---------- 多轮 KV cache 路径 ----------
            conversation.append({"role": "user", "content": prompt})

            if args.weight == 'pretrain':
                # raw 续写：不走模板，直接续接上一轮生成的末尾
                text = prompt if kv_all_ids is not None else tokenizer.bos_token + prompt
                chunk_ids = tokenizer(text, return_tensors='pt', truncation=True).to(args.device).input_ids
                kv_all_ids = chunk_ids if kv_all_ids is None else torch.cat([kv_all_ids, chunk_ids], dim=1)
            else:
                # 全量渲染当前对话，再与已缓存 token 求差，只把增量喂给模型
                templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
                if args.weight == 'reason':
                    templates["enable_thinking"] = True
                full_ids = tokenizer(tokenizer.apply_chat_template(**templates), truncation=True).input_ids

                cached_ids = kv_all_ids[0].tolist() if kv_all_ids is not None else None
                if cached_ids is not None and full_ids[:len(cached_ids)] == cached_ids:
                    chunk_ids = torch.tensor(full_ids[len(cached_ids):], dtype=torch.long, device=args.device).unsqueeze(0)
                else:
                    # 前缀不匹配（历史被裁剪 / 重编码漂移）→ 丢弃缓存，全量重算
                    kv_cache = None
                    chunk_ids = torch.tensor(full_ids, dtype=torch.long, device=args.device).unsqueeze(0)
                kv_all_ids = torch.tensor(full_ids, dtype=torch.long, device=args.device).unsqueeze(0)

            print('🤖: ', end='')
            st = time.time()

            generated_ids, kv_cache, kv_all_ids = generate_kv(
                model, chunk_ids, kv_all_ids, kv_cache,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            gen_tokens = generated_ids.shape[1]
        else:
            # ---------- 原路径：每轮全量重编码 ----------
            conversation = conversation[-args.historys:] if args.historys else []
            conversation.append({"role": "user", "content": prompt})

            if args.weight == 'pretrain':
                inputs = tokenizer(tokenizer.bos_token + prompt, return_tensors='pt', truncation=True).to(args.device)
            else:
                templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
                if args.weight == 'reason':
                    templates["enable_thinking"] = True
                text = tokenizer.apply_chat_template(**templates)
                inputs = tokenizer(text, return_tensors='pt', truncation=True).to(args.device)

            print('🤖: ', end='')
            st = time.time()

            generated_ids = model.generate(
                inputs=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )

            response = tokenizer.decode(generated_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            gen_tokens = len(generated_ids[0]) - len(inputs['input_ids'][0])

        conversation.append({"role": "assistant", "content": response})

        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n')
        else:
            print()


if __name__ == '__main__':
    main()
