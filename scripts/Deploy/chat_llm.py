import os
import sys
import time
import argparse
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

sys.path.insert(0, os.getcwd())
from scripts.Model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.Model.model_lora import apply_lora, load_lora
from scripts.Trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if 'model' in args.load_from.lower():
        config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling,
        )
        model = MiniMindForCausalLM(config)
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device, weights_only=False), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


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
    parser.add_argument('--repetition_penalty', default=1.0, type=float)
    parser.add_argument('--show_speed', default=1, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = init_model(args)

    conversation = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        prompt = input('💬: ').strip()
        if not prompt:
            break

        setup_seed(2026)
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
        conversation.append({"role": "assistant", "content": response})

        gen_tokens = len(generated_ids[0]) - len(inputs['input_ids'][0])
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n')
        else:
            print()


if __name__ == '__main__':
    main()
