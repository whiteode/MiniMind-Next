import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.getcwd())
from scripts.Model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.Model.model_lora import apply_lora, load_lora
from scripts.Trainer.trainer_utils import get_model_params


def init_model(args):
    """加载 tokenizer 与模型。
    args.load_from 含 'model' → 加载原生 .pth 权重；否则按 HF 格式目录加载。"""
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
        model.load_state_dict(torch.load(ckp, map_location=args.device, weights_only=True), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer
