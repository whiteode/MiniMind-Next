"""模型 / 分词器加载：ModelConfig 配置 + add_model_args 统一 CLI 参数 + init_model 加载。

`format` 显式指定权重格式（native=原生 .pth，hf=HF 格式目录），不靠路径字符串推断。
"""
import sys
import os
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.getcwd())
from scripts.Model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.Model.model_lora import apply_lora, load_lora
from scripts.Trainer.trainer_utils import get_model_params


@dataclass
class ModelConfig:
    """模型加载所需的全部配置（不依赖 argparse）。"""
    load_from: str = 'scripts/Model'
    save_dir: str = 'models'
    weight: str = 'full_sft'
    hidden_size: int = 512
    num_hidden_layers: int = 8
    use_moe: bool = False
    lora_weight: str = 'None'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference_rope_scaling: bool = False
    format: str = 'native'          # native=原生 .pth，hf=HF 格式目录


WEIGHT_HELP = (
    "权重名称前缀，用于指定加载哪一阶段训练出的模型权重。\n"
    "  pretrain   - 预训练（基础语言能力）\n"
    "  full_sft   - 全量指令微调（默认，能对话）\n"
    "  dpo        - DPO 偏好优化\n"
    "  reason     - 推理微调\n"
    "  ppo_actor / grpo / spo - 强化学习阶段权重\n"
    "推理时 --weight 主要用于定位权重文件，模型结构不因训练阶段改变。"
)


def add_model_args(parser):
    """把模型加载相关 CLI 参数统一加到 parser 上，供各 Deploy 脚本复用。"""
    parser.add_argument('--load_from', default='scripts/Model', type=str, help='模型加载路径（原生权重目录或 HF 格式目录）')
    parser.add_argument('--save_dir', default='models', type=str, help='模型权重目录')
    parser.add_argument('--weight', default='full_sft', type=str, help=WEIGHT_HELP)
    parser.add_argument('--lora_weight', default='None', type=str, help='LoRA 权重名称（None=不使用）')
    parser.add_argument('--hidden_size', default=512, type=int, help='隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='隐藏层数量（Small/MoE=8, Base=16）')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='是否使用 MoE 架构（0=否，1=是）')
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help='启用 RoPE 位置编码外推（4 倍）')
    parser.add_argument('--format', default='native', type=str, choices=['native', 'hf'], help='权重格式：native=原生 .pth（默认），hf=HF 格式目录')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='运行设备')
    return parser


def config_from_args(args) -> ModelConfig:
    """从 argparse 结果构造 ModelConfig。"""
    return ModelConfig(
        load_from=args.load_from,
        save_dir=args.save_dir,
        weight=args.weight,
        lora_weight=args.lora_weight,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling,
        device=args.device,
        format=args.format,
    )


def init_model(cfg: ModelConfig):
    """加载 tokenizer 与模型。cfg.format 指定权重格式（native / hf）。"""
    tokenizer = AutoTokenizer.from_pretrained(cfg.load_from)

    if cfg.format == 'hf':
        model = AutoModelForCausalLM.from_pretrained(cfg.load_from, trust_remote_code=True)
    else:
        config = MiniMindConfig(
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            use_moe=cfg.use_moe,
            inference_rope_scaling=cfg.inference_rope_scaling,
        )
        model = MiniMindForCausalLM(config)
        moe_suffix = '_moe' if cfg.use_moe else ''
        ckp = f'./{cfg.save_dir}/{cfg.weight}_{cfg.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=cfg.device, weights_only=True), strict=True)
        if cfg.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{cfg.save_dir}/lora/{cfg.lora_weight}_{cfg.hidden_size}.pth')

    get_model_params(model, model.config)
    return model.eval().to(cfg.device), tokenizer
