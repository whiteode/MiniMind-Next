"""导出随机初始化的模型权重（不训练），作为训练效果对比的基线。

随机初始化的同架构模型可作为 baseline：先用它跑一轮对话/评测，再对比训练后的
模型，能直观验证「训练确实让模型学到了东西」（随机模型输出通常是乱码/重复）。

用法（默认导出 models/random_512.pth）：
    python scripts/Tools/export_random_model.py
    python scripts/Tools/export_random_model.py --hidden_size 768 --num_hidden_layers 16
    python scripts/Tools/export_random_model.py --use_moe 1 --hidden_size 640
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import warnings

import torch

from scripts.Model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='导出随机初始化权重（训练基线，不训练）')
    parser.add_argument('--hidden_size', default=512, type=int, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='隐藏层数量')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='是否MoE架构（0=否，1=是）')
    parser.add_argument('--save_weight', default='random', type=str, help='保存权重前缀名')
    parser.add_argument('--save_dir', default='models', type=str, help='保存目录')
    args = parser.parse_args()

    config = MiniMindConfig(hidden_size=args.hidden_size,
                            num_hidden_layers=args.num_hidden_layers,
                            use_moe=bool(args.use_moe))
    model = MiniMindForCausalLM(config)
    os.makedirs(args.save_dir, exist_ok=True)
    moe_suffix = '_moe' if args.use_moe else ''
    torch_path = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}{moe_suffix}.pth'
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, torch_path)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'随机权重已导出: {torch_path}（{n_params:.2f}M，未训练）')


if __name__ == '__main__':
    main()
