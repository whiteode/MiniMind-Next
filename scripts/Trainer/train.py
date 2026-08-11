"""MiniMind 训练入口：--stage 路由到 scripts/Trainer/stages/<stage>.py 执行。

SFT 系阶段（pretrain / full_sft / reason / lora / dpo / distillation）由各 stage 模块实现；
RL 阶段（GRPO/PPO/SPO，多模型 + reward）结构差异大，保留独立脚本 train_grpo/ppo/spo.py。
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.Trainer.train_common import add_train_args

STAGES = ['pretrain', 'full_sft', 'reason', 'lora', 'dpo', 'distillation']


def main():
    parser = argparse.ArgumentParser(description='MiniMind SFT 系训练（--stage 路由到 stages/ 模块）')
    parser.add_argument('--stage', type=str, choices=STAGES, default='full_sft',
                        help='训练阶段: ' + '/'.join(STAGES))
    stage = parser.parse_known_args()[0].stage

    module = importlib.import_module(f'scripts.Trainer.stages.{stage}')
    add_train_args(parser, **module.STAGE_DEFAULTS)
    module.add_args(parser)
    args = parser.parse_args()
    module.run(args)


if __name__ == '__main__':
    main()
