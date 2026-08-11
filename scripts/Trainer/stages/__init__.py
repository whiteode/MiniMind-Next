"""SFT 系各阶段训练模块：由 scripts/Trainer/train.py 按 --stage 路由导入。

每个模块导出 STAGE_DEFAULTS（add_train_args 默认值）/ add_args(parser)（阶段专属参数）/ run(args)。
RL 阶段（GRPO/PPO/SPO）结构差异大，保留独立脚本 train_grpo/ppo/spo.py。
"""
