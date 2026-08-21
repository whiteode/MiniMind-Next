"""full_sft 阶段：基于预训练权重的全量监督指令微调（让模型学会对话格式）。"""
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DistributedSampler

from scripts.Dataset.lm_dataset import SFTDataset
from scripts.Trainer.train_common import (
    TrainCtx, init_train_env, make_lm_config, make_autocast_ctx, init_wandb,
    get_ckp_data, load_resume_state, wrap_ddp, for_each_epoch,
)
from scripts.Trainer.trainer_utils import Logger, init_model
from scripts.Trainer.stages._base import train_epoch_sft

STAGE_DEFAULTS = dict(save_weight='full_sft', epochs=2, batch_size=16, learning_rate=1e-6,
                      max_seq_len=340, data_path='resource/minimind_dataset/sft_t2t_mini.jsonl',
                      wandb_project='MiniMind-Full-SFT', from_weight='pretrain')


def add_args(parser):
    """full_sft 无阶段专属 CLI 参数。"""
    return parser


def run(args):
    local_rank = init_train_env(args)
    lm_config = make_lm_config(args)
    ckp_data = get_ckp_data(args, lm_config)
    wandb = init_wandb(
        args,
        f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}",
        ckp_data,
    )

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    ctx = TrainCtx(args=args, lm_config=lm_config, model=model, tokenizer=tokenizer,
                   optimizer=optimizer, scaler=scaler,
                   autocast_ctx=make_autocast_ctx(args), wandb=wandb)

    start_epoch, start_step = load_resume_state(ckp_data, model, optimizer, scaler=scaler)
    model = wrap_ddp(model, local_rank)

    for epoch, loader, iters, st, _ in for_each_epoch(
            train_ds, args, start_epoch, start_step, train_sampler=train_sampler):
        train_epoch_sft(ctx, epoch, loader, iters, st)

    if dist.is_initialized():
        dist.destroy_process_group()
