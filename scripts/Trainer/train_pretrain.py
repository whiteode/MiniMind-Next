import os
import sys

sys.path.insert(0, os.getcwd())

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DistributedSampler

from scripts.Dataset.lm_dataset import PretrainDataset
from scripts.Trainer.train_common import (
    add_train_args, init_train_env, make_lm_config, make_autocast_ctx, init_wandb,
    get_ckp_data, load_resume_state, wrap_ddp, for_each_epoch, save_checkpoint,
)
from scripts.Trainer.trainer_utils import get_lr, Logger, is_main_process, init_model

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, indices=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            


            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: 
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
            

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(lm_config, model, optimizer, args, epoch, step,
                            scaler=scaler, wandb=wandb, extra_state={'indices': indices})

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    add_train_args(parser, save_weight='pretrain', epochs=1, batch_size=32,
                   learning_rate=5e-4, accumulation_steps=8, max_seq_len=340,
                   data_path='resource/minimind_dataset/pretrain_t2t.jsonl',
                   wandb_project='MiniMind-Pretrain', from_weight='none')
    args = parser.parse_args()

    local_rank = init_train_env(args)
    lm_config = make_lm_config(args)
    ckp_data = get_ckp_data(args, lm_config)
    autocast_ctx = make_autocast_ctx(args)
    wandb = init_wandb(
        args,
        f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}",
        ckp_data,
    )

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    saved_indices = ckp_data.get('indices') if ckp_data else None
    start_epoch, start_step = load_resume_state(ckp_data, model, optimizer, scaler=scaler)
    model = wrap_ddp(model, local_rank)

    for epoch, loader, iters, st, indices in for_each_epoch(
            train_ds, args, start_epoch, start_step,
            train_sampler=train_sampler, saved_indices=saved_indices):
        train_epoch(epoch, loader, iters, st, wandb, indices=indices)

    if dist.is_initialized():
        dist.destroy_process_group()
