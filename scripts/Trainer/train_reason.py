import os
import sys

sys.path.insert(0, os.getcwd())

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.utils.data import DistributedSampler

from scripts.Dataset.lm_dataset import SFTDataset
from scripts.Trainer.train_common import (
    add_train_args, init_train_env, make_lm_config, make_autocast_ctx, init_wandb,
    get_ckp_data, load_resume_state, wrap_ddp, for_each_epoch, save_checkpoint,
)
from scripts.Trainer.trainer_utils import get_lr, Logger, is_main_process, init_model

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, tokenizer, lm_config, start_step=0, wandb=None):
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids)
            shift_logits = res.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

            loss_mask = (shift_labels != -100).float()
            sp_ids = torch.isin(shift_labels.view(-1),
                                torch.tensor(start_of_think_ids + end_of_think_ids
                                             + start_of_answer_ids + end_of_answer_ids
                                             ).to(args.device))
            loss_mask_flat = loss_mask.view(-1)
            loss_mask_sum = loss_mask_flat.sum()
            loss_mask_flat[sp_ids] = 10
            loss_mask = loss_mask_flat.view(shift_labels.size())
            logits_loss = (loss * loss_mask).sum() / loss_mask_sum
            loss = logits_loss + res.aux_loss
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
            current_logits_loss = logits_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(lm_config, model, optimizer, args, epoch, step,
                            scaler=scaler, wandb=wandb)

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Reasoning Distillation")
    add_train_args(parser, save_weight='reason', epochs=1, batch_size=8,
                   learning_rate=1e-6, save_interval=100, max_seq_len=720,
                   data_path='resource/minimind_dataset/r1_mix_1024.jsonl',
                   wandb_project='MiniMind-Reasoning', from_weight='dpo')
    args = parser.parse_args()

    local_rank = init_train_env(args)
    lm_config = make_lm_config(args)
    ckp_data = get_ckp_data(args, lm_config)
    autocast_ctx = make_autocast_ctx(args)
    wandb = init_wandb(
        args,
        f"MiniMind-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}",
        ckp_data,
    )

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = load_resume_state(ckp_data, model, optimizer, scaler=scaler)
    model = wrap_ddp(model, local_rank)

    for epoch, loader, iters, st, _ in for_each_epoch(
            train_ds, args, start_epoch, start_step, train_sampler=train_sampler):
        train_epoch(epoch, loader, iters, tokenizer, lm_config, st, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
