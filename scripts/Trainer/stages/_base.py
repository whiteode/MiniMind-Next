"""SFT 系共用基础组件（私有模块，不对外提供入口）。"""
import time

import torch

from scripts.Trainer.train_common import save_checkpoint
from scripts.Trainer.trainer_utils import get_lr, Logger, is_main_process


def train_epoch_sft(ctx, epoch, loader, iters, start_step, indices=None):
    """标准自回归 CE 训练（pretrain / full_sft 共用；pretrain 续训透传 indices 存档）。"""
    args, model, optimizer, scaler = ctx.args, ctx.model, ctx.optimizer, ctx.scaler
    autocast_ctx, lm_config, wandb = ctx.autocast_ctx, ctx.lm_config, ctx.wandb
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
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, '
                   f'logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                   f'lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss,
                           "aux_loss": current_aux_loss, "learning_rate": current_lr,
                           "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(lm_config, model, optimizer, args, epoch, step, iters=iters,
                            scaler=scaler, wandb=wandb, extra_state={'indices': indices})

        del input_ids, labels, res, loss
