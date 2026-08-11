"""MiniMind SFT 系训练统一入口：--stage 选择 pretrain / full_sft / reason / lora / dpo / distillation。

RL 阶段（GRPO/PPO/SPO，多模型 + reward + scheduler）结构差异大，保留独立脚本 train_grpo/ppo/spo.py。
"""
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.getcwd())

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DistributedSampler

from scripts.Dataset.lm_dataset import PretrainDataset, SFTDataset, DPODataset
from scripts.Model.model_lora import apply_lora, save_lora
from scripts.Trainer.train_common import (
    add_train_args, init_train_env, make_lm_config, make_autocast_ctx, init_wandb,
    get_ckp_data, load_resume_state, wrap_ddp, for_each_epoch, save_checkpoint,
)
from scripts.Trainer.trainer_utils import get_lr, Logger, is_main_process, init_model

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# 各阶段参数
# ---------------------------------------------------------------------------

STAGES = {
    'pretrain': dict(save_weight='pretrain', epochs=1, batch_size=32, learning_rate=5e-4,
                     accumulation_steps=8, max_seq_len=340,
                     data_path='resource/minimind_dataset/pretrain_t2t.jsonl',
                     wandb_project='MiniMind-Pretrain', from_weight='none'),
    'full_sft': dict(save_weight='full_sft', epochs=2, batch_size=16, learning_rate=1e-6,
                     max_seq_len=340, data_path='resource/minimind_dataset/sft_t2t_mini.jsonl',
                     wandb_project='MiniMind-Full-SFT', from_weight='pretrain'),
    'reason': dict(save_weight='reason', epochs=1, batch_size=8, learning_rate=1e-6,
                   save_interval=100, max_seq_len=720,
                   data_path='resource/minimind_dataset/r1_mix_1024.jsonl',
                   wandb_project='MiniMind-Reasoning', from_weight='dpo'),
    'lora': dict(save_dir='models/lora', save_weight='lora_identity', epochs=50, batch_size=32,
                 learning_rate=1e-4, log_interval=10, max_seq_len=340,
                 data_path='resource/minimind_dataset/lora_identity.jsonl',
                 wandb_project='MiniMind-LoRA', from_weight='full_sft'),
    'dpo': dict(save_weight='dpo', epochs=1, batch_size=4, learning_rate=4e-8,
                save_interval=100, max_seq_len=1024, data_path='resource/minimind_dataset/dpo.jsonl',
                wandb_project='MiniMind-DPO', from_weight='full_sft'),
    'distillation': dict(save_weight='full_dist', epochs=6, batch_size=32, learning_rate=5e-6,
                         save_interval=100, max_seq_len=340,
                         data_path='resource/minimind_dataset/sft_t2t_mini.jsonl',
                         wandb_project='MiniMind-Distillation', from_weight=None),
}


def add_stage_args(parser, stage):
    """注册各阶段专属 CLI 参数。"""
    if stage == 'lora':
        parser.add_argument('--lora_name', type=str, default='lora_identity',
                            help='LoRA权重名称(如lora_identity/lora_medical等)')
    if stage == 'dpo':
        parser.add_argument('--beta', default=0.1, type=float, help='DPO loss 中的 beta 温度参数')
    if stage == 'distillation':
        parser.add_argument('--student_hidden_size', default=512, type=int, help='学生模型隐藏层维度')
        parser.add_argument('--student_num_layers', default=8, type=int, help='学生模型隐藏层数量')
        parser.add_argument('--teacher_hidden_size', default=768, type=int, help='教师模型隐藏层维度')
        parser.add_argument('--teacher_num_layers', default=16, type=int, help='教师模型隐藏层数量')
        parser.add_argument('--from_student_weight', default='full_sft', type=str, help='学生模型基于哪个权重')
        parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help='教师模型基于哪个权重')
        parser.add_argument('--alpha', default=0.5, type=float, help='CE损失权重，总损失=alpha*CE+(1-alpha)*KL')
        parser.add_argument('--temperature', default=1.5, type=float, help='蒸馏温度（推荐范围1.0-2.0）')


# ---------------------------------------------------------------------------
# 各阶段训练循环
# ---------------------------------------------------------------------------

def train_epoch_sft(epoch, loader, iters, start_step, wandb, indices=None):
    """pretrain / full_sft 共用：标准自回归 CE 训练（pretrain 续训时透传 indices 存档）。"""
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
            save_checkpoint(lm_config, model, optimizer, args, epoch, step,
                            scaler=scaler, wandb=wandb, extra_state={'indices': indices})

        del input_ids, labels, res, loss


def train_epoch_reason(epoch, loader, iters, tokenizer, lm_config, start_step, wandb):
    """推理蒸馏：<think>/<answer> 特殊 token 位置加权 10 倍。"""
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
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)).view(shift_labels.size())

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
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, '
                   f'logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                   f'lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss,
                           "aux_loss": current_aux_loss, "learning_rate": current_lr,
                           "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(lm_config, model, optimizer, args, epoch, step,
                            scaler=scaler, wandb=wandb)

        del input_ids, labels, res, loss


def train_epoch_lora(epoch, loader, iters, lora_params, start_step, wandb):
    """LoRA：只对 lora 参数剪裁/优化，保存时只落盘 LoRA 权重（save_lora）。"""
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
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
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
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                   f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, '
                   f'epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss,
                           "aux_loss": current_aux_loss, "learning_rate": current_lr,
                           "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(lm_config, model, optimizer, args, epoch, step,
                            scaler=scaler, wandb=wandb,
                            weight_saver=lambda m, p: save_lora(m, p))

        del input_ids, labels, res, loss


def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch_dpo(epoch, loader, iters, ref_model, lm_config, start_step, wandb, beta=0.1):
    """DPO：策略模型 + 冻结参考模型，chosen/rejected 拼接。"""
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
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
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, '
                   f'aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, '
                   f'epoch_time: {eta_min:.3f}min')
            if wandb:
                wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss,
                           "aux_loss": current_aux_loss, "learning_rate": current_lr,
                           "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(lm_config, model, optimizer, args, epoch, step,
                            scaler=scaler, wandb=wandb)

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl


def train_epoch_distillation(epoch, loader, iters, teacher_model, lm_config_student, start_step, wandb, alpha=0.0, temperature=1.0):
    """知识蒸馏：学生 CE 损失 + 教师 KL 损失按 alpha 加权。"""
    start_time = time.time()

    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        loss_mask = (labels[..., 1:] != -100).float()
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()

        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        if lm_config_student.use_moe:
            ce_loss = ce_loss_raw + res.aux_loss
        else:
            ce_loss = ce_loss_raw

        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

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
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, '
                   f'ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                   f'distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, '
                   f'epoch_time: {eta_min:.3f}min')
            if wandb:
                wandb.log({"loss": current_loss, "ce_loss": current_ce_loss,
                           "aux_loss": current_aux_loss,
                           "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                           "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            save_checkpoint(lm_config_student, model, optimizer, args, epoch, step,
                            scaler=scaler, wandb=wandb)

        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    global args, model, tokenizer, optimizer, scaler, autocast_ctx, lm_config, wandb
    global lora_params, ref_model, teacher_model, lm_config_student

    parser = argparse.ArgumentParser(description='MiniMind SFT 系训练（--stage 选择阶段）')
    parser.add_argument('--stage', type=str, choices=list(STAGES), default='full_sft',
                        help='训练阶段: ' + '/'.join(STAGES))
    stage = parser.parse_known_args()[0].stage
    add_train_args(parser, **STAGES[stage])
    add_stage_args(parser, stage)
    args = parser.parse_args()
    if stage == 'lora':
        args.save_weight = args.lora_name

    local_rank = init_train_env(args)
    autocast_ctx = make_autocast_ctx(args)

    if stage == 'distillation':
        lm_config = make_lm_config(args, hidden_size=args.student_hidden_size,
                                   num_hidden_layers=args.student_num_layers)
        lm_config_student = lm_config
        lm_config_teacher = make_lm_config(args, hidden_size=args.teacher_hidden_size,
                                           num_hidden_layers=args.teacher_num_layers)
        ckp_data = get_ckp_data(args, lm_config)
        wandb = init_wandb(args,
                           f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-"
                           f"Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}", ckp_data)

        model, tokenizer = init_model(lm_config, args.from_student_weight, device=args.device)
        if args.use_compile == 1:
            model = torch.compile(model)
            Logger('torch.compile enabled')
        Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
        teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')

        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

        start_epoch, start_step = load_resume_state(ckp_data, model, optimizer, scaler=scaler)
        model = wrap_ddp(model, local_rank)
        for epoch, loader, iters, st, _ in for_each_epoch(
                train_ds, args, start_epoch, start_step, train_sampler=train_sampler):
            train_epoch_distillation(epoch, loader, iters, teacher_model, lm_config_student,
                                     st, wandb, args.alpha, args.temperature)

    elif stage == 'lora':
        lm_config = make_lm_config(args)
        ckp_data = get_ckp_data(args, lm_config)
        wandb = init_wandb(args,
                           f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-"
                           f"BatchSize-{args.batch_size}-LR-{args.learning_rate}", ckp_data)

        model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
        if args.use_compile == 1:
            model = torch.compile(model)
            Logger('torch.compile enabled')
        apply_lora(model)

        total_params = sum(p.numel() for p in model.parameters())
        lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
        Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
        Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
        Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

        lora_params = []
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
                lora_params.append(param)
            else:
                param.requires_grad = False

        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
        optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

        start_epoch, start_step = load_resume_state(ckp_data, model, optimizer, scaler=scaler, strict=False)
        model = wrap_ddp(model, local_rank)
        for epoch, loader, iters, st, _ in for_each_epoch(
                train_ds, args, start_epoch, start_step, train_sampler=train_sampler):
            train_epoch_lora(epoch, loader, iters, lora_params, st, wandb)

    elif stage == 'dpo':
        lm_config = make_lm_config(args)
        ckp_data = get_ckp_data(args, lm_config)
        wandb = init_wandb(args,
                           f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}", ckp_data)

        model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
        if args.use_compile == 1:
            model = torch.compile(model)
            Logger('torch.compile enabled')
        Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

        ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
        ref_model.eval()
        ref_model.requires_grad_(False)
        Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

        train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

        start_epoch, start_step = load_resume_state(ckp_data, model, optimizer, scaler=scaler)
        model = wrap_ddp(model, local_rank)
        for epoch, loader, iters, st, _ in for_each_epoch(
                train_ds, args, start_epoch, start_step, train_sampler=train_sampler):
            train_epoch_dpo(epoch, loader, iters, ref_model, lm_config, st, wandb, args.beta)

    else:
        # pretrain / full_sft / reason：单模型 + scaler 的标准训练
        lm_config = make_lm_config(args)
        ckp_data = get_ckp_data(args, lm_config)
        wandb = init_wandb(args,
                           f"{STAGES[stage]['wandb_project']}-Epoch-{args.epochs}-"
                           f"BatchSize-{args.batch_size}-"
                           f"{'LR' if stage == 'reason' else 'LearningRate'}-{args.learning_rate}", ckp_data)

        model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
        if args.use_compile == 1:
            model = torch.compile(model)
            Logger('torch.compile enabled')

        ds_cls = PretrainDataset if stage == 'pretrain' else SFTDataset
        train_ds = ds_cls(args.data_path, tokenizer, max_length=args.max_seq_len)
        train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

        saved_indices = ckp_data.get('indices') if ckp_data else None
        start_epoch, start_step = load_resume_state(ckp_data, model, optimizer, scaler=scaler)
        model = wrap_ddp(model, local_rank)

        epoch_fn = (train_epoch_reason if stage == 'reason' else train_epoch_sft)
        for epoch, loader, iters, st, indices in for_each_epoch(
                train_ds, args, start_epoch, start_step,
                train_sampler=train_sampler, saved_indices=saved_indices):
            if stage == 'reason':
                epoch_fn(epoch, loader, iters, tokenizer, lm_config, st, wandb)
            else:
                epoch_fn(epoch, loader, iters, st, wandb, indices=indices)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
