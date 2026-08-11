"""distillation 阶段：知识蒸馏（学生 CE 损失 + 教师 KL 损失按 alpha 加权）。"""
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DistributedSampler

from scripts.Dataset.lm_dataset import SFTDataset
from scripts.Trainer.train_common import (
    TrainCtx, init_train_env, make_lm_config, make_autocast_ctx, init_wandb,
    get_ckp_data, load_resume_state, wrap_ddp, for_each_epoch, save_checkpoint,
)
from scripts.Trainer.trainer_utils import get_lr, Logger, is_main_process, init_model

STAGE_DEFAULTS = dict(save_weight='full_dist', epochs=6, batch_size=32, learning_rate=5e-6,
                      save_interval=100, max_seq_len=340,
                      data_path='resource/minimind_dataset/sft_t2t_mini.jsonl',
                      wandb_project='MiniMind-Distillation', from_weight=None)


def add_args(parser):
    parser.add_argument('--student_hidden_size', default=512, type=int, help='学生模型隐藏层维度')
    parser.add_argument('--student_num_layers', default=8, type=int, help='学生模型隐藏层数量')
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help='教师模型隐藏层维度')
    parser.add_argument('--teacher_num_layers', default=16, type=int, help='教师模型隐藏层数量')
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help='学生模型基于哪个权重')
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help='教师模型基于哪个权重')
    parser.add_argument('--alpha', default=0.5, type=float, help='CE损失权重，总损失=alpha*CE+(1-alpha)*KL')
    parser.add_argument('--temperature', default=1.5, type=float, help='蒸馏温度（推荐范围1.0-2.0）')
    return parser


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl


def train_epoch_distillation(ctx, epoch, loader, iters, start_step, alpha=0.0, temperature=1.0):
    """知识蒸馏：学生 CE 损失 + 教师 KL 损失按 alpha 加权。"""
    args, model, optimizer, scaler = ctx.args, ctx.model, ctx.optimizer, ctx.scaler
    autocast_ctx, lm_config_student, teacher_model, wandb = (
        ctx.autocast_ctx, ctx.lm_config, ctx.teacher_model, ctx.wandb)
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


def run(args):
    local_rank = init_train_env(args)
    lm_config_student = make_lm_config(args, hidden_size=args.student_hidden_size,
                                       num_hidden_layers=args.student_num_layers)
    lm_config_teacher = make_lm_config(args, hidden_size=args.teacher_hidden_size,
                                       num_hidden_layers=args.teacher_num_layers)
    ckp_data = get_ckp_data(args, lm_config_student)
    wandb = init_wandb(
        args,
        f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-"
        f"Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}",
        ckp_data,
    )

    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
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

    ctx = TrainCtx(args=args, lm_config=lm_config_student, model=model, tokenizer=tokenizer,
                   optimizer=optimizer, scaler=scaler, teacher_model=teacher_model,
                   autocast_ctx=make_autocast_ctx(args), wandb=wandb)

    start_epoch, start_step = load_resume_state(ckp_data, model, optimizer, scaler=scaler)
    model = wrap_ddp(model, local_rank)

    for epoch, loader, iters, st, _ in for_each_epoch(
            train_ds, args, start_epoch, start_step, train_sampler=train_sampler):
        train_epoch_distillation(ctx, epoch, loader, iters, st,
                                 alpha=args.alpha, temperature=args.temperature)

    if dist.is_initialized():
        dist.destroy_process_group()
