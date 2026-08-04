import os
import sys

sys.path.insert(0, os.getcwd())

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from scripts.Model.model_minimind import MiniMindConfig
from scripts.Dataset.lm_dataset import SFTDataset
from scripts.Model.model_lora import save_lora, apply_lora
from scripts.Trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
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
            model.eval()
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer,
                          scaler=scaler, epoch=epoch, step=step, wandb=wandb,
                          save_dir='checkpoints')
            model.train()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir",        type=str, default="models/lora",          help="模型保存目录")
    parser.add_argument("--lora_name",       type=str, default="lora_identity",        help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs",          type=int, default=50,                     help="训练轮数")
    parser.add_argument("--batch_size",      type=int, default=32,                     help="每个step的batch size")
    parser.add_argument("--learning_rate",   type=float, default=1e-4,                help="初始学习率")
    parser.add_argument("--device",          type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype",           type=str, default="bfloat16",             help="混合精度类型（bfloat16 / float16）")
    parser.add_argument("--num_workers",     type=int, default=8,                      help="DataLoader 子进程数")
    parser.add_argument("--accumulation_steps", type=int, default=1,                   help="梯度累积步数（模拟更大的batch size）")
    parser.add_argument("--grad_clip",       type=float, default=1.0,                  help="梯度裁剪最大范数")
    parser.add_argument("--log_interval",    type=int, default=10,                     help="日志打印间隔（step）")
    parser.add_argument("--save_interval",   type=int, default=1000,                   help="模型保存间隔（step）")
    parser.add_argument('--hidden_size',       default=512,     type=int,              help="Transformer 隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8,       type=int,              help="Transformer 层数")
    parser.add_argument('--max_seq_len',       default=340,     type=int,              help="训练的最大序列长度（中文约1.5~1.7字符/token）")
    parser.add_argument('--use_moe',           default=0,       type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path",       type=str, default="resource/minimind_dataset/lora_identity.jsonl", help="LoRA 训练数据（jsonl 格式）")
    parser.add_argument('--from_weight',     default='full_sft', type=str,             help="基座权重名称（从该 checkpoint 加载初始参数）")
    parser.add_argument('--from_resume',     default=0,  type=int, choices=[0, 1],    help="是否自动检测 checkpoint 并续训（0=否，1=是）")
    parser.add_argument("--use_wandb",       action="store_true",                      help="启用 wandb / swanlab 实验记录")
    parser.add_argument("--wandb_project",   type=str, default="MiniMind-LoRA",        help="wandb 项目名称")
    parser.add_argument("--use_compile",     default=0,  type=int, choices=[0, 1],    help="是否启用 torch.compile 加速（0=否，1=是）")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (f"MiniMind-LoRA-{args.lora_name}-"
                          f"Epoch-{args.epochs}-BatchSize-{args.batch_size}-"
                          f"LR-{args.learning_rate}")
        wandb.init(project=args.wandb_project, name=wandb_run_name,
                   id=wandb_id, resume=resume)

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

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
