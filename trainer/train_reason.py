import os
import sys

# 设置包名，并将项目根目录加入模块搜索路径，确保后续 import 能正确找到 model、dataset 等模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, tokenizer, lm_config, start_step=0, wandb=None):
    """
    训练一个 epoch。
    
    该函数在推理蒸馏任务中对特殊 token（<think>、</think>、<answer>、</answer>）施加更大的 loss 权重，
    以鼓励模型学习推理过程中的结构化输出格式。

    参数:
        epoch:      当前 epoch 编号（从 0 开始）
        loader:     数据加载器
        iters:      当前 epoch 的总迭代步数（含跳过的步数）
        tokenizer:  分词器
        lm_config:  模型配置对象
        start_step: 起始步数偏移（用于断点续训时跳过已经训练过的 step）
        wandb:      wandb 日志对象（可选）
    """
    # 将特殊 token 文本通过分词器编码为 token ID 列表
    # .input_ids 是分词器返回对象的一个字段，存放字符串对应的整数 token 序列（list[int]）
    # 例如 tokenizer('<think>').input_ids 可能返回 [101, 205, 102]
    # 这些整数 ID 会被模型用作输入或参与 loss 计算，后续用于对特殊 token 施加额外权重
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids
    # reduction='none' 表示不对 loss 求和或求平均，保留每个 token 的 loss 值
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # input_ids 和 labels 由 SFTDataset（dataset/lm_dataset.py）构造：
        #   - input_ids: 原始对话文本经 tokenizer 编码为 token ID 序列，不足 max_length 的部分用 pad_token_id 右侧补齐
        #   - labels: 全量初始化为 -100，然后扫描序列，仅在 assistant 的回答区间（bos_id ~ eos_id 之间）
        #     将 label 设为对应位置的 token ID；其余位置（system/user 的文本、padding）保留 -100
        #   - 这样训练时 CrossEntropyLoss 会忽略 -100 的位置，只对 assistant 回答部分计算 loss
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 根据当前进度计算余弦退火学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 前向传播
            res = model(input_ids)
            # 语言模型 loss 计算：预测下一个 token
            # 语言模型的任务是"给定前文，预测下一个 token"（自回归）。
            # 因此模型在位置 i 的 logits 应当预测位置 i+1 的真实 token。
            # 通过 shift 对齐：将 logits 去掉最后一个位置（[:-1]），label 去掉第一个位置（[1:]），
            # 这样 shift_logits[:, i, :] 与 shift_labels[:, i] 一一对应：
            #    shift_logits[:, i, :] = 模型对"第 i 个位置之后的下一个 token"的预测
            #    shift_labels[:, i]    = 第 i+1 个位置的真实 token ID
            # shift_logits: [batch, seq_len-1, vocab_size]
            shift_logits = res.logits[..., :-1, :].contiguous()
            # shift_labels: [batch, seq_len-1]
            shift_labels = labels[..., 1:].contiguous()
            # loss_fct = nn.CrossEntropyLoss(reduction='none') 定义于本函数开头
            # reduction='none' 表示不求和也不平均，保留每个 token 独立的 loss 值
            # 先将 logits 展平为 [batch*(seq_len-1), vocab_size]，labels 展平为 [batch*(seq_len-1)]
            # 计算完逐 token loss 后再 reshape 回 [batch, seq_len-1]
            # loss 形状 [batch, seq_len-1]，每个位置的交叉熵 loss
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

            # 构造 loss 掩码：忽略 label 为 -100 的位置（即 padding 部分）
            loss_mask = (shift_labels != -100).float()
            # 找出所有属于特殊 token 的位置（<think>、</think>、<answer>、</answer>）
            sp_ids = torch.isin(shift_labels.view(-1),
                                torch.tensor(start_of_think_ids + end_of_think_ids
                                             + start_of_answer_ids + end_of_answer_ids
                                             ).to(args.device))
            loss_mask_flat = loss_mask.view(-1)
            loss_mask_sum = loss_mask_flat.sum()
            # 对特殊 token 位置（<think>、</think>、<answer>、</answer>）的 loss 权重设为 10，其余非 padding 位置为 1
            # 原因：推理蒸馏（reasoning distillation）的目标是让模型学会结构化思考的格式。
            # 这些特殊 token 是定义思考/回答边界的核心标记，数量远少于普通文本 token。
            # 增大它们的权重可以迫使优化器更关注这些关键位置的预测正确性，
            # 否则模型很容易忽略这些稀疏但重要的标记，导致生成的思考过程格式混乱。
            loss_mask_flat[sp_ids] = 10
            loss_mask = loss_mask_flat.view(shift_labels.size())
            # 加权 loss 求和并归一化（按非 padding token 数做平均）
            logits_loss = (loss * loss_mask).sum() / loss_mask_sum
            # 总 loss = 加权交叉熵 loss + 辅助 loss（如 MoE 的负载均衡 loss）
            loss = logits_loss + res.aux_loss
            # 梯度累积：将 loss 除以累积步数，使得多个 micro-batch 的梯度平均后等效于一个全局 batch
            loss = loss / args.accumulation_steps

        # 反向传播，累积梯度
        scaler.scale(loss).backward()

        # 当累积步数达到 accumulation_steps 时，更新模型参数
        if (step + 1) % args.accumulation_steps == 0:
            # 对梯度解缩放（混合精度训练时，梯度是 scaled 的，需要 unscaled 后才能 clip）
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 优化器更新（内部会判断是否使用混合精度）
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志打印
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = logits_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            # 预计剩余时间（分钟）
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 定期保存模型权重
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 从 DDP 或 torch.compile 中取出原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 将权重转为半精度再保存，减小存储空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整的 checkpoint（含优化器状态、scaler 状态等，用于断点续训）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    # ========================== 命令行参数 ==========================
    parser = argparse.ArgumentParser(description="MiniMind Reasoning Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='reason', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=720, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl", help="推理蒸馏数据路径")
    parser.add_argument('--from_weight', default='dpo', type=str, help="基于哪个权重训练，默认dpo")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Reasoning", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化分布式环境和随机种子 ==========
    local_rank = init_distributed_mode()
    # 如果启动了分布式（多卡训练），自动将设备设置为当前进程对应的 GPU
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 不同进程使用不同随机种子，保证数据打乱不重复
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 创建保存目录、初始化模型配置、检测续训 checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果启用断点续训（from_resume==1），自动从 ../checkpoints 目录加载保存的训练状态
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度上下文 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 下不支持 amp，使用空上下文；GPU 下启用自动混合精度
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 初始化 wandb 日志（仅主进程） ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 如果从 checkpoint 恢复，尝试沿用之前的 wandb run id
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型、分词器、数据集、优化器、梯度缩放器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式训练时使用 DistributedSampler 自动分配数据分片
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler：float16 训练时用于动态缩放 loss，防止梯度下溢
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从 checkpoint 恢复模型、优化器、scaler 状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 用 DistributedDataParallel 包装模型（多卡训练） ==========
    if dist.is_initialized():
        # 忽略 RoPE 位置编码中的频率缓存，它们在各卡之间相同，无需同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 多 epoch 训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式模式下调用 set_epoch 确保每个 epoch 数据打乱方式不同
        train_sampler and train_sampler.set_epoch(epoch)
        # 单机模式下用随机索引实现不打乱
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果要从某个 step 续训，计算需要跳过的样本数
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, tokenizer, lm_config, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), tokenizer, lm_config, 0, wandb)
    
    # ========== 9. 训练结束，销毁分布式进程组 ==========
    if dist.is_initialized(): dist.destroy_process_group()
