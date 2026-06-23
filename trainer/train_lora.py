import os
import sys

# 将项目根目录加入 sys.path，确保后续 import 能够找到 model、dataset、trainer 等顶层包
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
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    训练一个完整 epoch。

    ⚠️ 注意：本函数依赖模块级全局变量（在 __main__ 中初始化）：
      model        — 插入 LoRA 后的模型（init_model + apply_lora）
      optimizer    — AdamW 优化器（仅含 lora_params）
      scaler       — GradScaler（fp16 时启用）
      args         — 命令行参数命名空间
      lm_config    — MiniMindConfig 模型配置
      autocast_ctx — 混合精度自动上下文

    这些变量未通过参数传递，而是利用 Python 的 LEGB 作用域规则：
      函数内找不到局部定义 → 向 enclosing 函数（无）→ 再向模块全局作用域查找。
    这是训练脚本中常见的简化写法（省去在参数列表里写一堆长参数）。

    Args:
        epoch:       当前轮次（0-based）
        loader:      数据加载器
        iters:       该 epoch 的总迭代步数（=len(loader)+skip）
        lora_params: 待优化的 LoRA 参数列表，传给 optimizer
        start_step:  起始 step 偏移（用于断点续训时跳过已完成的 batch）
        wandb:       可选，wandb / swanlab 日志对象
    """
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将输入和标签搬到指定设备
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 根据当前进度计算余弦退火学习率，并更新 optimizer
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度上下文（bf16 / fp16 / fp32）
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # res.loss: 交叉熵损失；res.aux_loss: MoE负载均衡辅助损失（非 MoE 时为 0）
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps          # 梯度累积：将 loss 缩放到单步等效

        # 反向传播（scaler 自动处理 fp16 grad scaling）
        scaler.scale(loss).backward()

        # 每 accumulation_steps 步更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)                     # 反缩放梯度，供 clip 使用
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)  # 梯度裁剪，防止梯度爆炸
            # 整个训练只更新 LoRA 参数，基座模型完全冻结：
            #   第 134-139 行：非 LoRA 参数 requires_grad=False → 不计算梯度；
            #   只有含 "lora" 的参数 requires_grad=True → 只有它们有梯度。
            # lora_params 按此规则收集 → 等价于"全部有梯度的参数"集合。
            # 因此裁剪 lora_params 就等于裁剪模型中所有会更新梯度的部分，没有遗漏。
            scaler.step(optimizer)                         # 优化器更新参数
            scaler.update()                                # 更新 scaler 的缩放因子
            optimizer.zero_grad(set_to_none=True)          # 清空梯度（set_to_none 比 zero_grad 更省显存）

        # 日志打印
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps          # 还原真实 loss
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss          # 纯 logits 交叉熵损失
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60  # 预估剩余时间（分钟）
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                   f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, '
                   f'epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss,
                           "aux_loss": current_aux_loss, "learning_rate": current_lr,
                           "epoch_time": eta_min})

        # 定期保存 LoRA 权重和完整 checkpoint
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            # LoRA 只保存低秩适配矩阵（A 和 B），不保存基座模型参数
            save_lora(model, lora_save_path)
            # 完整 checkpoint：保存 optimizer / scaler / epoch / step 等信息，用于续训
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer,
                          scaler=scaler, epoch=epoch, step=step, wandb=wandb,
                          save_dir='../checkpoints')
            model.train()

        # 及时释放中间变量，减少显存占用
        del input_ids, labels, res, loss


if __name__ == "__main__":
    # ========================= 命令行参数定义 =========================
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir",        type=str, default="../out/lora",          help="模型保存目录")
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
    parser.add_argument("--data_path",       type=str, default="../dataset/lora_identity.jsonl", help="LoRA 训练数据（jsonl 格式）")
    parser.add_argument('--from_weight',     default='full_sft', type=str,             help="基座权重名称（从该 checkpoint 加载初始参数）")
    parser.add_argument('--from_resume',     default=0,  type=int, choices=[0, 1],    help="是否自动检测 checkpoint 并续训（0=否，1=是）")
    parser.add_argument("--use_wandb",       action="store_true",                      help="启用 wandb / swanlab 实验记录")
    parser.add_argument("--wandb_project",   type=str, default="MiniMind-LoRA",        help="wandb 项目名称")
    parser.add_argument("--use_compile",     default=0,  type=int, choices=[0, 1],    help="是否启用 torch.compile 加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化分布式训练环境与随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 不同进程使用不同 seed 避免数据加载冲突
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 创建保存目录、构造模型配置、尝试加载已有 checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度自动上下文 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 不支持 amp，使用 nullcontext（空上下文，等价于全精度）
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 初始化 wandb / swanlab 实验跟踪 ==========
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

    # ========== 5. 创建模型、加载基座权重、插入 LoRA 模块 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 在目标线性层（通常为 Q/K/V/O/proj）上并行叠加 LoRA A/B 两个低秩矩阵：
    #   原始权重被冻结（requires_grad=False），前向时 output = Wx + BAx。
    # 为什么用两个矩阵（B∈ℝ^{d×r}, A∈ℝ^{r×k}）而不是一个？
    #   LoRA 的核心假设：微调带来的权重变化 ΔW = W_微调后 - W_预训练 是"低秩"的。
    #   低秩 ≈ 矩阵的信息集中在少量独立方向上，大量维度是冗余的。
    #   拿本项目举例：hidden_size=512，LoRA 默认 r=8（具体值见 model_lora.py）。
    #   - 若用一个矩阵 P∈ℝ^{512×512}：参数量 = 512×512 = 262,144
    #   - 用 BA 分解：B∈ℝ^{512×8}, A∈ℝ^{8×512}，参数量 = 512×8 + 8×512 = 8,192
    #   对比：8,192 ÷ 262,144 = 3.1%，节省了约 97% 的参数。
    #   这个节省来自"微调 ΔW 是低秩的"这一假设：微调只需在 8 个关键方向上调整，
    #   其余 504 个维度基本不变。如果 ΔW 实际需要满秩（所有方向都要大调），
    #   LoRA 的效果就会变差——实践表明对大部分下游任务这个假设是成立的。
    #   这就是 LoRA 高效微调的本质：用低秩分解参数化增量更新，大幅减少可训练参数量。
    # 如何控制 LoRA 参数量？
    #   - rank（秩）是最直接的旋钮：r 越大 → 可训练参数越多 → 表达能力越强，但显存/时间成本也越高
    #   - 本项目默认 r=8（写死在 apply_lora(model) 的默认参数中，见 model_lora.py:63）
    #   - 修改方式：调用 apply_lora(model, rank=16) 或 apply_lora(model, rank=4)
    #   - rank 的典型选择：4~64，一般 8 或 16 在效果和效率间取得较好平衡
    #   - 此外还可以控制"哪些层插 LoRA"（当前实现是对所有方阵 Linear 层都插）
    apply_lora(model)

    # 统计参数量并打印，直观展示 LoRA 的高效性：
    #   让用户一眼看清"只训练了不到 1% 的参数就把任务做好了"。
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    # 冻结基座所有参数，仅 LoRA 参数可训练
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # ========== 6. 数据集、分布式采样器、GradScaler、优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 仅在 fp16 下启用；bf16 自动处理数值范围，无需 scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 仅优化 LoRA 参数，极大减少优化器状态量
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    # ========== 7. 续训：从 checkpoint 恢复模型 / 优化器 / scaler 状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 8. 用 DistributedDataParallel 封装模型 ==========
    if dist.is_initialized():
        # 忽略 RoPE 缓存 buffer（不需要同步）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 9. 多 epoch 训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)  # 保证分布式下每个 epoch 数据重排不同
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()  # 单机下的随机排列
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler 支持跳过前 skip 个 batch，用于续训
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)

    # ========== 10. 清理分布式进程组 ==========
    if dist.is_initialized():
        dist.destroy_process_group()


# ====================================================================
# 📚 练习题 — LoRA 微调核心概念自测
# ====================================================================
#
# 1. (概念) LoRA 为什么用两个矩阵 (B, A) 而不是一个矩阵 P？
#
# 2. (计算) hidden_size=512, rank=4, 一个 Linear 层插入 LoRA 后，
#    该层 LoRA 部分的可训练参数量是多少？相比 512×512 的满矩阵，
#    节省了多少？
#
# 3. (阅读) train_epoch() 里直接用了 model、optimizer、scaler 等
#    变量却没有报错，为什么？
#
# 4. (推理) 把 rank 从 8 改为 64，会带来哪些影响？
#    （参数量、显存、训练时间、表达能力等角度）
#
# 5. (改错) 以下代码有什么问题？
#     trainable_params = [p for p in model.parameters() if p.requires_grad]
#     all_params = list(model.parameters())
#     torch.nn.utils.clip_grad_norm_(all_params, 1.0)
# ====================================================================
#
#
# ========================= 参考答案 =========================
#
# Q1: 用一个矩阵 P∈ℝ^{d×k} 参数量为 d×k，更新仍是满秩的，没有节省。
#    分解为 B∈ℝ^{d×r}、A∈ℝ^{r×k}，参数量降为 r×(d+k)，
#    利用"微调 ΔW 是低秩的"这一假设，用远少于原始权重的参数
#    来参数化增量更新 —— 这就是 LoRA 高效微调的本质。
#
# Q2:
#   B: 512×4 = 2,048 个参数
#   A: 4×512 = 2,048 个参数
#   合计: 4,096 个参数
#   满矩阵 512×512 = 262,144 个参数
#   节省: 1 - 4096/262144 = 98.44%
#
# Q3: Python 的 LEGB 作用域规则。
#    train_epoch 内部找不到局部定义 → 向模块全局作用域查找。
#    model / optimizer / scaler / args / lm_config / autocast_ctx
#    都是在 if __name__ == "__main__" 中初始化的全局变量。
#
# Q4: rank 从 8 → 64：
#   - 每层 LoRA 参数量变为原来的 8 倍（512×64+64×512=65,536）
#   - 总参数量、优化器状态、显存占用和训练时间均显著增加
#   - 表达能力更强（更高 rank 能捕捉更复杂的 ΔW），
#     但也更容易过拟合，对小数据集未必更好
#   - 实践中 rank=8~16 通常在效果和效率间取得较好平衡
#
# Q5: 有 bug。all_params 包含了 requires_grad=False 的参数，
#    这些参数没有梯度（.grad = None），
#    clip_grad_norm_ 遇到 None 梯度会报错。
#    正确做法是只传 trainable_params（有梯度的参数），
#    实际上本脚本里 lora_params 就是正确的集合。
# ====================================================================

