import os
import sys

# 将项目根目录加入 sys.path，确保后续 import 能够找到 model、dataset、trainer 等顶层包
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将模型输出的 logits 转换为每个 token 的 log probability。

    与 SFT 不同，DPO 需要的是「概率的对数」而非交叉熵 loss。
    SFT 中 nn.CrossEntropyLoss 内部已经做了 log_softmax + nll_loss，
    而 DPO 需要显式拿到 log_probs 才能计算 chosen/rejected 之间的差值。

    Args:
        logits: shape (batch_size, seq_len, vocab_size) — 模型原始输出
        labels: shape (batch_size, seq_len) — 目标 token ID

    Returns:
        log_probs_per_token: shape (batch_size, seq_len)
            每个 token 位置在其真实 label 上的 log probability
    """
    # 在 vocab 维度上做 log_softmax，得到每个 token 在每个 vocab 上的 log prob
    # log_softmax(x_i) = log( e^{x_i} / Σ_j e^{x_j} ) = x_i - log(Σ_j e^{x_j})
    # 相比直接算 softmax 再取 log，log_softmax 在数值上更稳定（避免 exp 溢出）
    log_probs = F.log_softmax(logits, dim=2)
    # 用 gather 从 vocab 维取出 label 对应位置的 log prob
    # labels.unsqueeze(2) → (B, L, 1)，gather 后 squeeze 回 (B, L)
    #
    # 举个具体例子帮助理解：
    #   假设 vocab_size=3, log_probs 的某个位置是 [log(0.2), log(0.5), log(0.3)]
    #   该位置的 label=1（"狗"），gather 操作就是：
    #     在最后一维（dim=2）上，取 index=1 对应的值 → log(0.5)
    #   相当于：从 vocab 分布中"捞"出真实 label 对应的那个 log_prob。
    #
    # 为什么需要 squeeze？
    #   gather 前 labels 是 (B, L)，unsqueeze(2) 变成 (B, L, 1)
    #   gather 后得到 (B, L, 1)，squeeze(-1) 去掉多余维度变回 (B, L)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算 DPO（Direct Preference Optimization）损失。

    DPO 的核心思想（对比 RLHF-PPO）：

      传统 RLHF 的流程（用具体数据讲）：
        你有 1000 条偏好数据（chosen/rejected 对）。

        【阶段一：训练 Reward Model】
          把 chosen 和 rejected 同时喂给一个初始模型，让它学会打分。
          训练目标：对 chosen 打高分，对 rejected 打低分。
          产物：一个单独的 Reward Model（评分模型）。

        【阶段二：PPO 优化】
          现在你有 4 个模型同时在内存里：
            ① Policy 模型（要训练的，不断更新）
            ② Reference 模型（冻结的，就是 SFT 阶段训出来的初始模型，
               与 Policy 模型初始权重相同但不再更新，
               用来衡量 policy 跑了多远，防止学偏）
            ③ Reward Model（阶段一训练出来的，给回答打分的）
            ④ Critic 模型（PPO 需要的，预估未来收益的）

          PPO（Proximal Policy Optimization，近端策略优化）是 OpenAI 提出的强化学习算法。
          核心思想：每次更新策略时不要步子太大，防止学崩了。
          用学骑自行车类比：你每次调整身体重心只能微调，不能猛地一歪——否则直接摔了。
          PPO 通过 clip 机制限制每次更新幅度，确保训练稳定。

          每步流程：
            1. Policy 生成一段回答
            2. Reward Model 给这个回答打分（比如 0.7 分）
            3. Reference 模型算当前 policy 偏离了多少（KL 散度）
            4. 综合得分和 KL 散度算出"优势值"
            5. Critic 模型（又是一个神经网络，通常与 Policy 共享主干但加一个
               价值头（value head）输出标量分值）预估期望优势值。
               Critic 的价值头是随机初始化的（没有预训练权重），
               在 PPO 训练中从零开始学"预估得分"——就像让一个人从没批过卷子
               的人边看边学：一开始估得完全不准，但通过不断对比实际得分，
               逐步学会预估。这就是"批评家"的成长过程。

               Critic 的更新目标：让预估值尽量接近实际优势值（MSE loss）。
               实际优势值（Reward Model 打分 + KL 惩罚）与 Critic 预估值的
               差值就是 PPO 的 Advantage，用于更新 Policy。
               Critic 自身也在训练中不断更新，让预估越来越准。
               这就是 Actor-Critic 架构：Policy 是演员（负责表演），
               Critic 是评委（负责挑刺），两者互相促进。
            6. 更新 Policy 模型

          问题：4 个模型占用巨大显存，PPO 超参（clip_epsilon、GAE lambda 等）
          极其敏感，调参周期长。

        DPO 的做法：
          跳过所有中间环节——直接从偏好对算出 loss：
            1. 取一条数据 (chosen, rejected)
            2. policy 和 ref 分别算 chosen/rejected 的 log_probs
            3. 计算相对优势：(chosen_policy - rejected_policy) - (chosen_ref - rejected_ref)

          用品酒师来类比理解这个公式：

            假设 Ref 模型和 Policy 模型是两个品酒师，品尝两杯酒（chosen 和 rejected），
            每杯打分 0~100，分差代表"偏好程度"。

            Ref 品酒师（原来的你）：
              chosen  打了 70 分
              rejected 打了 30 分
              分差 = 70 - 30 = 40（Ref 已经倾向于 chosen）
            
            Policy 品酒师（现在的你）：
              chosen  打了 90 分
              rejected 打了 10 分
              分差 = 90 - 10 = 80（Policy 更倾向于 chosen）

            相对优势 = (90 - 10) - (70 - 30) = 80 - 40 = 40

            这个 +40 就是 Policy 相比 Ref「额外多偏好了 chosen 多少」。

            为什么负值时 loss 会变大？看 loss 公式：loss = -log σ(β × 相对优势)
            代数值算一遍（设 β=0.1）：
              相对优势 = +40 → σ(0.1×40)=σ(4)≈0.982 → log≈-0.018 → loss=0.018 ✅
              相对优势 = -20 → σ(0.1×(-20))=σ(-2)≈0.119 → log≈-2.13 → loss=2.13 ❌
            因为 -log σ(x) 这个函数，x 越大 loss 越小，x 越小 loss 越大。
            所以 DPO loss 天然地 push 模型让相对优势尽可能大。

            如果 Policy 分差小于 Ref（比如 Policy 只偏爱 chosen 20 分）：
              相对优势 = 20 - 40 = -20（负值 → loss 很大 → 被惩罚 → 模型调整方向）

            如果 Policy 两个都打高分（chosen=85, rejected=80）：
              分差 = 5，相对优势 = 5 - 40 = -35（虽然 abs 分数高了，但没有拉开差距，
              模型只是在无脑增大所有 token 概率，这对 DPO 来说是坏事）

            DPO 不关心"Policy 对 chosen 的绝对分数有多高"，
            只关心"Policy 比 Ref 更偏好 chosen 相对 rejected 的程度"。
            这防止了 model collapse（模型把所有回答概率都无限增大）。
            4. loss = -log σ(β × 相对优势)
          只需 2 个模型（policy + ref），没有 Reward Model 和 Critic，
          没有 KL 散度计算，没有优势估计，没有 PPO 的 clip 机制。

      用考试成绩来类比：
        RLHF：先请一个评分老师（Reward Model）学会判卷 →
              然后让考生（Policy 模型）做题 → 老师打分 → PPO 调整学习方法 →
              还需要一个教学督导（Critic）预估"这题如果多检查一遍能提多少分"。
              流程繁琐，牵涉角色多，协调困难。

        DPO：直接给学生两份答案（chosen = 学霸的，rejected = 学渣的）→
             学生自己对比两份答案的差异来调整。
             只有一个老师（Reference 模型）站在旁边告诉你"原来的你是什么水平"，
             不要额外打分，不要教学督导。

    DPO loss 公式（源自论文 https://arxiv.org/abs/2305.18290）：
      L = -E [ log σ( β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)) ) ]

    直观理解：
      - π(y_w|x) / π_ref(y_w|x)：policy 对 chosen 回答的相对概率（相比 ref 提升了多少）
      - π(y_l|x) / π_ref(y_l|x)：policy 对 rejected 回答的相对概率
      - 让 chosen 的相对概率尽可能大，rejected 的相对概率尽可能小
      - β 控制这个"拉开差距"的力度

    与 SFT loss 的对比：
      SFT: L = -log π(y|x)
           x = 输入序列（prompt，即用户的指令）
           y = 目标序列（label，即期望的 assistant 回答）
           π(y|x) = 模型在看到输入 x 时，生成目标 y 的概率
           SFT 的优化目标就是增大这个概率 —— 模型学会「当用户问 x 时，要回答 y」。
           但如果你喂给 SFT 的数据里全是好回答，它确实能学会生成好回答。
           但 SFT 无法区分"更好的"和"更差的"——它只知道增大 token 概率，
           不会主动降低坏回答的概率。

      DPO: L = -log σ(β * (优势_chosen - 优势_rejected))
           DPO 不关心绝对概率有多大，只关心"模型是否更偏好 chosen 胜过 rejected"。
           这相当于在 SFT 的基础上加了一个"拉高踩低"的调整：
           让 chosen 的概率相对更高，rejected 的概率相对更低。
           它让你从 SFT 的"学会说话"走向"学会说人话"。

      ⭐ 打个比方：
        SFT 就像把一本优秀作文集反复读给模型听，让它模仿。
        DPO 就像把好作文和差作文同时摆在模型面前，问它"哪个更好？为什么？"
          —— 模型不仅知道好作文长什么样，还知道好作文好在哪里、差作文差在哪里。

    Args:
        ref_log_probs:    shape (batch_size, seq_len) — ref 模型的 log_probs
        policy_log_probs: shape (batch_size, seq_len) — policy 模型的 log_probs
        mask:             shape (batch_size, seq_len) — 1 表示有效 token，0 表示 padding
        beta:             温度超参，控制"拉开差距"的力度（默认 0.1）

    Returns:
        标量 loss（batch 内求平均）
    """
    # ⚠️ 注意：本函数也依赖模块级全局变量（与 train_epoch 同理），
    #   包括 model, optimizer, scaler, args, lm_config, autocast_ctx 等。
    #   详见 train_epoch 开头的注解。

    # 计算每个序列的有效长度（排除 padding），并防止除零
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    # 对每个序列，在有效 token 范围内求 log_probs 的平均值
    # 这样不同长度序列的 log_probs 具有可比性
    #
    # 举个数值例子（假设 batch_size=2, seq_len=4）：
    #   mask = [[1, 1, 1, 0],      # 序列1 有效3个token
    #           [1, 1, 0, 0]]      # 序列2 有效2个token
    #   log_probs = [[-0.1, -0.2, -0.3, -9.9],   # 序列1 每token的log_prob
    #                [-0.4, -0.5, -9.9, -9.9]]    # 序列2
    #   乘以 mask 后 padding 位置归零：
    #   [[-0.1, -0.2, -0.3, 0],    → sum/3 = (-0.6)/3 = -0.2
    #    [-0.4, -0.5, 0, 0]]       → sum/2 = (-0.9)/2 = -0.45
    #   如果不除以长度，长序列的 sum 天然比短序列小很多，无法比较。
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将 chosen 和 rejected 数据分开
    # 数据拼接策略：batch 前半 = chosen，后半 = rejected（详见 train_epoch）
    #
    # 接上面的例子，现在 shape 从 (4,) 变成了 (2,)：
    #   ref_log_probs = [-0.20, -0.45]          ← 这是两条序列在 ref 下的平均 log_prob
    #   policy_log_probs = [-0.15, -0.50]       ← 这是两条序列在 policy 下的平均 log_prob
    #   但此时 batch 里是 [seq_chosen, seq_rejected]（chosen 在前，rejected 在后）
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]       # 前半：chosen 的 ref log_probs
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]       # 后半：rejected 的 ref log_probs
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2] # 前半：chosen 的 policy log_probs
    reject_policy_log_probs = policy_log_probs[batch_size // 2:] # 后半：rejected 的 policy log_probs

    # 现在我们来算一条完整数据（一个 chosen/rejected 对）：
    #   ref 对 chosen 的 log_prob = -0.20
    #   ref 对 rejected 的 log_prob = -0.45
    #   policy 对 chosen 的 log_prob = -0.15
    #   policy 对 rejected 的 log_prob = -0.50
    #
    # policy 对 chosen/rejected 的偏好差距
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    #   pi_logratios = (-0.15) - (-0.50) = +0.35
    #   → Policy 认为 chosen 比 rejected 好 0.35
    #
    # ref 对 chosen/rejected 的偏好差距（作为基线）
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    #   ref_logratios = (-0.20) - (-0.45) = +0.25
    #   → Ref 认为 chosen 比 rejected 好 0.25
    #
    # 相对优势：policy 相比 ref 额外多偏好了 chosen 多少
    logits = pi_logratios - ref_logratios
    #   logits = 0.35 - 0.25 = +0.10
    #   → Policy 比 Ref 额外多偏好 chosen 0.10（方向正确，数值较小）
    #
    # DPO loss：-log σ(β * advantage)
    #   σ(0.1 × 0.10) = σ(0.01) ≈ 0.5025
    #   loss = -log(0.5025) ≈ 0.69（还有优化空间，需要继续拉大差距）
    #
    # 优化目标：让 logits 尽量大 → σ 趋近 1 → loss 趋近 0
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    训练一个完整 epoch。

    ⚠️ 注意：本函数依赖模块级全局变量（在 __main__ 中初始化）：
      model        — policy 模型（将被优化）
      optimizer    — AdamW 优化器（优化 policy 全部参数）
      scaler       — GradScaler（fp16 时启用）
      args         — 命令行参数命名空间
      autocast_ctx — 混合精度自动上下文

    这些变量未通过参数传递，而是利用 Python 的 LEGB 作用域规则：
      函数内找不到局部定义 → 向模块全局作用域查找。
    这是训练脚本中常见的简化写法。

    Args:
        epoch:       当前轮次（0-based）
        loader:      数据加载器
        iters:       该 epoch 的总迭代步数
        ref_model:   冻结的 reference 模型
        lm_config:   MiniMindConfig 模型配置
        start_step:  起始 step 偏移（断点续训用）
        wandb:       可选，wandb/swanlab 日志对象
        beta:        DPO loss 的温度参数
    """
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        # ==================== 数据准备 ====================
        # DPO 每一条数据包含 chosen（好回答）和 rejected（差回答）各一组序列
        # x: input token IDs, y: label token IDs (x 右移一位), mask: loss mask
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        # 关键策略：将 chosen 和 rejected 拼成一个大 batch
        # 这样一次 forward 就能同时算出 chosen 和 rejected 的 logits
        # 前一半 = chosen，后一半 = rejected，后续 dpo_loss 按 batch 分半
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # ==================== 学习率调度 ====================
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ==================== 前向传播 ====================
        with autocast_ctx:
            # 先跑 ref_model（不需要梯度），拿到 ref_log_probs 作为基线
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 再跑 policy_model（需要梯度），拿到 policy_log_probs
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            # ⭐ DPO loss：对比 policy 和 ref 在 chosen/rejected 上的概率差异
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # 总 loss = DPO loss + MoE 辅助 loss（非 MoE 时 aux_loss = 0）
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps  # 梯度累积缩放

        # ==================== 反向传播 ====================
        scaler.scale(loss).backward()

        # ==================== 梯度累积步 ====================
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 注意：DPO 是全参数训练（不像 LoRA 只裁剪 lora_params）
            # 因为 DPO 中所有参数 requires_grad=True，model.parameters() 就是全部有梯度的参数
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ==================== 日志 ====================
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

        # ==================== 保存 checkpoint ====================
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 扒壳：兼容 DDP 和 torch.compile 包装
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存为 fp16 格式以节省存储空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整 checkpoint（含 optimizer/scaler 状态）用于续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model,
                          optimizer=optimizer, scaler=scaler, epoch=epoch,
                          step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # ==================== 显存清理 ====================
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    # ========================= 命令行参数定义 =========================
    # 与 SFT/LoRA 相比，DPO 的新增参数：
    #   --beta: DPO loss 的温度超参
    #   --learning_rate: 极小（4e-8），避免破坏 SFT 已学知识
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir",          type=str, default="../out",             help="模型保存目录")
    parser.add_argument('--save_weight',       default='dpo',            type=str,    help="保存权重的前缀名")
    parser.add_argument("--epochs",            type=int, default=1,                    help="训练轮数（DPO 通常只需 1 epoch）")
    parser.add_argument("--batch_size",        type=int, default=4,                    help="batch size（注意 chosen+rejected 拼接后显存翻倍）")
    parser.add_argument("--learning_rate",     type=float, default=4e-8,               help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device",            type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype",             type=str, default="bfloat16",           help="混合精度类型")
    parser.add_argument("--num_workers",       type=int, default=8,                    help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1,                   help="梯度累积步数")
    parser.add_argument("--grad_clip",         type=float, default=1.0,                help="梯度裁剪阈值")
    parser.add_argument("--log_interval",      type=int, default=100,                  help="日志打印间隔")
    parser.add_argument("--save_interval",     type=int, default=100,                  help="模型保存间隔")
    parser.add_argument('--hidden_size',       default=512,     type=int,              help="Transformer 隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8,       type=int,              help="Transformer 层数")
    parser.add_argument('--max_seq_len',       default=1024,    type=int,              help="训练的最大序列长度（DPO 通常需要比 SFT 更长的序列）")
    parser.add_argument('--use_moe',           default=0,       type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path",         type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径（含 chosen/rejected 对）")
    parser.add_argument('--from_weight',       default='full_sft', type=str,            help="基于哪个权重训练（通常是 full_sft 或 lora_xxx）")
    parser.add_argument('--from_resume',       default=0,  type=int, choices=[0, 1],  help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta',              default=0.1, type=float,                help="DPO loss 中的 beta 温度参数")
    parser.add_argument("--use_wandb",         action="store_true",                     help="是否使用 wandb / swanlab")
    parser.add_argument("--wandb_project",     type=str, default="MiniMind-DPO",        help="wandb 项目名")
    parser.add_argument("--use_compile",       default=0,  type=int, choices=[0, 1],  help="是否使用 torch.compile 加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化分布式训练环境与随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 创建保存目录、构造模型配置、检测续训 checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度自动上下文 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 初始化 wandb / swanlab 实验跟踪 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (f"MiniMind-DPO-Epoch-{args.epochs}-"
                          f"BatchSize-{args.batch_size}-LR-{args.learning_rate}")
        wandb.init(project=args.wandb_project, name=wandb_run_name,
                   id=wandb_id, resume=resume)

    # ========== 5. 定义 policy 模型和 reference 模型 ==========
    # ⭐ DPO 与 SFT/LoRA 最核心的差异：双模型架构
    #
    #   policy 模型 (model)：
    #     将被训练的模型，参数 requires_grad=True，传给 optimizer。
    #     这是最终要使用的模型。
    #
    #   reference 模型 (ref_model)：
    #     与 policy 完全相同的初始权重（都从 --from_weight 加载），
    #     但被冻结（requires_grad=False, eval 模式）。
    #     作用：作为"起点"基线，衡量 policy 偏离了多少。
    #
    #   DPO 的优化目标不是让 policy 在 chosen 上概率更高，
    #   而是让 policy 比 ref 更偏好 chosen（即相对概率提升）。
    #   这样可以防止 policy 单纯增大所有 token 的概率（即 model collapse）。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # 初始化 reference 模型：完全独立的副本，从相同权重初始化
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()               # ref 始终在 eval 模式（不使用 dropout 等）
    ref_model.requires_grad_(False)  # 冻结 ref 全部参数
    # ⚠️ ref_model 不传给 DDP，也不传给 optimizer，只在前向推理时使用
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    # ========== 数据集、采样器、scaler、优化器 ==========
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # DPO 是全参数更新（不是 LoRA），所以 optimizer 接收 model.parameters()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 续训：从 checkpoint 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP 封装 policy 模型 ==========
    # 注意：ref_model 不封装 DDP——它不会被分布式同步，也不需要
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 多 epoch 训练循环 ==========
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
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config,
                        start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ========== 9. 清理分布式进程组 ==========
    if dist.is_initialized():
        dist.destroy_process_group()


# ====================================================================
# 📚 练习题 — DPO 偏好对齐核心概念自测
# ====================================================================
#
# 1. (概念) DPO 和 RLHF-PPO 的核心区别是什么？DPO 为什么不需要 Reward Model？
#
# 2. (概念) DPO 中为什么需要两个模型（policy 和 ref）？
#    ref 模型的权重从哪来？为什么必须冻结？
#
# 3. (计算) 假设 ref_log_probs = [-0.2, -0.5]（分半后 chosen=-0.2, rejected=-0.5），
#    policy_log_probs = [-0.1, -0.6]，beta=0.1，计算相对优势和最终 DPO loss。
#
# 4. (阅读) dpo_loss 中，为什么对每个序列的 log_probs 要除以 seq_lengths？
#    如果不除会有什么问题？
#
# 5. (推理) DPO 的 learning rate（4e-8）为什么比 SFT（1e-6）还要小两个数量级？
#
# 6. (对比) 以下两种情况的相对优势和 loss 分别是多少？哪个学对了，哪个学歪了？
#    情况A：policy 对 chosen=0.9, rejected=0.3；ref 对 chosen=0.7, rejected=0.5
#    情况B：policy 对 chosen=0.8, rejected=0.7；ref 对 chosen=0.7, rejected=0.3
#
# 7. (数据) DPODataset 中的 generate_loss_mask 是如何区分 prompt 和 response 的？
#    和 SFTDataset 的 generate_labels 有什么异同？
#
# 8. (推理) DPO 训练中将 chosen 和 rejected 拼接成一个大 batch 前向一次，
#    如果改成分别前向两次会有什么优劣？
#
# 9. (深入) 如果用 LoRA 做 DPO，ref 模型是否也需要注入 LoRA？为什么？
# ====================================================================
#
#
# ========================= 参考答案 =========================
#
# Q1: 你的理解基本正确。
#     RLHF-PPO：4个模型（policy, ref, reward, critic），流程复杂
#     DPO：2个模型（policy, ref），直接对比 chosen/rejected 的 log_probs 差异
#     不需要 Reward Model 是因为 DPO 通过数学推导从偏好数据中直接得到 loss，
#     不需要一个额外的模型来打分。核心就是你说的"对比两个模型对 chosen/rejected 的差异"。
#
# Q2: 你的理解完全正确。
#     两个模型：一个训练中更新（policy），一个冻结当参照（ref）。
#     ref 从相同的 --from_weight 加载，权重与 policy 初始时完全一样。
#     冻结是因为它必须作为一个固定标尺——如果 ref 也在变，就不知道"相对于谁"了。
#
# Q3: 关于 β（beta）的作用：
#     β 控制"拉开 chosen 和 rejected 之间差距的力度"。
#     回到 loss 公式：loss = -log σ(β × 相对优势)
#     β 大 → σ 曲线更陡峭 → 同样的相对优势会被放大 → loss 对差距更敏感
#     β 小 → σ 曲线更平缓 → 同样的相对优势被压缩 → loss 对差距不敏感
#     极端情况：
#       β = 0 → loss = -log σ(0) = -log(0.5) = 0.693（恒定值，完全学不到任何偏好）
#       β = 1 → loss 对差距非常敏感，训练可能不稳定
#     β = 0.1 是一个常用的温和值，在稳定性和有效性之间取得平衡。
#     本题的具体计算：
#       pi_logratios = (-0.1) - (-0.6) = 0.5
#       ref_logratios = (-0.2) - (-0.5) = 0.3
#       logits = 0.5 - 0.3 = 0.2
#       loss = -log σ(0.1 × 0.2) = -log σ(0.02) ≈ -log(0.505) ≈ 0.687
#
# Q4: 你的理解正确。
#     不同序列长度不等，如果直接 sum，长序列的累积 log_probs 天然比短序列小得多。
#     除以 seq_lengths 得到平均 log_prob，使不同长度序列的数值具有可比性。
#
# Q5: DPO 学习率极小（4e-8）的原因（SFT 是 1e-6，LoRA 是 1e-4）：
#     DPO 是在 SFT 已经训好的模型上做"微调中的微调"：
#     - SFT 阶段：模型从零学，需要较大学习率（1e-6）
#     - LoRA 阶段：LoRA 矩阵从零初始化，学习率最大（1e-4）
#     - DPO 阶段：模型已经能很好地回答问题了，只需微调偏好方向，
#       学习率再大一点就会破坏 SFT 学到的知识（灾难性遗忘）。
#     类比：SFT 是学写字，LoRA 是练特定字体，DPO 是调整笔锋角度——
#     调整幅度最小，所以步长也应该最小。
#
# Q6: 你答反了——是 A 学对了，B 学歪了。
#     情况A：pi=0.9-0.3=0.6, ref=0.7-0.5=0.2, logits=0.4 ✓ 正向，学对了
#     情况B：pi=0.8-0.7=0.1, ref=0.7-0.3=0.4, logits=-0.3 ✗ 负向，学歪了
#     B 的问题：Policy 的偏好差距（0.1）反而小于 Ref 的基线差距（0.4），
#     说明 Policy 相比 Ref 反而更不偏好 chosen 了。
#     更具体来说：B 中 Policy 给 chosen=0.8、rejected=0.7，几乎打平——模型没有
#     区分好坏，就是你说的"无脑增大所有 token 概率"。
#
# Q7: DPODataset.generate_loss_mask 的原理（lm_dataset.py:223-239）：
#     它扫描 input_ids 中所有的 bos_id（即 "<|bos|>assistant\n"），
#     从每个 bos_id 之后到下一个 eos_id（即 "<|eos|>\n"）之前标记为 1（有效），
#     其余位置标记为 0（mask 掉）。
#     与 SFTDataset 的 generate_labels 相比：
#     - SFT label 把 prompt 区域设为 -100（CrossEntropyLoss 忽略 -100）
#     - DPO mask 把 prompt 区域设为 0，response 区域设为 1（后续乘以 mask 归零）
#     本质相同的策略，只是 SFT 用 -100 丢给 loss 忽略，DPO 用 mask 手动乘零。
#
# Q8: 你的理解正确。
#     拼接成大 batch 一次 forward：
#       ✓ 节省一次 forward 时间（chosen 和 rejected 共享模型权重计算）
#       ✓ loss 函数内部按 batch 分半，计算简洁
#       ✗ 显存翻倍（等价于 batch_size×2）
#     分成两次 forward：
#       ✓ 显存减半
#       ✗ 多一次前向时间
#       ✗ loss 需要分别收集 chosen/rejected 的 log_probs 再手动计算
#     当前实现选择了"一次 forward"策略，显存换速度。
#
# Q9: 你的理解正确。
#     ref 模型不需要注入 LoRA。因为 ref 是冻结的参照物，它的输出作为基线衡量
#     policy 的偏离程度。如果给 ref 也注入 LoRA，ref 本身也会发生变化，
#     标尺就变了，DPO 的相对优势计算就失去了意义。
# ====================================================================

