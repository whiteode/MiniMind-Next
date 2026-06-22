# -*- coding: utf-8 -*-
"""
===================================================
全参数指令微调（Full SFT）训练脚本
===================================================

与 pretrain（预训练）的核心差异：

    ┌───────────────────────────────┬──────────────────────────────────┐
    │          预训练               │          指令微调 (SFT)          │
    ├───────────────────────────────┼──────────────────────────────────┤
    │ 数据：大规模原始文本           │ 数据：高质量(指令, 回答)对        │
    │ 学习：下一个 token 预测        │ 学习：学会对话格式和遵循指令      │
    │ loss：所有 token 都算 loss    │ loss：只算助手的回答部分          │
    │ 学习率：~1e-4                 │ 学习率：~1e-6（防止破坏预训练知识）│
    │ 训练轮数：多轮（5~10+）       │ 训练轮数：少（1~3轮）            │
    │ 从零开始训练                  │ 基于预训练权重微调                │
    └───────────────────────────────┴──────────────────────────────────┘

整体训练流水线：

    train_pretrain.py（预训练，学语言建模）
        ↓ 产出 pretrain.pth
    train_full_sft.py（指令微调，学对话格式）  ← 你在这里
        ↓ 产出 full_sft.pth
    train_reason.py / dpo / ppo / grpo / spo（偏好对齐/推理微调）

Loss Masking 机制（SFT 最关键的概念）：

    LLM 在 SFT 中仍然做 next token prediction，但只有"助手的回答部分"贡献 loss。
    用户问题 / 系统提示词等位置的 loss 被设为 -100（PyTorch CrossEntropyLoss 默认忽略 -100 标签）。

    一条典型对话：

        <|system|>
        你是一个有用的助手。
        <|user|>
        今天天气怎么样？
        <|assistant|>
        今天天气晴朗，温度 25°C。
        <|end|>

    其中只有 "今天天气晴朗，温度 25°C。" 这部分参与 loss 计算。

    【追问：如果只有回答部分算 loss，那模型怎么学会 <|system|>、<|user|>、<|assistant|>
     这些特殊 token 的格式？推理时它应该从 <|assistant|> 后面开始生成，但训练时
     <|assistant|> 这个词本身的 loss 被 mask 了，它的 embedding 是怎么被训练的？】

    答案是：<|assistant|> 的 embedding 通过"下一个 token"的 loss 间接获得梯度。

    具体机制（以 LLM 的标准 next-token-prediction 为例）：

      假设完整序列为：

        input_ids:  [sys, ..., user, ..., <|a|>,  t0,   t1,   t2,   t3]
        labels:     [-100, ..., -100, ..., -100,  t0,   t1,   t2,   t3]
        参与 loss:                                 ✓     ✓     ✓     ✓

      causal attention 的规则：每个位置能看到自己及之前的所有位置。

      关键位置 n（<|assistant|> 所在位置）发生了什么：

        ① 模型读取 input[n] = <|assistant|> 的 embedding 向量
        ② attention 层让位置 n 看到所有之前位置：sys, user, ..., <|assistant|>
           位置 n 的 hidden state = f( 位置 0~n 所有信息的加权和 )
        ③ 这个 hidden state 经过 lm_head 输出 logits[n] → 预测 token t0
        ④ 位置 n 的 label 是 t0（因为它恰好是 assistant 回答的第一个字，
           没有被 mask），计算出 loss
        ⑤ loss 反向传播时，梯度从位置 n → 经过 attention 权重 → 流回到
           位置 0~n 的所有 embedding，包括 <|assistant|> 自身的 embedding

      【追问：梯度反传时只影响 embedding 吗？还是也会影响其他参数？
       能不能画个图说明？】

      这是个很好的问题。梯度不仅影响 embedding，而是影响从 loss 到输入之间的
      **所有可训练参数**。下图展示了以位置 n 为中心的完整数据流和梯度流：

      ========================== FORWARD（前向，计算>方向）==========================

        e_0      e_1      e_2      ...    e_{n-1}    e_n (= <|assistant|> embedding)
         │        │        │               │          │
         └────────┴────────┴───────...─────┴──────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Embedding 查表   │  ← 可训练参数
                    │  (vocab × d_model)│
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  W_q, W_k, W_v   │  ← 可训练参数：三个投影矩阵
                    │  Q=W_q·h_n       │
                    │  K_j=W_k·e_j     │
                    │  V_j=W_v·e_j     │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Attention 计算   │  ← 无参数（纯数学运算）
                    │  α_j = softmax(  │
                    │    Q·K_j/√d )    │
                    │  h_n = Σ α_j·V_j │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐  ← 可训练参数
                    │  FFN / LayerNorm  │  (W_1, W_2, b_1, b_2, γ, β)
                    │  ⋮               │
                    └──────────────────┘
                              │  经过所有 Transformer Layer
                              ▼
                    ┌──────────────────┐
                    │  lm_head         │  ← 可训练参数 (d_model × vocab)
                    │  logits = W·h_n  │
                    └──────────────────┘
                              │
                              ▼
                         CrossEntropyLoss(logits, target=t0)
                              │
                              ▼
                            loss (标量)

      ========================== BACKWARD（反向，梯度 <方向）==========================

                            loss
                              │
                              ▼
                    ┌──────────────────┐
                    │  ∂loss/∂logits   │
                    │  ∂loss/∂lm_head  │──→ 更新 lm_head 权重矩阵
                    └──────────────────┘
                              │
                              │  ∂loss/∂h_n
                              ▼
                    ┌──────────────────┐
                    │  ∂loss/∂α_j × V_j│
                    │  + α_j × ∂loss/∂V_j
                    └──────────────────┘
                     ╱    │    ╲   (链式法则拆分到三条支路)
                     ①    ②    ③
                     │    │    │
                     ▼    ▼    ▼
               ┌────────┐┌────────┐┌────────┐
               │∂loss/∂Q││∂loss/∂K││∂loss/∂V│
               └───┬────┘└───┬────┘└───┬────┘
                   │         │         │
                   ▼         ▼         ▼
               ┌──────────────────────────┐
               │   ∂loss/∂W_q             │──→ 更新 W_q 矩阵
               │   ∂loss/∂W_k             │──→ 更新 W_k 矩阵
               │   ∂loss/∂W_v             │──→ 更新 W_v 矩阵
               └──────────────────────────┘
                   │         │         │
                   ▼         ▼         ▼
               ┌──────────────────────────┐
               │   ∂loss/∂e_0             │
               │   ∂loss/∂e_1             │
               │   ...                    │──→ 更新位置 0~n 所有 embedding
               │   ∂loss/∂e_n (=<|a|>)    │
               └──────────────────────────┘

      ========================== 总结 ==========================

      ✅ 哪些参数收到了梯度（从而被更新）？

         参数分组               参数量级          是否收到梯度
         ──────────────────────────────────────────────────────
         Embedding 表           vocab × d_model    ✓（仅被"用到"的行）
         W_q / W_k / W_v       3 × d_model²       ✓
         FFN 权重 + bias        8 × d_model²       ✓（通过后续 layer 回传）
         LayerNorm γ/β          2 × d_model        ✓
         lm_head                d_model × vocab    ✓

      【追问：Embedding 表更新时，是整个表一次性更新，还是只更新用到的那些行？】

      只更新**本次 forward 中用到的行**，不是整个表。

      Embedding 层本质上是一个 lookup table：
        输入是 token ID（整数） → 输出是对应的那一行向量
        Embedding.shape = [vocab_size, d_model]

      假设输入序列是 [sys, ..., user, ..., <|a|>, t0]，对应 ID 为 [101, ..., 205, ..., 340, 550]：

        e_101 = Embedding[101, :]   ← 只取了第 101 行
        e_205 = Embedding[205, :]   ← 只取了第 205 行
        e_340 = Embedding[340, :]   ← 只取了第 340 行
        e_550 = Embedding[550, :]   ← 只取了第 550 行

      forward 中**只有这 4 行**参与了计算，其余 vocab_size-4 行完全没有被触及。

      backward 时梯度只流回这 4 行：

        ∂loss/∂Embedding[101, :] = 有值（本轮更新）
        ∂loss/∂Embedding[205, :] = 有值（本轮更新）
        ∂loss/∂Embedding[340, :] = 有值（本轮更新）
        ∂loss/∂Embedding[550, :] = 有值（本轮更新）
        ∂loss/∂Embedding[其余所有行] = 0（本轮不更新）

      所以一 step 训练后：
        - "今天"、"天气"、"<|assistant|>" 等当前 batch 出现的 token embedding 被微调了
        - "量子"、"恐龙"、"微积分" 等没出现在本 batch 的 token embedding 纹丝不动

      但随着训练持续（成千上万个 step），不同 batch 覆盖到词表中的不同 token，
      所有行的 embedding 最终都会被更新到。

      【追问：那 lm_head 呢？它也是 [d_model, vocab_size] 的大矩阵，
      它的梯度也是稀疏的吗？】

      不一样！lm_head 的梯度是**稠密（dense）**的，所有行都有值。

      原因：
        CrossEntropyLoss 的计算涉及整个词表的 logits：
          logits = h_n · W_lm_head     形状 [1, vocab_size]
          softmax(logits) 对每个 vocab 位置都算了一个概率
          每个 vocab 位置的梯度 ∂loss/∂logits[i] 都非零

        所以：
          ∂loss/∂W_lm_head[:, i] = h_n · ∂loss/∂logits[i]  对每个 i 都有值

        lm_head 的每一行（对应词表里的每个词）都收到了梯度——即使这个词
        本轮 forward 根本没出现在输入里。

      关键区别：
        ┌──────────────────────────┬──────────────────────────┐
        │     Embedding 层         │     lm_head 层           │
        ├──────────────────────────┼──────────────────────────┤
        │ 梯度稀疏（仅用到的行）    │ 梯度稠密（所有行）        │
        │ 因为：lookup 只取了部分行 │ 因为：matmul 涉及全部行   │
        └──────────────────────────┴──────────────────────────┘

        如果模型做了 weight tying（embedding 和 lm_head 共享同一个权重张量，
        即 Embedding.weight is lm_head.weight，注意是 Python 的 `is`，
        不是值相等，而是**同一个对象**），那么虽然 embedding 侧
        的梯度是稀疏的，但 lm_head 侧的稠密梯度会补上来——最终**所有行
        都有非零梯度**。

        【追问：共享的话，同一个张量会不会被 optimizer.step() 更新两次？】

        不会。只有一个 `.grad`，`optimizer.step()` 也只读一次。

        原因：
          Embedding.weight 和 lm_head.weight 是**同一个 Tensor 对象**。
          PyTorch 的 autograd 在反向传播时，把来自 embedding 路径的梯度
          和来自 lm_head 路径的梯度**累加**到这个共享 Tensor 的 `.grad` 属性上：

            weight.grad = ∂loss/∂weight_embedding_path + ∂loss/∂weight_lm_head_path

          optimizer.step() 遍历所有参数，对每个参数执行一次更新：
            weight.data -= lr × weight.grad

          所以流程是：
            ① Embedding 路径 → 在某些行上累加梯度（稀疏）
            ② lm_head 路径 → 在所有行上累加梯度（稠密）
            ③ optimizer 读取 weight.grad，**一次**更新整张表

          最终效果：词表中每个词的 embedding 都从 lm_head 路径获得了"输出侧"
          的监督信号，而当前 batch 中出现的词还额外从 embedding 路径获得了
          "输入侧"的监督信号。两个路径的梯度叠加在一起，共享的权重同时受益
          于两方面的学习信号。

      所以，位置 n 的 t0 预测不仅是 embedding 被更新——从头到尾所有参与计算的
      可训练参数都收到了梯度。embedding 只是链条上的最后一环。

      类比：
        embedding 更新就像"厨师调整配料的用量"，
        W_q/W_k/W_v 更新就像"厨师调整切菜的手法"，
        FFN 更新就像"厨师调整火候的大小"。
        t0 的预测错了，所有环节一起承担、一起调整——这就是端到端学习的核心。

      推理时，调用方构建好完整 prompt（含特殊 token），模型从 <|assistant|> 之后
      开始续写。因为训练时它见过无数遍"<|user|>问题<|assistant|>答案"的格式，
      所以能正确理解每个特殊 token 的位置含义，从正确的位置开始生成。

      类比：你学英语时，老师只批改你的回答，不批改题目本身。但你看了 10000 道
      题目之后，自然就学会了"疑问句以 what/how/why 开头"的格式——题目（特殊 token）
      虽然没被"评分"，但作为条件反复出现，你的大脑（模型）学会了它们的模式。

    详见 dataset/lm_dataset.py SFTDataset.generate_labels() 的实现。
"""
# ═══════════════════════════════════════════════════════════════════════════
# 标准库：文件路径与系统操作
# ═══════════════════════════════════════════════════════════════════════════
import os
import sys

# 将当前包标记为 "trainer"，确保相对导入（from .xxx import ...）能正常工作
__package__ = "trainer"
# 把项目根目录加入 sys.path，使得 from model.xxx / dataset.xxx 等导入能找到模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ═══════════════════════════════════════════════════════════════════════════
# 标准库：命令行参数、时间、警告
# ═══════════════════════════════════════════════════════════════════════════

# argparse：Python 标准库的命令行参数解析器
#   让用户通过命令行传参（如 --epochs 2 --batch_size 16）
#   比在代码里硬编码参数灵活得多
import argparse

# time：时间戳与计时
#   主要用于计算训练耗时（每步/每 epoch），打印 ETA（剩余时间）
#   在 train_epoch() 中通过 time.time() - start_time 算耗时
import time

# warnings：警告控制
#   用 warnings.filterwarnings('ignore') 屏蔽掉第三方库的烦人警告
#   （如 PyTorch 的 deprecation warning、tokenizer 的 padding 警告等）
import warnings

# ═══════════════════════════════════════════════════════════════════════════
# PyTorch 核心库
# ═══════════════════════════════════════════════════════════════════════════

# torch：PyTorch 主库
#   提供了 Tensor 运算、autograd（自动求导）、神经网络层、
#   CUDA 支持（torch.cuda）、以及模型保存/加载等全部核心功能
import torch

# torch.distributed（简称 dist）：PyTorch 分布式训练模块
#   用于多卡 DDP（Distributed Data Parallel）训练：
#   - dist.init_process_group()：初始化分布式进程组（NCCL backend）
#   - dist.get_rank()：获取当前进程的全局编号（主进程=0）
#   - dist.is_initialized()：判断当前是否在分布式模式下
#   - dist.barrier()：同步所有进程（等大家都到齐了再继续）
#   - dist.destroy_process_group()：训练结束时清理
import torch.distributed as dist

# ═══════════════════════════════════════════════════════════════════════════
# Python 标准库 & torch 工具
# ═══════════════════════════════════════════════════════════════════════════

# contextlib.nullcontext：空上下文管理器
#   当不需要做任何特殊处理时，作为 autocast 的 fallback（降级替代）
#   在 CPU 训练时：autocast_ctx = nullcontext()，相当于"什么都不做"
#   在 GPU 训练时：autocast_ctx = autocast(dtype=...)，启用混合精度
#   好处：不需要写 if/else，直接把 nullcontext 当成上下文管理器用即可
from contextlib import nullcontext

# torch.optim：优化器模块
#   实现各种深度学习优化算法：
#   - optim.AdamW：带权重衰减的 Adam，LLM 训练最常用的优化器
#     （decoupled weight decay，将 L2 正则与 Adam 的动量机制解耦）
#   - optim.lr_scheduler：学习率调度器（本例中通过 get_lr 手动调度）
#   - optim 还提供 SGD、Adam、Adagrad、RMSprop 等
from torch import optim

# torch.nn：神经网络模块
#   提供构建神经网络所需的所有基础组件：
#   - nn.Module：所有模型的基类
#   - nn.Linear、nn.Embedding、nn.LayerNorm 等常用层
#   - nn.CrossEntropyLoss：分类损失（内部含 softmax + NLLLoss）
#   - nn.utils.clip_grad_norm_：梯度裁剪工具
#   - nn.Parameter：可训练参数包装器
from torch import nn
from torch.nn.parallel import DistributedDataParallel
# DistributedDataParallel（DDP）：PyTorch 的多卡数据并行方案
#   工作原理：每张卡维护一个完整模型副本 → 分到不同 batch → 各自前向/反向
#   → AllReduce 同步梯度（求和平均）→ 所有卡保持一致的参数
#   和 DataParallel 的区别：DDP 每卡一个独立进程，通信效率远高于 DP
#   （DP 靠单进程多线程 + GIL 瓶颈，DDP 靠 NCCL 多进程通信）

# ═══════════════════════════════════════════════════════════════════════════
# PyTorch DataLoader 相关
# ═══════════════════════════════════════════════════════════════════════════

from torch.utils.data import DataLoader, DistributedSampler
# DataLoader：数据加载器
#   封装了 Dataset，提供：
#   - 自动 batch 组装
#   - 多进程数据预读取（num_workers）
#   - pin_memory 加速 CPU→GPU 传输
#   - shuffle / sampler 控制数据顺序
#
# DistributedSampler：分布式数据采样器
#   在 DDP 下，每张卡需要拿到不同的数据子集。
#   它会根据总进程数和当前进程 rank，把数据集切分成互不重叠的碎片，
#   每张卡只取属于自己的那份。确保数据不重复、不遗漏。
#   关键方法：set_epoch(epoch)，每个 epoch 调用一次，让 shuffle 顺序不同

# ═══════════════════════════════════════════════════════════════════════════
# 项目内部模块
# ═══════════════════════════════════════════════════════════════════════════

from model.model_minimind import MiniMindConfig
# MiniMindConfig：模型配置类（dataclass / 简单容器）
#   集中管理所有超参数：
#   - hidden_size（隐藏层维度，默认 512）
#   - num_hidden_layers（Transformer 层数，默认 8）
#   - use_moe（是否使用 MoE 架构）
#   - vocab_size、num_attention_heads、intermediate_size 等
#   通过 from model.model_minimind import MiniMindConfig 导入并实例化后
#   传给 init_model()，控制模型的初始化参数

from dataset.lm_dataset import SFTDataset
# SFTDataset：指令微调数据集类
#   继承自 torch.utils.data.Dataset
#   和 PretrainDataset 的关键区别：
#   - 输入是 JSONL 对话数据（含 system/user/assistant 多轮）
#   - 用 tokenizer.apply_chat_template() 将对话转为模型输入格式
#   - generate_labels() 实现 loss masking：只把 assistant 回答部分
#     设为真实 token ID（参与 loss），其余位置设为 -100（被忽略）

from trainer.trainer_utils import (
    get_lr,           # 学习率调度函数（warmup + cosine decay）
    Logger,           # 带等级/颜色的日志工具（同时写入文件 + 终端输出）
    is_main_process,  # 判断当前进程是否为主进程（rank=0），控制 I/O
    lm_checkpoint,    # 模型 checkpoint 保存/恢复（含 optimizer + scaler）
    init_distributed_mode,  # 解析分布式环境变量，初始化进程组
    setup_seed,       # 设置全局随机种子（确保可复现性）
    init_model,       # 创建模型实例，可选加载预训练权重
    SkipBatchSampler, # 断点续训专用：跳过已处理过的 batch
)
# 这些工具函数集中在 trainer/trainer_utils.py 中，避免各脚本重复代码

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    在单 epoch 上训练一个 SFT 步循环。

    参数
    ----------
    epoch : int
        当前轮次编号（从 0 开始），主要用于日志和 lr 调度。
    loader : DataLoader
        产生 (input_ids, labels) 的迭代器。
    iters : int
        当前 epoch 的总步数（含经过的 skip），用于 lr cosine 调度和 ETA。
    start_step : int
        断点续训时从 start_step+1 开始计数（日志连续）。
    wandb : object or None
        swanlab 可视化对象，非 None 时记录 loss/lr 曲线。

    数据流
    ------
        DataLoader ──→ (input_ids, labels)
                            │
                            ▼
                    model(input_ids, labels=labels)
                            │
                            ▼
                    loss = logits_loss + aux_loss
                            │
                            ▼
                    loss / accumulation_steps → backward (梯度累积)
                            │
                            ▼
                    optimizer.step() → 更新模型参数

    Loss 构成
    ---------
    - logits_loss : 主语言模型损失（cross-entropy），只有助手的回答部分参与
    - aux_loss : MoE 辅助损失（仅 use_moe=True 时有），平衡各专家的负载
                 防止路由崩溃（router collapse）
    """
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # input_ids : token ID 序列，形状 [B, L]
        # labels    : 目标序列（含 -100 掩码），形状 [B, L]
        #   -100 表示该位置不参与 loss 计算（用户/系统部分）
        #   非 -100 表示该位置属于助手回答，需要预测
        #
        # 【追问：-100 为什么能"让 loss 忽略这个位置"？】
        #   这是 PyTorch 的 nn.CrossEntropyLoss 内置行为。
        #   ignore_index 参数（默认 -100）被设为忽略目标值等于该值的样本。
        #   在前向传播中，模型对所有 token 都算出 logits，但 loss 函数
        #   在 reduce 时会把 ignore_index 对应的元素排除掉，不参与求和也不除它们。
        #   效果：梯度只从 non-masked 的位置反传，模型只学这些位置的知识。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # ========== 学习率调度 (Cosine Decay + Warmup) ==========
        #
        # global_step 的计算：
        #   当前 epoch 之前的所有步数：epoch * iters
        #   当前 epoch 已走的步数：step
        #   合起来 = epoch * iters + step = 全局训练步数
        #
        # get_lr() 内部实现了 warmup + 余弦退火：
        #   - warmup 阶段（前 ~10% 步数）：lr 从 0 线性升到 target_lr
        #   - cosine 阶段（后续 ~90% 步数）：lr 从 target_lr 按余弦曲线降到 10% target_lr
        #
        # SFT 的学习率（~1e-6）比预训练（~1e-4）低两个数量级的原因是：
        #   预训练权重已经学到了高质量的语言表示，SFT 只需要"微调"而非"重塑"，
        #   太大的学习率会破坏预训练知识，导致灾难性遗忘（catastrophic forgetting）。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== 混合精度前向传播 ==========
        #
        # autocast 自动将计算密集型算子（matmul, linear）降为 bf16/fp16，
        # 数值敏感型算子（softmax, layernorm）保留 fp32。
        with autocast_ctx:
            # model(input_ids, labels=labels) 做了两件事：
            #   1. 前向推理：input_ids → embedding → transformer layers → lm_head → logits
            #   2. 计算 loss：logits 与 labels 算 cross-entropy（忽略 -100 位置）
            res = model(input_ids, labels=labels)
            # logits_loss + aux_loss：
            #   logits_loss = 语言模型主损失（next token prediction cross-entropy）
            #   aux_loss    = MoE 辅助平衡损失（仅 use_moe=True 时有值）
            loss = res.loss + res.aux_loss
            # 梯度累积缩放：除以 accumulation_steps，使得 accumulation_steps 次
            # 小 batch 的梯度之和 = 等效大 batch 的一次梯度
            loss = loss / args.accumulation_steps

        # ========== 反向传播（带 GradScaler） ==========
        #
        # scaler.scale(loss).backward() 做了：
        #   1. loss *= 2^16 （放大 loss，防止梯度下溢）
        #   2. loss.backward() （反传，梯度也被缩放了 2^16 倍）
        #   3. 在 optimizer.step() 前 unscale 除回去
        #
        # 为什么 SFT 也需要 scaler？
        #   SFT 的 loss 通常比预训练小（因为只算回答部分，序列更短），
        #   梯度可能更小，更容易下溢 → scaler 在 fp16 下是必要的。
        scaler.scale(loss).backward()

        # ========== 梯度累积边界：每 accumulation_steps 步更新一次参数 ==========
        #
        # 为什么需要梯度累积？
        #   显存有限，装不下太大的 batch。但太大的 accumulation_steps 会让
        #   参数更新间隔过长，影响收敛。
        #
        # SFT 场景下 accumulation_steps=1（默认）意味着每步都更新：
        #   因为 SFT 数据集小（通常几千~几万条），batch_size=16 已经够用，
        #   不需要像预训练那样用梯度累积模拟大 batch。
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪 (Gradient Clipping)
            #   将梯度的全局 L2 范数限制在 grad_clip=1.0 以内。
            #   如果 ||g|| > 1.0，则 g = g / ||g|| × 1.0。
            #
            # 为什么需要梯度裁剪？
            #   某些 batch 可能产生异常大的梯度（outlier batch），
            #   一步更新把模型参数推到一个很差的区域，loss 飙升。
            #   裁剪后虽然这步的方向被缩短了，但保留了正确的方向，
            #   不会破坏之前学到的知识。
            #
            # 【追问：model.parameters() 包含所有参数还是只包含可训练的？】
            #   它返回所有 requires_grad=True 的参数。
            #   对于 SFT，所有参数都是可训练的（full fine-tune），
            #   所以 model.parameters() 包含了整个模型的全部参数。
            #   但如果是 LoRA 微调，model.parameters() 就只包含
            #   LoRA 的低秩矩阵和偏置——这正是"全参数微调"和"参数高效微调"的区别。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # ========== 日志打印 ==========
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # current_loss 反除了 accumulation_steps，恢复真实的 loss 值
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ========== 保存 checkpoint ==========
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 扒壳（unwrap）：
            #   DDP 给模型包了一层 DistributedDataParallel（module.前缀）
            #   torch.compile 加了一层 _orig_mod
            #   需要逐层扒开才能拿到原始模型的 state_dict
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存半精度权重（fp16），将显存占用的模型权重压缩到一半大小以节省磁盘
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 完整 checkpoint（含 optimizer/scaler/step/epoch）用于断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")

    # ========== SFT 的关键参数说明 ==========
    #
    # SFT 和预训练共用一套模型架构，但以下几点必须注意：
    #
    # 1. from_weight（默认 'pretrain'）
    #    这是 SFT 最重要的参数。init_model 会加载预训练产出的权重文件
    #   （默认路径 ../out/pretrain_512.pth），作为 SFT 的初始权重。
    #    如果设为 'none'，则从随机初始化开始训练（几乎没有人这么做，
    #    因为没有预训练的 SFT 质量极差）。
    #
    # 2. batch_size（默认 16）
    #    SFT 数据集通常较小，batch_size 不需要太大。
    #    和预训练不同，SFT 不需要 accumulation_steps 模拟大 batch。
    #
    # 3. learning_rate（默认 1e-6）
    #    比预训练（~1e-4）低两个数量级，防止灾难性遗忘。
    #
    # 4. max_seq_len（默认 340）
    #    和预训练一致。但要特别注意：SFT 数据中一条对话的 token 数
    #    通常比原始文本更长（因为加了 system/user/assistant 标记）。
    #    过小的 max_seq_len 可能导致回答被截断。
    #
    # 5. from_weight vs from_resume 的区别：
    #    from_weight = 'pretrain'：加载预训练权重（初始化模型）
    #    from_resume = 1：加载 SFT 自身的 checkpoint（断点续训）
    #    两者可以同时使用——先 from_weight 初始化，然后用 from_resume 覆盖。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    """
    =====================================================================
    训练主流程（和 pretrain 高度相似，但关键差异已在下文标注）
    =====================================================================
    """

    # ========== 1. 初始化分布式环境和随机种子 ==========
    #
    # init_distributed_mode() 解析环境变量（RANK, WORLD_SIZE, MASTER_ADDR 等）
    # 并初始化 PyTorch 分布式进程组（NCCL backend）。
    # 如果是单卡训练，它什么也不做（dist.is_initialized() == False）。
    #
    # local_rank：当前进程的本地 GPU 编号（0~n-1），用于设置 device。
    # 如果 DDP 未初始化，local_rank = -1（详见 trainer_utils）。
    #
    # 【追问：分布式训练时 dist.get_rank() 和 local_rank 有什么区别？】
    #   dist.get_rank() 是全局 rank（多机多卡下，所有进程的唯一编号）。
    #   local_rank 是当前机器上的 GPU 编号（0 ~ n_gpu_per_node-1）。
    #
    #   举例：2 台机器，每台 4 张 GPU：
    #     机器 1 的 rank 0~3，local_rank 0~3
    #     机器 2 的 rank 4~7，local_rank 0~3
    #
    #   local_rank 用于分配具体的 GPU device（f"cuda:local_rank"），
    #   而 rank 用于全局同步（如 barrier）和判断主进程（rank=0 负责 I/O）。
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # setup_seed 设随机种子，保证每张卡数据顺序一致。
    # 分布式下每个进程的 seed = 42 + rank（不同卡的 shuffle 结果不同，
    # 但各自内部确定，保证数据不重复）。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查 checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # from_resume=1 时检测 ../checkpoints/ 下是否有 full_sft_resume.pth
    # 有则加载（模型权重 + 优化器动量 + scaler + 轮次/步数）
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # GPU 上用 autocast（自动混合精度），CPU 上用 nullcontext（不做事）
    #
    # 【追问：为什么 CPU 不启用 autocast？】
    #   autocast 依赖 GPU 的 Tensor Core 做低精度矩阵乘加速。
    #   CPU 上没有高效的 bf16/fp16 硬件加速单元，
    #   类型转换只有开销没有收益，所以直接 nullcontext 用 fp32 算。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # SwanLab 是 wandb 的开源替代品
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    #
    # init_model 加载模型和 tokenizer：
    #   1. 创建 MiniMindForCausalLM 实例（随机初始化）
    #   2. 如果 args.from_weight != 'none'，加载预训练权重覆盖模型参数
    #
    # 【追问：SFT 的 tokenizer 和 pretrain 用的是同一个吗？】
    #   是的。tokenizer 不变，词汇表不变。
    #   唯一变化的是数据格式：pretrain 吃原始文本，SFT 吃对话格式。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # ========== SFTDataset：SFT 特有的数据封装 ==========
    #
    # SFTDataset (dataset/lm_dataset.py:113) 和 PretrainDataset 的关键区别：
    #
    #   ┌─────────────────┬─────────────────────┬─────────────────────┐
    #   │                 │  PretrainDataset    │  SFTDataset         │
    #   ├─────────────────┼─────────────────────┼─────────────────────┤
    #   │ 输入格式        │ 原始文本             │ 对话 JSON           │
    #   │ Tokenize        │ 直接 tokenize        │ apply_chat_template │
    #   │ labels          │ input_ids 右移一位   │ 仅 assistant 部分   │
    #   │ loss 掩码       │ 所有 token          │ assistant 以外的 -100│
    #   │ padding         │ 通常不需要           │ 需要（batch 内对齐） │
    #   └─────────────────┴─────────────────────┴─────────────────────┘
    #
    # generate_labels 的具体逻辑：
    #   1. 搜索 bos_id（即 "<|assistant|>\n" 的 token ID 序列）
    #   2. 从 bos_id 之后开始，到 eos_id 之前结束，标记为"参与 loss"
    #   3. 其余位置（system/user/pad）标记为 -100
    #
    # 为什么只用 assistant 部分算 loss？
    #   我们不希望模型学会"生成用户问题"或"重复系统提示词"，
    #   只希望模型学会"怎么以助手的身份回答"。
    #   如果所有 token 都算 loss，模型会花大量容量背诵用户问题，
    #   浪费参数在无关的模式上。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从 checkpoint 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重（含训练中更新的参数，如 attention、mlp 等）
        model.load_state_dict(ckp_data['model'])
        # 恢复优化器动量（AdamW 的 exp_avg / exp_avg_sq 缓冲区）
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 恢复 GradScaler 状态（loss scale 因子和 steps 计数）
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP 包装模型 ==========
    if dist.is_initialized():
        # 忽略 RoPE 位置编码缓存（freqs_cos, freqs_sin）的梯度同步。
        # 这些是固定不可训练的缓冲区，每张卡完全一样，
        # DDP 不需要在 allreduce 中同步它们，节省通信带宽。
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # DistributedSampler.set_epoch(epoch)：每个 epoch 改变数据 shuffle 顺序
        train_sampler and train_sampler.set_epoch(epoch)
        # torch.randperm 生成本轮数据顺序的随机排列
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        # 断点续训：如果从中间恢复，跳过前面已处理的步数
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True  # 页锁定内存加速 CPU→GPU 数据传输
        )
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()

# ========== 🧪 自测练习题 ==========
"""
《train_full_sft.py 学习自测题》
每题答案可以在本文件注释中找到。

────────────────────────────────────
一、基础题（确认你理解了核心概念）
────────────────────────────────────

1. SFT 和预训练在 loss 计算上的核心区别是什么？为什么 SFT 要这样做？

   【你的回答区域】


   【✅ 参考答案】
   预训练时所有 token 都参与 loss 计算（input_ids 右移一位就是 labels），
   因为预训练的目标是"学会语言建模"——每个位置的 next token 都值得学。

   SFT 只让 assistant 回答部分参与 loss，其余位置（system/user/pad）设为 -100。
   原因是：
   ① 不希望模型花参数去背诵用户问题或系统提示词——那是调用方的责任
   ② 只训练"怎么以助手的身份回答"，让模型专注学习对话能力和指令遵循
   ③ 计算资源集中在关键部分，训练更高效

2. SFTDataset.generate_labels() 中 -100 标签的作用是什么？
   PyTorch 的 CrossEntropyLoss 如何处理 -100？

   【你的回答区域】


   【✅ 参考答案】
   -100 是 CrossEntropyLoss 的 ignore_index 默认值。
   当 loss 函数遇到标签为 -100 的位置时，该位置不参与 loss 计算：
   - 不计入 loss 求和
   - 不参与分母（token 总数）的计算
   - 梯度不从这个位置反向传播

   效果：-100 位置的模型输出无论对错都不影响训练，模型只从非 -100 的位置学习。

3. SFT 的学习率（默认 1e-6）为什么比预训练（~1e-4）低两个数量级？
   如果 SFT 用了和预训练一样大的学习率会怎样？

   【你的回答区域】


   【✅ 参考答案】
   预训练是从零或随机初始化开始学语言建模，需要大学习率快速收敛。
   SFT 基于预训练权重微调，权重已经包含了高质量的语言知识。
   用小学习率（1e-6）的原因是：
   ① 防止灾难性遗忘（catastrophic forgetting）——大学习率会把预训练
      学到的语言知识"冲掉"
   ② SFT 只需要在预训练基础上做微小调整——学会对话格式就够
   ③ 大学习率会导致 loss 震荡甚至发散，训练不稳定

   如果用了预训练的学习率：模型会快速遗忘预训练知识，生成质量严重下降，
   可能出现胡言乱语或语法错误。

────────────────────────────────────
二、进阶题（需要关联多个知识点）
────────────────────────────────────

4. 如果 SFT 时设置 from_weight='none'（从随机初始化开始训练），
   和 from_weight='pretrain' 相比会有什么差异？为什么几乎没有人这样做？

   【你的回答区域】


   【✅ 参考答案】
   from_weight='none' 时模型参数是随机初始化的，没有经过任何预训练。
   此时做 SFT 会出现：
   ① 模型根本没有学会语言建模（语法、知识、上下文理解），直接学对话格式
      相当于"还没学会说话就学怎么回答问题"
   ② 收敛极慢——需要大量 SFT 数据来同时学语言和对话
   ③ 最终效果差——SFT 数据量通常很小（几千~几十万条），远不够学语言
   ④ 实际上模型输出的文本质量极低

   from_weight='pretrain' 加载了预训练权重，模型已经学会了语言建模。
   SFT 只需要微调对话格式——数据少、收敛快、效果好。
   这就是"预训练 + 微调"范式的核心假设。

5. 为什么 SFTDataset 需要对序列做 padding 来对齐 batch 内长度，
   而 PretrainDataset 通常不需要？padding 的 token 在 loss 计算中
   会被如何处理？

   【你的回答区域】


   【✅ 参考答案】
   预训练数据是原始文本，每条样本通常独立截断到 max_seq_len，长度一致，
   不需要 padding（或者 padding 很少）。

   SFT 数据经过 apply_chat_template 后，不同对话的 token 数差异很大
   （简单问答 vs 多轮复杂对话），必须 padding 到相同的 max_seq_len
   才能组成 batch。

   SFTDataset 的实现（lm_dataset.py:159）：
     input_ids += [pad_token_id] * (max_length - len(input_ids))
   然后 generate_labels 时，pad_token_id 对应的位置不会被识别为
   bos_id → 不会被标记为"参与 loss" → 保持 -100 → loss 忽略它们。

6. 在 SFT 中，<|system|>、<|user|>、<|assistant|> 这些特殊 token
   的 loss 被 mask 了（label=-100），它们的 embedding 是如何被训练的？
   请从 causal attention 和反向传播的角度解释。

   【你的回答区域】


   【✅ 参考答案】
   虽然特殊 token 本身的 label 是 -100（直接 loss 忽略），
   但它们的 embedding 通过以下方式获得梯度：

   ① attention 机制让每个位置能看到所有之前的位置
   ② 位置 n（<|assistant|>）的 hidden state 聚合了
      位置 0~n 的所有信息（包括 system/user 等）
   ③ 这个 hidden state 被用来预测 token t0（回答的第一个字）
   ④ t0 的 label 没有被 mask → t0 的 loss 正常反向传播
   ⑤ 梯度从 t0 → 经过 attention → 流回位置 0~n 的所有 embedding
   ⑥ <|assistant|> 的 embedding 作为"贡献者之一"被更新

   而且后续 token（t1, t2, t3...）的 loss 也会一再地通过 attention
   回传梯度到特殊 token 的 embedding。一个特殊 token 在一步训练中
   会被更新多次（回答里有多少个 token 就传回来多少次）。

────────────────────────────────────
三、深入题（需要理解和推理，不要求代码细节）
────────────────────────────────────

7. 假设一条 SFT 数据包含两轮对话：

      <|system|>你是一个助手<|user|>天气怎么样？<|assistant|>今天晴天<|user|>明天呢？<|assistant|>明天多云

   generate_labels 会如何标记 loss 区域？
   模型在第二轮时，能"看到"第一轮 assistant 的回答吗？

   【你的回答区域】


   【✅ 参考答案】
   generate_labels 会标记两个 assistant 回答区域都参与 loss：
     [sys, user, <|a|>, 今天晴天, <|user|>, 明天呢？, <|a|>, 明天多云]
     [-100, -100, -100, 今天晴天, -100,   -100,   -100, 明天多云]
                             ✓参与loss                         ✓参与loss

   是的，causal attention 允许位置 n 看到所有之前的位置。
   当模型在第二轮预测"明天多云"时，它可以看到：
   - 第一轮的对话（天气怎么样？→ 今天晴天）
   - 第二轮的用户问题（明天呢？）
   所以模型能利用完整的对话历史来生成第二轮的回答。

   这也意味着虽然第一轮 assistant 的回答内容在 loss 上被 mask，
   但它的 hidden state 作为上下文传递到了第二轮——对第二轮预测有贡献。

8. embedding 的梯度是稀疏的（只有用到的行有梯度），而 lm_head 的梯度
   是稠密的（所有行都有梯度）。请分别解释为什么。

   如果模型做了 weight tying（Embedding.weight is lm_head.weight），
   optimizer.step() 会把这个共享权重更新两次吗？

   【你的回答区域】


   【✅ 参考答案】
   Embedding 梯度稀疏的原因：
     Embedding 是 lookup 操作——输入 token ID → 取对应的一行向量。
     forward 中只触及了输入序列中用到的 token 对应的行。
     backward 时梯度只流回被触及的行，其余行为零。

   lm_head 梯度稠密的原因：
     lm_head 是线性层：logits = h · W_lm_head，形状 [1, vocab_size]。
     这是一个完整的矩阵乘法，W_lm_head 的每一列都参与了计算。
     CrossEntropyLoss 的梯度对 vocab 的每个位置都有值 → W_lm_head 的
     所有行都有非零梯度。

   weight tying 时不会更新两次：
     共享的是同一个 Tensor 对象（is 关系，不是值相等）。
     PyTorch autograd 把两条路径的梯度累加到同一个 .grad 属性上：
       weight.grad = grad_from_embedding_path + grad_from_lm_head_path
     optimizer.step() 遍历参数一次，对每个 .grad 执行一次更新。
     最终效果：所有行都从 lm_head 路径获得稠密梯度，当前 batch 出现的
     token 还额外叠加 embedding 路径的稀疏梯度。

9. 假设 SFT 数据集中，有 90% 的样本是"你好"→"你好！有什么可以帮助你的？"
   （极短回答），10% 的样本是复杂的多轮推理（长回答）。
   训练完成后，模型在长回答上的表现可能会有什么问题？为什么？

   【你的回答区域】


   【✅ 参考答案】
   可能出现"生成退化"问题：模型在长回答场景下过早截断或回答简短。

   原因：
   ① SFT 的 loss 只算 assistant 回答部分，且默认对所有 token 做 mean reduction。
      短回答的 loss 很小（token 少），但每条样本对梯度的贡献是均等的
      （不考虑回答长度）→ 短回答模式被过度学习
   ② batch 内大多数是短回答 → 模型看到"回答应该简短"的模式更多
   ③ 复杂推理的长回答在数据中占比小 → 模型接触少 → 学不好

   解决方法：
   ① 数据平衡：确保长回答样本有足够的数量或权重
   ② Loss 按长度加权：对长回答的每个 token 赋予更高权重
   ③ 课程学习：先训练短回答，再逐步加入长回答

10. SFT 结束后产出的 full_sft.pth 和预训练的 pretrain.pth 有什么本质区别？
    如果拿着 full_sft.pth 去做第二轮 SFT（新数据集），from_weight
    应该设为 'pretrain' 还是 'full_sft'？为什么？

    【你的回答区域】


    【✅ 参考答案】
    pretrain.pth：模型学会了语言建模（语法、知识、上下文），但不会对话格式。
    full_sft.pth：在预训练基础上额外学会了对话格式和指令遵循。

    做第二轮 SFT 时 from_weight 应该设为 'full_sft'（当前轮的输出）。
    而不是 'pretrain'（上一轮的输出），因为：
    ① full_sft.pth 已经包含了对话格式的知识，继续训练是在已有基础上微调
    ② 如果用 pretrain.pth 重新开始，第一轮 SFT 学到的对话格式就浪费了
    ③ 连续微调（sequential fine-tuning）是常见做法——先在通用 SFT 数据上
       微调，再在特定领域数据上继续微调

    但要注意连续微调时的灾难性遗忘风险：第二轮 SFT 如果学习率太大，
    或在非常不同的数据上微调，可能覆盖第一轮学到的通用对话能力。
    实践中常用混合训练（新数据 + 少量旧数据）来缓解。

────────────────────────────────────
建议：
  - 基础题：每道 30 秒~2 分钟，全部答对说明你掌握了 SFT 的核心差异
  - 进阶题：每道 2~5 分钟，需要翻阅注释或代码对照
  - 深入题：开放性问题，没有标准答案，重点是思考过程
"""
