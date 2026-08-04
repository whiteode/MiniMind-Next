# model_minimind.py 学习计划（模型架构）

## 一、写在前面：为什么要学模型架构？

### 1.1 你已经学过的一切都在这里交汇

回顾你的学习历程：
- `train_tokenizer.py` → 文字变成数字（token IDs）
- `train_pretrain.py` → 数字流过 Transformer，输出下一个数字的预测
- `train_full_sft.py` → 对话数据教会模型"问答格式"
- `train_dpo.py` / `train_grpo.py` → 模型学会"什么答案更好"
- `train_distillation.py` → 小模型模仿大模型

每一个训练脚本都在调用 `MiniMindForCausalLM.forward()`，但你还没看过这个函数内部到底发生了什么。这篇文档就是要带你**逐行拆解 MiniMind 模型的每一块砖**。

### 1.2 学完这篇你能回答的问题

| 问题 | 涉及的模块 |
|------|-----------|
| 为什么 LLaMA 用 RMSNorm 而不是 LayerNorm？ | RMSNorm |
| RoPE 是怎么把位置信息"旋转"进 Q/K 的？ | precompute_freqs_cis / apply_rotary_pos_emb |
| GQA（分组查询注意力）为什么省显存？ | Attention + repeat_kv |
| SwiGLU 为什么比传统 ReLU-FFN 好？ | FeedForward |
| MoE 的路由器怎么决定一个 token 去哪个专家？ | MoEGate |
| 权重绑定（Weight Tying）是什么？为什么做？ | MiniMindForCausalLM |
| 整个模型的 104M 参数是怎么算出来的？ | MiniMindConfig |

### 1.3 阅读姿势

1. **先看结构图**（第二章），建立全局认知
2. **逐模块深入**（第三章），每个模块配代码 + 大白话
3. **做自测**（末尾 Q&A），检验是否真的理解
4. 把每个模块的参数量在心里默算一遍——这是检验是否理解的最快方式

---

## 二、全局结构：一张图看清 MiniMind 的骨架

```
输入 token IDs: [batch, seq_len]  例: [1, 10] 表示 1 句话，10 个 token

       │
       ▼
┌──────────────────────────┐
│  embed_tokens            │  词嵌入查表：token ID → 向量
│  [vocab_size, hidden]    │  [1, 10] → [1, 10, 512]
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  MiniMindBlock × 8 层 (num_hidden_layers=8)          │
│                                                      │
│  每层的处理流程（Pre-Norm 结构）：                      │
│                                                      │
│  hidden_states ──┬──▶ RMSNorm ──▶ Attention ──▶ Add   │
│                  │                     │              │
│                  └── 残差连接 ──────────┘              │
│                                         │              │
│                                         ▼              │
│  hidden_states ──┬──▶ RMSNorm ──▶ SwiGLU/MoE ──▶ Add │
│                  │                     │              │
│                  └── 残差连接 ──────────┘              │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────┐
│  RMSNorm (final)         │  最终归一化
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  lm_head                 │  线性映射到词表
│  [hidden, vocab_size]    │  [1, 10, 512] → [1, 10, 6400]
└──────────────────────────┘
           │
           ▼
输出 logits: [batch, seq_len, vocab_size]
形状 [1, 10, 6400]：10 个位置，每个位置有 6400 个候选词的分数
```

### 结构图拆解：每个 MiniMindBlock 内部到底发生了什么

上面结构图中最核心的部分是 `MiniMindBlock × 8 层`。每个 Block 是一个独立的"处理器"，一共堆叠 8 次。下面逐层拆解：

#### 子块 1：Attention（注意力机制）

```
输入 hidden_states: [batch, seq_len, hidden_size]
    │
    ├──▶ RMSNorm (input_layernorm)     ← 先归一化（Pre-Norm）
    │
    ├──▶ Attention (self_attn)          ← 让每个 token "看到"序列里其他 token 的信息
    │
    └──▶ + residual (残差连接)           ← 把原始输入直接加上去
         │
         ▼
    输出 hidden_states（同形状）
```

对应代码（`model_minimind.py:2456-2462`）：

```python
residual = hidden_states                                    # 1. 保存原始输入
hidden_states = self.self_attn(self.input_layernorm(hidden_states), ...)  # 2. Norm → Attention
hidden_states += residual                                   # 3. 残差相加
```

#### 子块 2：FFN / MoE（前馈神经网络）

```
hidden_states（来自子块1）
    │
    ├──▶ RMSNorm (post_attention_layernorm)  ← 再次归一化
    │
    ├──▶ MLP: SwiGLU 或 MOEFeedForward       ← 对每个 token 的特征做"深度加工"
    │
    └──▶ + residual (残差连接)
         │
         ▼
    输出 hidden_states（同形状）
```

对应代码（`model_minimind.py:2465`）：

```python
hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
```

#### 关键概念解释

**Pre-Norm 是什么意思？**

标准 Transformer 是"先计算，再归一化"（Post-Norm）。MiniMind 用的是 LLaMA 风格的"先归一化，再计算"（Pre-Norm），训练更稳定。

```
Pre-Norm (MiniMind):   x → Norm → Sublayer → + x
Post-Norm (原始):      x → Sublayer → + x → Norm
```

**残差连接（`+= residual`）的作用？**

相当于高速公路旁边的"快捷通道"，让梯度可以不经过复杂计算直接回传。没有它，8 层堆叠会导致梯度消失，模型训不动。

**为什么有两个 RMSNorm？**

Attention 和 FFN 是两个独立的操作，各自处理前都需要归一化。如果共用一个 Norm，会导致 Attention 输出的分布干扰 FFN 的输入。

#### 完整数据流（以 [1, 10, 512] 为例）

```
[1, 10, 512]   进入第 1 层 Block
    │  RMSNorm → Attention → + residual
    │  RMSNorm → SwiGLU    → + residual
    ▼
[1, 10, 512]   进入第 2 层 Block（形状完全不变！）
    │  ...同样处理...
    ▼
  ...重复 8 次...
    ▼
[1, 10, 512]   出来
```

整个过程中 `hidden_size`（512）维度**始终不变**，变化的是每个位置的向量内容——它们逐渐包含了越来越多的上下文语义信息。

> 大白话：把 8 层 MiniMindBlock 想象成 8 个接力赛选手。每个选手拿到一个"信息包裹"（hidden_states），先整理一下（RMSNorm），然后从不同角度吸收周围的信息（Attention），再加上原始包裹的内容（残差）。接着再整理一次，做一次深度加工（FFN/SwiGLU），再加回原始内容。8 个选手传下来，包裹里的信息越来越丰富、越来越"懂上下文"。

### 配置速查表

| 配置参数 | Small | Base | MoE | 含义 |
|---------|:-----:|:----:|:---:|------|
| hidden_size | 512 | 768 | 640 | 每个 token 的特征向量维度 |
| num_hidden_layers | 8 | 16 | 8 | Transformer 堆叠层数 |
| num_attention_heads | 8 | 12 | 8 | 注意力头数 |
| num_key_value_heads | 2 | 4 | 2 | KV 头数（GQA） |
| vocab_size | 6400 | 6400 | 6400 | 词表大小 |
| max_position_embeddings | 32768 | 32768 | 32768 | 最大上下文长度 |
| intermediate_size | 1365 | 1384 | — | FFN 中间维度（8/3 × hidden，对齐到 64 倍） |
| rope_theta | 1e6 | 1e6 | 1e6 | RoPE 基数 |
| use_moe | False | False | True | 是否启用 MoE |

---

## 三、逐模块详解

### 3.1 RMSNorm — 归一化层的"省油版"

**位置**：`model_minimind.py` 第 1111 行

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def _norm(self, x):
        # RMS = sqrt(mean(x²) + ε)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

**一句话总结**：把向量的"平均能量"归一化到 1，然后乘以可学习的缩放 weight。

**对比 LayerNorm**：
```
LayerNorm:  y = (x - μ) / σ × γ + β   需要减均值，除标准差，加偏置
RMSNorm:    y = x / RMS(x) × γ         只除均方根，无偏置
```

RMSNorm 省略了两步：
1. **不减均值**（去中心化）：实践证明对 Transformer 训练不是必需的
2. **不加偏置 β**：只需要缩放，不需要平移

直接效果：**计算量减约 25%，训练更稳定**。LLaMA、Mistral、Qwen 全都在用。

> 大白话：LayerNorm 是"把全班成绩调整到均分 70 分、标准差 10 分"。RMSNorm 是"只调整标准差到 10 分，不调均分"。后者更简单更快，而且对 Transformer 来说效果一样好。

#### Q: 为什么要乘以可学习的缩放 weight？

核心原因：**归一化后向量的"尺度"被固定了，但模型需要能自由调整每个维度的重要性。**

**没有 weight 会怎样？**

假设一个 512 维的向量，归一化后每个维度的 RMS = 1。这意味着所有维度被"一视同仁"地压到了同一个量级。但问题是：

- 某些维度可能对下游 Attention 特别重要，需要放大
- 某些维度可能噪声较多，需要缩小
- 不同层需要不同的缩放策略

如果没有 `self.weight`，模型没有任何手段调整这些——它被锁死在"每个维度贡献相等"的状态。

**weight 做了什么？**

```python
self.weight = nn.Parameter(torch.ones(dim))  # 初始化为全 1
```

初始化为 1 意味着**一开始不改变任何东西**（和没加一样）。但在训练过程中：

- 某些维度的 weight 会变大 → 模型学到"这个维度重要，放大它"
- 某些维度的 weight 会变小 → 模型学到"这个维度不重要，压低它"
- 每一层、每个维度的 weight 都是**独立可学习**的

**为什么不加偏置 β？**

LayerNorm 有 `γ`（缩放）和 `β`（平移）两个参数。RMSNorm 只保留了 `γ`（缩放），去掉了 `β`（平移）。原因是：

- 减均值已经让数据中心化了，再加平移（β）意义不大
- 去掉 β 少了一半参数，计算更快
- 实验证明去掉 β 对性能几乎无影响

**总结**

| 组件 | 作用 | 类比 |
|------|------|------|
| `x / RMS(x)` | 消除量纲差异，让向量"能量"归一 | 把全班成绩调到标准差 10 |
| `self.weight` | 让模型自由调整每个维度的重要性 | 给不同科目设置不同的加权系数 |

> 大白话：想象你把全班成绩标准化到均分 0、标准差 1。标准化之后，你发现数学好的同学在数学维度上得分高，但你可能想**额外加权**数学维度（因为数学是核心能力）。`self.weight` 就是这个"加权旋钮"——归一化负责消除量纲差异，weight 负责让模型自己决定每个维度该占多大比重。

---

### 3.2 RoPE — 把位置信息"旋转"进去

RoPE 是 **Rotary Position Embedding** 的缩写。它的核心思想是：**不修改 token 向量的值，而是用旋转矩阵把位置信息编码进去**。

这个过程分两步：
1. **预计算**：提前算好所有位置的正余弦值（`precompute_freqs_cis`）
2. **应用**：在每次 Attention 计算前施加旋转（`apply_rotary_pos_emb`）

#### 3.2.1 预计算：precompute_freqs_cis

**位置**：第 1134 行

```python
def precompute_freqs_cis(dim, end, rope_base, rope_scaling=None):
    # 频率公式: freq_i = 1 / (rope_base ^ (2i / dim))
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    # ... YaRN 缩放逻辑 ...
    t = torch.arange(end)  # 位置索引: [0, 1, 2, ..., end-1]
    freqs = torch.outer(t, freqs)  # 外积: [end, dim//2]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
```

**频率是怎么算的？**

假设 `dim=64`（每个注意力头的维度），`rope_base=1,000,000`：

```
维度索引 i:        0     1     2   ...    31
频率 f_i:         1.0  0.65  0.42  ...  1e-6（极小）
角速度（快慢）:   快←────────────────→慢
```

| 维度区间 | 角速度 | 作用 |
|---------|:------:|------|
| i=0（最快） | 1 rad/token | 区分相邻 token 的精确位置 |
| i=31（最慢） | 1e-6 rad/token | 区分 10000+ token 外的远距离位置 |

**为什么 rope_base 越大越好？**

最小频率 = `1/rope_base`，对应的"波长"（旋转一圈需要的 token 数）= `2π × rope_base`：

| rope_base | 最慢维度波长 | 能区分的最大距离 |
|-----------|:----------:|:--------------:|
| 10,000 | ~62,832 tokens | ~4K |
| 100,000 | ~628,320 tokens | ~32K |
| 1,000,000 | ~6,283,185 tokens | ~100K |

MiniMind 用 1,000,000 是因为它支持 32768 的超长上下文。

> 大白话：把 RoPE 想象成钟表的时针和秒针。秒针转得快，能精确到秒；时针转得极慢，能告诉你现在是几点。rope_base 越大，时针转得越慢，模型就能区分越远的 token 对。

#### 3.2.2 应用：apply_rotary_pos_emb

**位置**：第 1371 行

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        # [x0, x1, x2, x3] → [-x2, -x3, x0, x1]
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**为什么只作用于 Q 和 K，不作用于 V？**

因为注意力分数的计算是 `Q × K^T`。位置信息只需要影响"这个 token 有多关注那个 token"，而不需要影响 value 的语义内容。数学上：

```
不加 RoPE:  Attention = softmax(Q × K^T / √d) × V
加 RoPE:   Q_rot = Q × cos + rotate(Q) × sin
           K_rot = K × cos + rotate(K) × sin
           Attention = softmax(Q_rot × K_rot^T / √d) × V
```

RoPE 的旋转满足：`Q_rot(m) × K_rot(n)^T = f(Q, K, m-n)`，即**注意力分数只依赖 token 之间的相对位置（m-n）**，这就是 RoPE 的核心性质。

> 大白话：你问"苹果"在句子的哪里，我旋转一下 Q（查询）和 K（键），旋转量取决于它们各自的绝对位置。内积算出来的分数自动变成"相对位置差"的函数。V（值）不需要旋转——它只负责表达语义，不负责表达位置。

---

### 3.3 Attention — 多头注意力（含 GQA + Flash Attention + KV Cache）

**位置**：第 1399 行

这是整个模型最复杂的模块，它同时支持三种工作模式：
- **训练阶段**：Flash Attention 加速
- **推理 Prefill 阶段**：一次性处理输入 prompt，构建 KV Cache
- **推理 Decoding 阶段**：利用 KV Cache，逐 token 生成

#### 3.3.1 初始化

```python
class Attention(nn.Module):
    def __init__(self, args):
        self.n_local_heads    = args.num_attention_heads        # 8
        self.n_local_kv_heads = args.num_key_value_heads        # 2 ← GQA！
        self.n_rep            = self.n_local_heads // self.n_local_kv_heads  # 4，每组 4 个 Q 共享 1 对 KV
        self.head_dim         = args.hidden_size // args.num_attention_heads  # 512/8 = 64

        self.q_proj = nn.Linear(512, 8×64, bias=False)    # 512 → 512
        self.k_proj = nn.Linear(512, 2×64, bias=False)    # 512 → 128  ← 只有 2 个 KV 头！
        self.v_proj = nn.Linear(512, 2×64, bias=False)    # 512 → 128
        self.o_proj = nn.Linear(8×64, 512, bias=False)    # 512 → 512
```

**关键观察**：
- Q 投影输出维度 = `n_local_heads × head_dim = 8 × 64 = 512`
- K 投影输出维度 = `n_local_kv_heads × head_dim = 2 × 64 = 128` ← 少很多！
- V 同理只有 128 维

这就是 GQA 的核心：**Q 头全部独立（8 个），K/V 头大幅缩减（2 个）**。

#### 3.3.2 GQA 如何工作：repeat_kv

**位置**：第 1385 行

```python
def repeat_kv(x, n_rep):
    # x: [batch, seq, 2, 64]  只有 2 个 KV 头
    # n_rep = 4
    bs, slen, n_kv_heads, head_dim = x.shape
    # 在 KV 头维度后面扩一维 → [batch, seq, 2, 1, 64]
    # expand 到 4 → [batch, seq, 2, 4, 64]
    # reshape → [batch, seq, 8, 64]  8 个 KV 头
    return x[:,:,:,None,:].expand(bs, slen, n_kv_heads, n_rep, head_dim)\
           .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
```

**为什么用 expand + reshape 而不是 repeat_interleave？**
- `expand` 不复制内存，它只是告诉 PyTorch"这个维度可以广播"——**零显存开销**
- `repeat_interleave` 会真的在内存中复制数据

#### 3.3.3 forward() 的两条计算路径

```python
def forward(self, x, position_embeddings, past_key_value=None,
            use_cache=False, attention_mask=None):

    bsz, seq_len, _ = x.shape

    # ① 线性投影
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

    # ② 重塑 + RoPE
    xq = xq.view(bsz, seq_len, 8, 64)   # Q: 8 个头
    xk = xk.view(bsz, seq_len, 2, 64)   # K: 2 个头
    xv = xv.view(bsz, seq_len, 2, 64)   # V: 2 个头
    cos, sin = position_embeddings
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # 施加 RoPE

    # ③ KV Cache：拼接历史缓存 + 当前输入
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)
        xv = torch.cat([past_key_value[1], xv], dim=1)
    past_kv = (xk, xv) if use_cache else None

    # ④ GQA 广播：将 2 个 KV 头扩展到 8 个 + 维度转置
    xq, xk, xv = (
        xq.transpose(1, 2),                    # [batch, 8, seq, 64]
        repeat_kv(xk, n_rep=4).transpose(1, 2), # [batch, 8, seq, 64]
        repeat_kv(xv, n_rep=4).transpose(1, 2)  # [batch, 8, seq, 64]
    )

    # ⑤ 注意力计算 — 两条路径
    if self.flash and seq_len > 1 and past_key_value is None:
        # 路径 A: Flash Attention（训练/Prefill）
        output = F.scaled_dot_product_attention(
            xq, xk, xv, dropout_p=...,
            is_causal=True    # ← 自动因果掩码
        )
    else:
        # 路径 B: 手动计算（Decoding）
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(64)  # QK^T/√d
        scores += causal_mask                                 # 不能看未来
        scores = F.softmax(scores, dim=-1)
        output = scores @ xv

    # ⑥ 合并多头 + 输出投影
    output = output.transpose(1, 2).reshape(bsz, seq_len, 512)
    output = self.o_proj(output)
    return output, past_kv
```

**两条路径的分工**：

| 路径 | 触发条件 | 场景 | 注意力矩阵形状 |
|------|---------|------|:------------:|
| A: Flash Attention | 训练 / Prefill（`seq_len>1 && 无 cache`） | 一次性处理整段文本 | N×N 方阵 |
| B: 手动计算 | Decoding（`seq_len=1 && 有 cache`） | 逐 token 生成 | 1×L 长条 |

**为什么 Decoding 时不能走 Flash Attention？**

Flash Attention 的 `is_causal=True` 要求 Q 和 K 的 seq_len 相等（N×N），但 Decoding 时 Q 只有 1 个新 token，而 K 是之前所有 L 个 token 的缓存。`[1, 64] × [L, 64]^T` 不是方阵，必须手动计算。

#### Q: 为什么先重塑再加 RoPE，而不是 RoPE 后再重塑？

因为 **RoPE 是按"单个头的维度"设计的旋转操作**，它要求输入的最后两个维度是 `(head_dim//2, 2)` 这样的配对结构。

**如果先 RoPE 再重塑会怎样？**

```python
# 错误顺序
xq = self.q_proj(x)                    # [batch, seq, 512]  ← 一整坨
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # ❌ 按 512 维切半旋转
xq = xq.view(bsz, seq_len, 8, 64)      # 再拆成 8 个头
```

两个致命问题：

**问题一：cos/sin 形状不匹配**

先追踪 `precompute_freqs_cis(dim=64)` 生成的 cos/sin 形状：

```python
# precompute_freqs_cis 内部：
freqs = 1.0 / (rope_base ** (torch.arange(0, 64, 2)[:32].float() / 64))  # [32]
t = torch.arange(end)          # [end]
freqs = torch.outer(t, freqs)  # [end, 32]
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # [end, 64]
```

在 `MiniMindModel.forward()` 中切片后，cos/sin 形状为 `[seq_length, 64]`，比如 `[10, 64]`。

`apply_rotary_pos_emb` 内部做了 `cos.unsqueeze(1)`，把 `[10, 64]` 变成 `[10, 1, 64]`。

正确顺序的广播：

```
q:   [batch=1, seq=10, heads=8,  head_dim=64]
cos: [      1, seq=10,     1,     64]          ← unsqueeze 后
───────────────────────────────────────
广播后: [1, 10, 8, 64]  ✅ 完美匹配
```

如果先 RoPE 再重塑：

```
q:   [batch=1, seq=10, 512]       ← 还没拆头
cos: [      1, seq=10, 64]        ← unsqueeze 后
───────────────────────────────
广播: [1, 10, 512] vs [1, 10, 64]
                         ↑
                    512 ≠ 64，对不上！❌
```

512 和 64 不兼容，PyTorch 会直接报 **RuntimeError: The size of tensor a (512) must match the size of tensor b (64)**。

**问题二：旋转语义错误（即使形状能对上也是错的）**

即使我们假设 cos/sin 能 somehow 对上（比如手动 pad 到 512），旋转的语义也是错的。

先看 `rotate_half` 和 RoPE 到底在做什么：

```python
def rotate_half(x):
    # [x0, x1, ..., x31, x32, x33, ..., x63]
    # → [-x32, -x33, ..., -x63, x0, x1, ..., x31]
    return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

q_embed = (q * cos) + (rotate_half(q) * sin)
```

这是一个**二维旋转**的等价实数形式。对 q 的 64 维向量，它是这样配对的：

```
维度:  [  0,   1,   2,  ...,  31,  32,  33,  ...,  63 ]
配对:  (0,32) (1,33) (2,34) ... (31,63)
         ↕     ↕     ↕          ↕
       各自独立做 2D 旋转，旋转角度由 cos/sin 决定
```

也就是说，64 维被切成 **32 对**，每对独立旋转。RoPE 的数学保证是：

```
q_rotated(m) · k_rotated(n) = f(q, k, m-n)
```

即注意力分数只依赖**相对位置差**。这个性质的前提是：**每对维度的旋转是独立的、且配对关系是固定的**。

现在看如果把 8 个头的 512 维当成一个整体来 rotate_half：

```
512 维: [  0,   1,  ..., 255, 256, 257, ..., 511 ]
配对:   (0,256) (1,257) ... (255,511)
```

问题来了：

| 原本的配对（正确） | 错误的配对 |
|---|---|
| 头 0 的维度 0 ↔ 头 0 的维度 32 | 头 0 的维度 0 ↔ **头 1 的维度 0** |
| 头 0 的维度 1 ↔ 头 0 的维度 33 | 头 0 的维度 1 ↔ **头 1 的维度 1** |
| ... | ... |
| 头 0 的维度 31 ↔ 头 0 的维度 63 | 头 0 的维度 31 ↔ **头 1 的维度 31** |

**跨头配对了！** 头 0 的 Q 向量和头 1 的 Q 向量被混在一起做旋转。这意味着：

1. 头 0 的位置编码"污染"了头 1 的位置编码
2. 每个头不再独立编码位置信息
3. `q_rot(m) · k_rot(n) = f(q, k, m-n)` 这个性质被打破——注意力分数不再只依赖相对位置差

结果就是：模型学到的"每个头从不同角度看问题"的能力被破坏，注意力计算的位置感知完全混乱。

**正确顺序的逻辑**

```python
# 正确顺序
xq = self.q_proj(x)                    # [batch, seq, 512]
xq = xq.view(bsz, seq_len, 8, 64)     # [batch, seq, 8, 64]  ← 拆成 8 个头
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # cos/sin: [seq, 64]，广播到 8 个头
```

| 步骤 | Q 的形状 | cos/sin 形状 | 操作 |
|------|---------|-------------|------|
| 重塑后 | `[batch, seq, 8, 64]` | — | — |
| RoPE | `[batch, seq, 8, 64]` | `[seq, 64]` | 按最后一维 64 切半，配对旋转 |

cos/sin 的 64 维广播到 8 个头 × 64 维，每个头独立拿到相同的位置编码——这正是 RoPE 的设计意图。

> 大白话：RoPE 就像给每个头的 Q/K 戴一副"位置手环"。你得先把 8 个头分开（重塑），然后每人戴一副（RoPE）。如果 8 个头还混在一起就戴手环，那手环是按"一整条胳膊"的尺寸做的，戴不上去。而且强行戴上去的话，张三的手环会绑到李四手上——每个头的位置编码就全乱了。

#### Q: 如果用 precompute_freqs_cis(dim=512) 先 RoPE 再分头，会怎样？

形状能对上，但语义有两个致命问题。

**问题一：配对关系跨头（和之前一样）**

`rotate_half` 按最后一维切半配对：

```
512 维: [  0,   1,  ..., 255, 256, 257, ..., 511 ]
配对:   (0,256) (1,257) ... (255,511)
```

拆成 8 个头后，配对 `(0, 256)` 意味着**头 0 的维度 0** 和**头 4 的维度 0** 配对旋转——跨头配对，和之前一样破坏每个头独立编码位置的性质。

**问题二：频率谱完全错乱**

`precompute_freqs_cis` 的频率公式是 `freq_i = 1 / (rope_base ^ (2i / dim))`。`dim` 从 64 变成 512 后，频率完全变了：

| 维度索引 i | dim=512（错误） | dim=64（正确） | 差异 |
|-----------|:---:|:---:|------|
| i=0 | 1.0 | 1.0 | 相同 |
| i=1 | 1/1e6^0.0039 ≈ 0.991 | 1/1e6^0.03125 ≈ 0.931 | 差 6% |
| i=15 | 1/1e6^0.0586 ≈ 0.764 | 1/1e6^0.46875 ≈ 0.340 | **差 2.2 倍** |
| i=31 | 1/1e6^0.121 ≈ 0.546 | 1/1e6^0.96875 ≈ 0.107 | **差 5 倍** |
| i=32~255 | 存在 | — (dim=64 只有 32 对) | 多出 224 对 |

三个后果：

1. **高频维度转得太慢**：i=0~1 的频率从 0.931 降到 0.991，区分相邻 token 的能力变弱
2. **低频维度转得太快**：i=31 的频率从 0.107 升到 0.546（波长从 ~59 降到 ~11.5），区分远距离 token 的能力丢失
3. **多出 224 对无意义维度**：RoPE 的 32 对是精心设计的，不是越多越好

**总结**

| 方案 | 形状 | 配对 | 频率谱 | 结果 |
|------|:----:|:----:|:------:|------|
| dim=64，先分头再 RoPE | ✅ | ✅ 头内配对 | ✅ 正确 | **正确** |
| dim=512，先 RoPE 再分头 | ✅ | ❌ 跨头配对 | ❌ 频率错乱 | **错误** |
| dim=64，先 RoPE 再分头 | ❌ 报错 | — | — | **直接崩** |

> 大白话：RoPE 的 `dim` 参数必须等于 `head_dim`（64），不能等于 `hidden_size`（512）。因为 RoPE 是给"每个头"戴手环，不是给"整条胳膊"戴。即使用 `dim=512` 绕过了形状检查，手环的尺寸（频率）也是按 512 维设计的，套在 64 维的头上完全不合适——转太快或太慢，位置感知全乱。

#### 3.3.4 GQA 的三个变体

| 变体 | Q 头数 | KV 头数 | 显存节省 | 代表模型 |
|------|:-----:|:------:|:-------:|---------|
| MHA (Multi-Head) | 8 | 8 | 0%（基线） | 原始 Transformer |
| GQA (Grouped-Query) | 8 | 2 | **75%** | LLaMA 2/3, MiniMind |
| MQA (Multi-Query) | 8 | 1 | 87.5% | PaLM, Gemini |

GQA 是 MHA 和 MQA 的折中：既大幅节省 KV Cache 显存，又不至于因为 KV 头太少导致训练不稳定。

> 大白话：8 个 Q 头就像 8 个分析师，每人从不同角度看问题。传统 MHA 给每个分析师配一个专属的"记忆助理（K/V）"。GQA 是让 4 个分析师共用 1 个助理——助理的记性一样好，但省了 3 个人的工资（显存）。MQA 更进一步，让 8 个分析师共用 1 个助理——太省了，但助理压力太大，训练容易崩。

---

### 3.4 FeedForward — SwiGLU 前馈网络

**位置**：第 1563 行

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        # intermediate_size = hidden_size × 8/3  →  512 × 8/3 ≈ 1365
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # 三个线性层（SwiGLU 需要 3 个矩阵）
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  # [512, 1365]
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)  # [512, 1365]
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)  # [1365, 512]
        self.act_fn = SiLU  # silu(x) = x × sigmoid(x)

    def forward(self, x):
        # SwiGLU(x) = down(SiLU(gate(x)) ⊙ up(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**SwiGLU 的数据流**：

```
输入 x: [batch, seq_len, 512]

     ┌─── gate_proj ──→ [..., 1365] ──→ SiLU ──┐
x ───┤                                         ├─→ ⊙（逐元素乘）──→ down_proj ──→ [..., 512]
     └─── up_proj ────→ [..., 1365] ───────────┘
```

**传统 FFN vs SwiGLU**：

| | 传统（2 矩阵） | SwiGLU（3 矩阵） |
|------|:----------:|:-----------:|
| 结构 | `down(ReLU(up(x)))` | `down(SiLU(gate(x)) ⊙ up(x))` |
| 参数量 | `2 × hidden × 4×hidden = 8H²` | `3 × hidden × (8/3)×hidden = 8H²` |
| 激活函数 | ReLU（硬截断） | SiLU（平滑门控） |
| 效果 | 基线 | 更好（门控机制 + 平滑梯度） |

**为什么中间维度是 8/3 倍而不是传统的 4 倍？**

LLaMA 的设计原则：**换了激活函数，但要保持总参数量不变**。

```
传统 FFN:  2 个矩阵 × hidden × (4×hidden) = 8H²
SwiGLU:   3 个矩阵 × hidden ×   I          = 3HI

令 3HI = 8H² → I = (8/3)H
```

这样 SwiGLU 的参数量 = 传统 FFN 的参数量，公平对比。

**SiLU 是什么？**

```
SiLU(x) = x × σ(x) = x / (1 + e^(-x))
```

和 ReLU 对比：
- ReLU: `x>0 时 =x, x<0 时 =0` → 负数区梯度为 0，"神经元死亡"
- SiLU: 负数区也有微小梯度 → 平滑、不截断、训练更稳定

> 大白话：传统 ReLU-FFN 就像一个"要么全部通过、要么全部阻断"的开关。SwiGLU 加上一个"门控"机制，让部分信息以不同比例通过——gate 分支负责决定"通过多少"，up 分支负责提供"信息内容"，两者相乘再映射回去。这比简单粗暴的 ReLU 更精细。

---

### 3.5 MoE — 混合专家

MoE 把 FFN 拆成多个"专家"，每个 token 只激活其中少数几个，实现"参数多、计算少"。

#### 3.5.1 MoEGate — 路由器（Router）

**位置**：第 1647 行

```python
class MoEGate(nn.Module):
    def __init__(self, config):
        self.top_k = config.num_experts_per_tok  # 2
        self.n_routed_experts = config.n_routed_experts  # 4
        # 路由器权重矩阵：[4, 512]，每行是"专家画像"
        self.weight = nn.Parameter(torch.empty((4, 512)))

    def forward(self, hidden_states):
        # ① 计算每个 token 在每个专家上的得分
        logits = F.linear(hidden_states, self.weight)  # [N, 512] × [4, 512]^T → [N, 4]
        scores = logits.softmax(dim=-1)                 # [N, 4] 概率

        # ② 选 Top-2 专家
        topk_weight, topk_idx = torch.topk(scores, k=2, dim=-1)

        # ③ 归一化：让选中的 2 个专家权重加起来 = 1
        if self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        # ④ 辅助损失：防止所有 token 都涌向同一个专家
        aux_loss = (Pi * fi).sum() * self.alpha
        #     Pi: 每个专家的平均门控分数
        #     fi: 每个专家的实际负载（被选中次数 × 归一化）
        #     如果某个专家被过度使用（fi 大）且门控分也高（Pi 大），辅助损失就大

        return topk_idx, topk_weight, aux_loss
```

**辅助损失（Auxiliary Loss）是 MoE 训练能正常进行的核心机制**。没有它，模型会发现"把 90% 的 token 发给专家 A"梯度更新最快，于是其他 3 个专家"饿死"，MoE 退化成普通 FFN。

```
aux_loss = Σ(Pᵢ × fᵢ) × α

Pᵢ = 专家 i 的平均门控分数（模型想选它的意愿）
fᵢ = 专家 i 的实际负载比例（乘 n_experts 归一化，均衡时应 = 1.0）
α   = 0.01（辅助损失权重，防止它主导总 loss）
```

> 大白话：MoEGate 就像一个分诊台。每个 token（"患者"）进来，分诊台给它对 4 个专家各打一个匹配分，选出分数最高的 2 个专家。但分诊台可能偷懒——把所有患者都扔给专家 A（因为 A 刚好在初始阶段表现好一点）。辅助损失就是：如果分诊台偏心，扣分！强迫它把患者均匀分配。

#### Q: Pᵢ 和 fᵢ 具体在哪里计算的？用数字走一遍

Pᵢ 和 fᵢ 不在 `MoEGate.forward()` 的前半段（路由选择部分），而是在后半段的**辅助损失计算**里。代码有两个分支：

**模式 1：批次级辅助损失（经典 Switch Transformer 方式，`seq_aux=False`）**

位置：`model_minimind.py:2217-2256`

```python
# 步骤 1：把选中的专家索引转成 one-hot
mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)

# 步骤 2：fᵢ = 专家 i 的实际选中频率
ce = mask_ce.float().mean(0)        # 形状: [n_routed_experts]

# 步骤 3：Pᵢ = 专家 i 的平均门控概率
Pi = scores_for_aux.mean(0)         # 形状: [n_routed_experts]

# 步骤 4：归一化负载因子
fi = ce * self.n_routed_experts     # 均衡时 fi = 1.0

# 步骤 5：最终辅助损失
aux_loss = (Pi * fi).sum() * self.alpha
```

**模式 2：序列级辅助损失（`seq_aux=True`）**

位置：`model_minimind.py:2127-2216`

```python
# ce = 每个专家在每条序列中被选中的归一化频率（相当于 fi）
ce = torch.zeros(bsz, self.n_routed_experts, ...)
ce.scatter_add_(1, topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * aux_topk, ...)).div_(
    seq_len * aux_topk / self.n_routed_experts)
# 归一化后 ce 的期望值 = 1.0（完全均衡）

# scores_for_seq_aux.mean(dim=1) = 每条序列中每个专家的平均门控概率（相当于 Pi）
aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
```

**用具体数字走一遍（批次级）**

假设 batch=1, seq_len=4, n_experts=4, top_k=2：

```
token 0 选了专家 [0, 1]    → 专家 0 得 1 票，专家 1 得 1 票
token 1 选了专家 [0, 2]    → 专家 0 得 1 票，专家 2 得 1 票
token 2 选了专家 [1, 3]    → 专家 1 得 1 票，专家 3 得 1 票
token 3 选了专家 [0, 1]    → 专家 0 得 1 票，专家 1 得 1 票
```

fᵢ（实际负载）计算：

```
总选票 = 4 tokens × 2 = 8 张
专家 0 被选 3 次 → ce[0] = 3/8 = 0.375
专家 1 被选 3 次 → ce[1] = 3/8 = 0.375
专家 2 被选 1 次 → ce[2] = 1/8 = 0.125
专家 3 被选 1 次 → ce[3] = 1/8 = 0.125

fi = ce × 4（n_experts）:
fi = [1.5, 1.5, 0.5, 0.5]
     ↑    ↑    ↑    ↑
   过载  过载  闲置  闲置（均衡时应该都是 1.0）
```

Pᵢ（平均门控概率）计算：

```
假设 softmax 后每个 token 对 4 个专家的概率分布：
token 0: [0.4, 0.3, 0.2, 0.1]
token 1: [0.5, 0.1, 0.3, 0.1]
token 2: [0.1, 0.4, 0.1, 0.4]
token 3: [0.3, 0.4, 0.2, 0.1]

Pi = 四行求平均:
Pi = [0.325, 0.3, 0.2, 0.175]
      ↑
   专家 0 的平均门控分数最高（模型最"想"用它）
```

辅助损失：

```
aux_loss = (Pi × fi).sum() × α
         = (0.325×1.5 + 0.3×1.5 + 0.2×0.5 + 0.175×0.5) × 0.01
         = (0.4875 + 0.45 + 0.1 + 0.0875) × 0.01
         = 1.125 × 0.01
         = 0.01125
```

如果所有 token 都涌向专家 0（极端不均衡）：

```
fi = [4.0, 0.0, 0.0, 0.0]
Pi = [0.8, 0.1, 0.05, 0.05]

aux_loss = (0.8×4.0 + 0.1×0 + 0.05×0 + 0.05×0) × 0.01
         = 3.2 × 0.01 = 0.032  ← 比均衡时大得多！
```

辅助损失通过反向传播迫使模型把 token 均匀分配。

**总结**

| 变量 | 含义 | 计算方式 | 代码位置 |
|------|------|---------|---------|
| **Pᵢ** | 专家 i 的平均门控概率（模型"想"用它的程度） | `scores.mean(0)` — 对所有 token 求平均 | L2242 |
| **fᵢ** | 专家 i 的实际负载比例 × n_experts | `ce.mean(0) × n_experts` — 统计实际被选中次数 | L2235, L2248 |
| **aux_loss** | 辅助损失 | `(Pi × fi).sum() × α` | L2256 |

> 大白话：Pᵢ 就是"群众基础"——所有 token 平均有多想选这个专家。fᵢ 就是"实际工作量"——这个专家实际干了多少活。如果一个专家群众基础好（Pᵢ 大）又干了很多活（fᵢ 大），说明它被过度依赖了，辅助损失就会给一个大的惩罚值，逼着模型把活分给别人。

#### 3.5.2 MOEFeedForward — 专家群的"总调度"

**位置**：第 2266 行

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        # 4 个独立的专家（每个都是完整的 SwiGLU FFN）
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(4)])
        self.gate = MoEGate(config)
        # 1 个共享专家（所有 token 必过，类似 DeepSeek-MoE）
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config)])

    def forward(self, x):
        identity = x  # 保存一份原始输入（给共享专家用）
        # ① 路由：决定每个 token 去哪些专家
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # ② 训练模式：扩展 token（通过 repeat_interleave）再分发计算
        if self.training:
            x = x.repeat_interleave(2, dim=0)  # 每个 token 复制 2 份（top_k=2）
            y = torch.empty_like(x)
            for i, expert in enumerate(self.experts):
                mask = (flat_topk_idx == i)
                y[mask] = expert(x[mask])       # 属于专家 i 的 token 送进去
            # 加权求和：token 0 的结果 = 权重[0]×专家A + 权重[1]×专家C
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)

        # ③ 推理模式：更高效的排序+分段处理（moe_infer）
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight)

        # ④ 加上共享专家的输出
        if config.n_shared_experts > 0:
            y = y + self.shared_experts[0](identity)

        self.aux_loss = aux_loss
        return y
```

**训练 vs 推理的不同处理方式**：

| | 训练模式 | 推理模式（moe_infer） |
|------|---------|-------------------|
| 核心操作 | `repeat_interleave` 复制 token | `argsort` + `bincount` 分段处理 |
| 专家计算 | for 循环逐个处理 | 同样 for 循环，但无冗余复制 |
| 加权融合 | `view + sum(dim=1)` | `scatter_add_` 原位累加 |
| 显存 | 高（token 复制 2 份） | 低 |
| 速度 | 快（大批量并行） | 足够快 |

#### Q: 为什么训练模式不能用 moe_infer 的方式？

理论上可以，但有三个原因导致训练不能用 `moe_infer` 的 argsort/bincount 方式。

**原因一：argsort 不可微，梯度链断了**

`moe_infer` 的核心操作：

```python
idxs = flat_expert_indices.argsort()                          # 排序（不可微）
tokens_per_expert = flat_expert_indices.bincount().cumsum(0)  # 统计+前缀和（不可微）
```

这两个是**离散整数操作**，PyTorch autograd 无法对它们求梯度。反向传播到这里梯度就断了——模型无法学到"应该把 token 路由给哪个专家"。

训练模式的 boolean mask 方式：

```python
mask = (flat_topk_idx == i)       # bool 比较，梯度不经过它
expert_out = expert(x[mask])      # 梯度通过 expert 参数正常回传
y[mask] = expert_out              # 梯度通过 y 继续流动
```

`mask` 从已 detach 的 `topk_idx` 产生，梯度不需要经过 mask 本身，而是通过 `expert(x[mask])` 直接作用在专家参数上。这条路是通的。

**原因二：DDP 空专家问题需要特殊处理**

训练模式代码里有一个关键细节：

```python
if expert_out.shape[0] > 0:
    y[flat_topk_idx == i] = expert_out.to(y.dtype)
else:
    # 防止空专家导致梯度图断裂
    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
```

当某个专家在当前 batch 中没有被任何 token 选中时，它的参数**没有任何梯度流过**。在 DDP（分布式数据并行）训练中，这会导致：

- 该专家的参数不同步（其他 GPU 上可能有 token 选了这个专家）
- 梯度桶（gradient bucket）中出现空值，DDP 同步报错

所以用 `0 * sum(p.sum() for p in expert.parameters())` 强制制造一个不改变数值但包含该专家所有参数梯度的"假梯度"。

`moe_infer` 里用 `continue` 跳过空专家，训练时这会直接导致 DDP 崩溃。

**原因三：scatter_add_ 的梯度行为不如直接索引清晰**

虽然 `scatter_add_` 在 PyTorch 中是可微的，但它的梯度是**累加**语义。当多个专家为同一个 token 计算结果时，梯度回传需要正确分派到每个专家。实际调试和数值稳定性都不如训练模式的直接赋值方式好控制。

训练模式的加权融合更直白：

```python
y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
```

每一步都是标准张量运算，梯度路径清晰、可调试。

> 大白话：训练时模型需要"知道自己错在哪里"（梯度回传）。moe_infer 的 argsort/bincount 就像快递员分拣包裹——分完就丢了记录，梯度回不去了。训练模式的 repeat_interleave 虽然笨一点（把每个 token 复制 2 份），但每一步都有清晰的"收据"，梯度能原路返回。加上 DDP 多卡训练时，空专家必须"假装收到梯度"否则会报错——moe_infer 的 `continue` 跳过做不到这一点。

---

### 3.6 MiniMindBlock — Transformer 的单层骨架

**位置**：第 2434 行

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id, config):
        self.self_attn = Attention(config)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, past_key_value, ...):
        # 子层 1: Self-Attention
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(
            self.input_layernorm(hidden_states),  # Pre-Norm！
            position_embeddings, past_key_value, ...
        )
        hidden_states += residual  # 残差连接

        # 子层 2: FFN / MoE
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)  # Pre-Norm！
        )

        return hidden_states, present_kv
```

**Pre-Norm vs Post-Norm**：

原始 Transformer 用的是 Post-Norm（Add → Norm），现代模型全都改成了 Pre-Norm（Norm → Sublayer → Add）。注意代码中 Norm 在 sublayer 之前：

```
Pre-Norm (MiniMind):   x → Norm → Sublayer → + x
Post-Norm (原始):      x → Sublayer → + x → Norm
```

Pre-Norm 的优势：
- 训练更稳定：梯度通过残差连接直通前面各层
- 不需要学习率 warm-up（Post-Norm 需要）
- 是大模型训练的标配（GPT-3、LLaMA 全都用）

#### Q: Qwen2.5 用的是什么 Norm？

Qwen2.5 用的是 **Pre-Norm**（和 MiniMind 一样）。它的架构是标准的 LLaMA 风格：

| 组件 | Qwen2.5 | MiniMind |
|------|---------|----------|
| 归一化 | Pre-Norm (RMSNorm) | Pre-Norm (RMSNorm) |
| FFN | SwiGLU | SwiGLU |
| 注意力 | GQA | GQA |
| 位置编码 | RoPE | RoPE |

这也是目前几乎所有主流开源模型的标准配置——LLaMA 2/3、Mistral、Gemma、DeepSeek 全都是 Pre-Norm + RMSNorm + SwiGLU + GQA + RoPE。原始 Transformer 的 Post-Norm + LayerNorm + ReLU-FFN 已经是"上一代"的设计了。

---

### 3.7 MiniMindModel — 模型主干

**位置**：第 2470 行

```python
class MiniMindModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab_size=6400, embedding_dim=512)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(8)])
        self.norm = RMSNorm(512)

        # 预计算 RoPE（只需算一次，存为 buffer，不参与梯度）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=64, end=32768, rope_base=1e6
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None,
                past_key_values=None, use_cache=False):
        # ① Token Embedding
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # ② 当前位置的 RoPE 切片
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # ③ 逐层传播
        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(hidden_states, position_embeddings,
                                           past_key_value=past_kv, ...)
            presents.append(present)

        # ④ 最终归一化
        hidden_states = self.norm(hidden_states)

        # ⑤ 汇总所有 MoE 层的辅助损失
        aux_loss = sum([l.mlp.aux_loss for l in self.layers
                        if isinstance(l.mlp, MOEFeedForward)])

        return hidden_states, presents, aux_loss
```

**关键设计**：
- **RoPE 预计算**：`register_buffer` 把 cos/sin 矩阵存在模型里，不用每次都算，也不用存到 checkpoint 文件（`persistent=False`）
- **位置编码切片**：Decoding 时 `start_pos > 0`，只取当前位置的 RoPE，不需要整个矩阵
- **辅助损失汇总**：`aux_loss` 只在 MoE 模式有值（Dense 模式下为 0），会被外层的 `MiniMindForCausalLM` 加到总损失中

---

### 3.8 MiniMindForCausalLM — 最外层包装

**位置**：第 2540 行

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(512, 6400, bias=False)

        # 权重绑定（Weight Tying）
        self.model.embed_tokens.weight = self.lm_head.weight
        # 注意：这不是拷贝！是让两个变量指向同一个 Tensor 对象

    def forward(self, input_ids, attention_mask=None, labels=None, ...):
        # ① 主干网络提取特征
        hidden_states, past_key_values, aux_loss = self.model(...)

        # ② lm_head 映射到词表
        logits = self.lm_head(hidden_states)

        # ③ 计算 Loss
        if labels is not None:
            # Shift: 位置 i 的 logits 预测位置 i+1 的 token
            shift_logits  = logits[..., :-1, :].contiguous()
            shift_labels   = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, 6400),
                                   shift_labels.view(-1),
                                   ignore_index=-100)

        output = CausalLMOutputWithPast(loss=loss, logits=logits, ...)
        output.aux_loss = aux_loss  # MoE 辅助损失附加到输出
        return output
```

**权重绑定（Weight Tying）**是什么？为什么做？

```python
self.model.embed_tokens.weight = self.lm_head.weight
```

这行代码让输入 embedding 矩阵和输出 lm_head 矩阵**共享同一个权重**。

```
原来（不绑定）：
  embed_tokens: [6400, 512]  ← 3.3M 参数
  lm_head:      [512, 6400]  ← 3.3M 参数
  总共: 6.6M 参数

绑定后：
  两者指向同一个 Tensor → 共享 3.3M 参数
```

**为什么可以共享？**

1. **数学对称性**：输入 embedding 把"token ID → 语义向量"，输出 head 把"语义向量 → token 概率"。它们在数学上是逆操作，共享权重让模型学到更一致的表示
2. **省参数**：对于 MiniMind，省了 3.3M / 26M ≈ 12.5% 的参数
3. **训练更稳定**：共享权重提供了一种隐式的正则化

**Shift 对齐是什么？**

```python
shift_logits  = logits[..., :-1, :]   # 去掉最后一个位置
shift_labels   = labels[..., 1:]       # 去掉第一个位置
```

这是因为自回归语言模型的规则：**位置 i 的输出来预测位置 i+1 的 token**。

```
输入:  [你, 好, 世, 界]  ← token IDs
       ↓   ↓   ↓   ↓
模型:  [h0, h1, h2, h3] ← hidden states
       ↓   ↓   ↓   ↓
logits:[L0, L1, L2, L3] ← 每个位置的预测分布

Loss 计算:
L0 预测 "好",  L1 预测 "世",  L2 预测 "界",  L3 丢弃
   shift: labels[1:] = ["好", "世", "界"]
   shift: logits[:-1] = [L0, L1, L2]
```

---

## 四、参数计算：26M 是怎么来的？

以 **Small 配置**（`hidden_size=512, num_hidden_layers=8`）为例，逐模块计算：

### 4.1 Embedding 层

```
embed_tokens:     6400 × 512  = 3,276,800  （约 3.3M）
lm_head:          与 embed_tokens 共享 → 0（额外参数）
```

### 4.2 每层 Transformer Block（8 层）

**Attention 部分（每层）**：
```
Q 投影:  512 × 512 = 262,144
K 投影:  512 × 128 =  65,536   ← GQA 只有 2 个 KV 头！
V 投影:  512 × 128 =  65,536
O 投影:  512 × 512 = 262,144
─────────────────────────
Attention 小计:     655,360
```

**FFN 部分（每层，SwiGLU）**：
```
gate_proj:  512 × 1365 = 698,880
up_proj:    512 × 1365 = 698,880
down_proj:  1365 × 512 = 698,880
─────────────────────────
FFN 小计:             2,096,640
```

**RMSNorm（每层 × 2）**：
```
RMSNorm: 512 × 2 = 1,024
```

**每层总计**：
```
655,360 + 2,096,640 + 1,024 = 2,753,024  （约 2.75M）
```

**8 层总计**：
```
2,753,024 × 8 = 22,024,192  （约 22.0M）
```

### 4.3 最后一层 RMSNorm

```
final RMSNorm: 512
```

### 4.4 总参数量

```
Embedding:    3,276,800
8 层 Block:  22,024,192
Final Norm:         512
─────────────────────────
总计:        25,301,504  ≈ 25.3M
```

**等一等**，MiniMind Small 宣称 26M（0.026B），为什么我们算出 ~25.3M？差在哪里？

差了 `Dropout`（无参数）、`lm_head` 权重绑定（共享不算新参数）。此外还有一些非常小的项（如 config 中的 buffer）。差异在误差范围内，**验证通过**。

> 动手练习：用类似方法算一下 Base 配置（`hidden_size=768, num_hidden_layers=16`）的总参数量，应该接近 104M。

---

## 五、代码结构总览

```
model_minimind.py（2596 行）
│
├── MiniMindConfig（L7-1095）        配置类，定义所有超参数
│
├── RMSNorm（L1111-1131）            RMS 归一化层
│
├── precompute_freqs_cis（L1134-1368） 预计算 RoPE 正余弦值
├── apply_rotary_pos_emb（L1371-1382） 应用 RoPE 到 Q/K
├── repeat_kv（L1385-1396）            GQA 的 KV 复制
│
├── Attention（L1399-1560）           多头注意力（MHA/GQA/Flash/KV Cache）
├── FeedForward（L1563-1644）         SwiGLU 前馈网络
├── MoEGate（L1647-2263）              MoE 门控/路由网络
├── MOEFeedForward（L2266-2432）       稀疏 MoE 专家群
│
├── MiniMindBlock（L2434-2467）        单层 Transformer Decoder
├── MiniMindModel（L2470-2537）        模型主干（不含 LM Head）
├── MiniMindForCausalLM（L2540-2596）  自回归 LM 包装类（含 LM Head + Weight Tying）
```

### 文件依赖关系

```
model_minimind.py
 ├── 被所有训练脚本 import（train_pretrain.py, train_full_sft.py 等）
 ├── 被 eval_llm.py import（推理入口）
 └── 被 serve_openai_api.py import（API 服务）
 
model_lora.py
 └── 被 train_lora.py import（LoRA 注入 + 保存/加载）
```

---

## 六、MiniMind 的三档配置

### Small（26M，默认）

```python
hidden_size=512, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2
```
- 适合学习、快速实验、CPU 推理
- 8 层 × 8 头 GQA

### Base（104M）

```python
hidden_size=768, num_hidden_layers=16, num_attention_heads=12, num_key_value_heads=4
```
- 效果更好的版本
- 16 层 × 12 头 GQA

### MoE（145M）

```python
hidden_size=640, num_hidden_layers=8, use_moe=True
n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1
```
- 参数多但计算少（每次只激活 2/4 专家）
- 路由专家 4 个 + 共享专家 1 个

---

## 七、检查你是否真的理解（Q&A）

### 基础

**1. RMSNorm 和 LayerNorm 的核心区别是什么？为什么不减均值？**

答案：LayerNorm 计算 `(x-μ)/σ × γ + β`，RMSNorm 只计算 `x/RMS(x) × γ`。省去了减均值（去中心化）和加偏置 β 两步。实践中发现 Transformer 不需要均值中心化——RMSNorm 用更少的计算达到了和 LayerNorm 相当的训练稳定性，因此计算量减少约 25%，成为 LLaMA 系列的标准选择。

**2. RoPE 为什么只作用于 Q 和 K，不作用于 V？**

答案：注意力分数的计算是 `Q × K^T`，位置信息只需要影响"token A 关注 token B 的强度"，不需要影响 value 本身的语义内容。对 Q 和 K 施加相同的旋转后，`Q_rot(m) × K_rot(n)^T` 自动变成 `f(Q, K, m-n)`，即只依赖相对位置差。

**3. GQA 中 repeat_kv 为什么用 expand+reshape 而不是 repeat_interleave？**

答案：`expand` 不复制内存，只标记"这个维度可以广播"——零显存开销。`repeat_interleave` 会实际在内存中创建数据的副本。对 GQA 这种高频操作，用 expand 在 Decoding 阶段能节省显著的显存和带宽。

**4. SwiGLU 的三个矩阵分别叫什么？中间维度为什么是 hidden_size 的 8/3 倍？**

答案：gate_proj（门控投影）、up_proj（升维投影）、down_proj（降维投影）。8/3 倍来自参数量对齐：传统 ReLU-FFN 有 2 个矩阵（8H² 参数），SwiGLU 有 3 个矩阵（3HI 参数）。令 3HI = 8H² 解得 I = (8/3)H。这样在引入门控机制的同时不增加总参数量。

**5. 什么是权重绑定（Weight Tying）？MiniMind 中绑定了哪两个权重？**

答案：让输入 embedding 矩阵和输出 lm_head 矩阵共享同一个 Tensor 对象。MiniMind 中 `embed_tokens.weight` 和 `lm_head.weight` 是同一个 Tensor。这样做省了 ~12.5% 的参数（3.3M/26M），并且因为 embedding 和 unembedding 互为逆操作，共享权重让模型学到更一致的表示。

### 深入

**6. Attention 的 forward 有两条计算路径，它们分别在什么条件下触发？为什么不同？**

答案：
- 路径 A（Flash Attention）：`seq_len > 1` 且无 KV Cache。用于训练和推理 Prefill。此时 Q 和 K 长度相等，可以用 `F.scaled_dot_product_attention(is_causal=True)` 高效计算 N×N 因果掩码。
- 路径 B（手动计算）：`seq_len = 1` 且有 KV Cache。用于逐 token 生成（Decoding）。此时 Q 长度为 1，K 长度为历史总长 L，形状不是方阵，必须手动计算 `Q × K^T / √d` 并手动拼接因果掩码。

**延伸：因果掩码在计算的什么时候注入？**

两条路径的注入时机不同：

路径 A — **参数注入**，在函数调用时传入：

```python
# L1508-1512
output = F.scaled_dot_product_attention(
    xq, xk, xv,
    is_causal=True    # ← 掩码封装在 fused kernel 内部
)
```

掩码的生成和应用全部在 PyTorch 的 fused kernel 里完成，代码层面看不到掩码矩阵。

路径 B — **softmax 之前手动注入**：

```python
# L1520: 先算原始分数
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
#        形状: [batch, heads, Q_len, K_len]

# L1525-1528: 加上三角掩码（注入点）
scores[:, :, :, -seq_len:] += torch.triu(
    torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
    diagonal=1
)

# L1540: softmax（掩码已经生效）
scores = F.softmax(scores.float(), dim=-1)
```

`triu(-inf, diagonal=1)` 生成上三角矩阵，对角线以上设为 `-inf`。加到 scores 上后，softmax 会让这些位置的权重变为 0——即"不能看未来"。

**seq_len=1 时掩码是 no-op**

当 Decoding 时 `seq_len=1`：

```python
torch.triu(torch.full((1, 1), float("-inf")), diagonal=1)
# 结果: [[0]]   ← 不是 [[-inf]]！
```

`triu` 的 `diagonal=1` 保留严格上三角的元素。对于 1×1 矩阵，位置 `(0,0)` 不在严格上三角内（`0 <= 0-1` 为假），所以被设为 0。

这意味着 `scores[:, :, :, -1:] += [[0]]` 什么都没做。这是正确的：**Decoding 时当前 token 可以 attend to 所有历史 token + 自己，没有"未来 token"需要屏蔽。**

> 大白话：训练时所有 token 同时输入，必须告诉模型"第 5 个词不能看第 6 个词"（因果掩码）。Flash Attention 把这个规则写在函数参数里，PyTorch 内部自动处理。手动计算时，在 scores 算完后、softmax 前，把"未来位置"的分数设为负无穷，softmax 后这些位置权重就变 0 了。而 Decoding 时每次只生成 1 个词，没有"未来"可看，所以掩码是 no-op。

**延伸：加了掩码后计算量变了吗？**

没有，**计算量完全不变**。掩码是在 `Q × K^T` 算完之后才加的：

```
第一步：Q × K^T / √d  → scores [N×N]    ← 计算量 N²×d，掩码还没出现
第二步：+ triu(-inf)   → scores [N×N]    ← 只是把上三角设为 -inf，零开销
第三步：softmax         → weights [N×N]    ← -inf 位置变 0，但矩阵大小没变
第四步：weights × V     → output [N×d]    ← 0×v=0，但矩阵乘法还是跑了
```

掩码的本质是**先算完所有 N×N 的注意力分数，再把"不该看的位置"的分数设为 `-inf`**。softmax 后这些位置权重为 0，第四步乘 V 时 0×v=0 不产生有效输出，但计算本身还是发生了。

为什么不跳过这些计算？因为 GPU 的 Tensor Core 按固定 tile 大小做并行计算，"跳过某些元素"反而不知道怎么高效调度。还不如全部算完，最后清零。

Flash Attention 之所以快，不是因为跳过计算，而是用 **tiling + online softmax** 减少了 HBM（显存）读写次数——这才是 Attention 的真正瓶颈。

| 阶段 | scores 形状 | 计算量 | 掩码作用 |
|------|:-----------:|:------:|---------|
| 训练/Prefill | N×N | O(N²d) | 把上三角设为 -inf，但不减少计算 |
| Decoding | 1×L | O(Ld) | no-op，没有未来可屏蔽 |

> 大白话：掩码就像考完试后老师划掉几道不该做的题——卷子还是全部写完了，只是那几道题的分数不计入总分。真正的计算节省不是来自掩码，而是来自 Decoding 时 Q 只有 1 个 token（1×L vs N×N）。

**延伸：有没有通过因果掩码减少计算量的方案？**

有，而且 Flash Attention 已经在做了。

**Flash Attention + causal：实际会跳过计算**

Flash Attention 内部按 block（tile）处理。当 `is_causal=True` 时，它知道上三角的 block 全部被 mask 掉，**直接跳过不计算**：

```
N=8 的注意力矩阵，按 4×4 分 block：

┌───────┬───────┐
│ Block │ Block │
│ (0,0) │ (0,1) │  ← (0,1) 全在上三角，Flash 直接跳过
├───────┼───────┤
│ Block │ Block │
│ (1,0) │ (1,1) │  ← (1,1) 全在上三角，直接跳过
└───────┴───────┘

实际计算: (0,0), (1,0) → 2/4 = 50%
```

Flash Attention 2 论文明确写了：**causal masking 时跳过被完全 mask 的 block，FLOPs 减少约 50%**。这就是为什么 `is_causal=True` 的 Flash Attention 比 non-causal 还快。

而手动计算的路径 B 不会跳过——GPU 算完所有 N×N 个元素才加掩码。

**其他利用因果结构减少计算的方案**

| 方案 | 策略 | FLOPs | 代价 | 代表模型 |
|------|------|:-----:|------|---------|
| Sliding Window | 每个 token 只 attend 最近 W 个 | O(N×W) | 丢失长距离依赖 | Mistral, Gemma 2 |
| StreamingLLM | 只保留前几个 + 最近 W 个 KV | 不减计算 | 只省 KV Cache 显存 | StreamingLLM |
| Mamba/SSM | 用状态空间模型替代 attention | O(N) | 某些任务效果不如 attention | Mamba, Mamba-2 |
| Hybrid (Mamba+Attn) | SSM + 少量 attention 层 | 介于 O(N) 和 O(N²) 之间 | 架构复杂 | Jamba |

**最实用的答案**：Flash Attention 的 `is_causal=True` 已经在利用因果结构减少计算了（跳过上三角 block），而且没有额外代价。这是目前工业界最主流的做法。

**延伸：Flash Attention 为什么能利用因果掩码减少计算？**

核心在于它**不是先算完整的 N×N 再掩码**，而是**按 block 处理，边算边判断要不要跳过**。

标准注意力 vs Flash Attention 的计算方式：

```
标准注意力:
  Q × K^T → [N×N 完整矩阵]   ← 全部算完
  + triu(-inf)               ← 再清零上三角

Flash Attention:
  Q 切成 [Q₀, Q₁, ..., Qₘ]  每块大小 B×d
  K 切成 [K₀, K₁, ..., Kₙ]  每块大小 B×d

  for 每个 Q 块 Qᵢ:
      for 每个 K 块 Kⱼ:
          if j > i:           ← 关键判断！
              skip            ← 这块全在上三角，直接跳过
          else:
              scores = Qᵢ × Kⱼ^T / √d
              online softmax 更新
              output += softmax(scores) × Vⱼ
```

用 N=8、block_size=4 看跳过效果：

```
         K₀(0-3)    K₁(4-7)
       ┌──────────┬──────────┐
Q₀(0-3)│  计算 ✅  │  跳过 ❌  │  ← Q₀ 只需要 attend K₀
       ├──────────┼──────────┤
Q₁(4-7)│  计算 ✅  │  计算 ✅  │  ← Q₁ 需要 attend K₀ 和 K₁
       └──────────┴──────────┘

计算: 3/4 block → 跳过 1/4
N→∞ 时，跳过比例 → 50%
```

为什么能跳过？因为因果掩码是下三角——位置 i 只能 attend 到 0..i：

```
Q 块处理 [0,1,2,3]，K 块处理 [4,5,6,7]:
  位置 0 最多看到位置 0
  位置 1 最多看到 0-1
  位置 2 最多看到 0-2
  位置 3 最多看到 0-3
  → 没有任何位置需要看 4-7 → 整块跳过
```

判断条件：**K 块的起始位置 > Q 块的结束位置 → 跳过**。

关键使能技术：**online softmax**

标准 softmax 需要看到所有分数才能算（分母是所有 `exp(score)` 的和）。Flash Attention 用 online softmax 逐块增量更新 running_max 和 running_sum，保证逐块计算的结果和一次性算完全矩阵**完全等价**。

```
标准注意力:    FLOPs = N²×d,    显存 = O(N²)
Flash (causal): FLOPs ≈ N²×d/2, 显存 = O(N)  ← 计算和显存都省了
```

> 大白话：标准注意力是"先把整张卷子写完，再划掉不该做的题"。Flash Attention 是"拿到卷子先看题号，发现最后几道大题根本不在我该做的范围内，直接跳过不做"。online softmax 就像"做一道批一道"——不需要等所有题都做完再统一评分，而是边做边更新成绩。

**7. MoE 的辅助损失（Auxiliary Loss）公式是什么？不用会怎样？**

答案：`aux_loss = Σ(Pᵢ × fᵢ) × α`，其中 Pᵢ 是专家 i 的平均门控概率，fᵢ 是专家 i 的实际负载比例（乘 n_experts 归一化，均衡时应 ≈ 1.0）。不用辅助损失会导致"路由崩溃"：模型在训练早期发现某个专家稍微好用一点，就把几乎所有 token 都路由给它，其他专家"饿死"，MoE 退化成普通 FFN。

**8. 为什么 Pre-Norm 比 Post-Norm 训练更稳定？**

答案：Post-Norm 的梯度需要穿过归一化层才能到达前面的网络层，梯度可能衰减。Pre-Norm 中残差连接绕过归一化层直连前面的层，梯度可以"无障碍"回传。这就是为什么 Pre-Norm 不需要学习率 warm-up，而原始 Transformer 的 Post-Norm 需要。

**延伸：用示意图看梯度流**

Post-Norm（原始 Transformer）的梯度路径：

```
x ───────────────────────────────────▶ output
│                                      ↑
▼                                      │
Sublayer(x)                            │
│                                      │
▼                                      │
Add: x + Sublayer(x)                   │
│                                      │
▼                                      │
Norm(·)  ← 梯度必须穿过这里             │
│                                      │
└──────────────────────────────────────┘
```

反向传播时，梯度从 output 回传到 x：

```
output → Norm → Add → Sublayer → x
           ↑
      梯度被 Norm 的 1/σ 缩放
      如果 σ 很大，梯度就被缩得很小
```

8 层堆叠后：

```
output → Norm₈ → ... → Norm₁ → x
          ↑       ↑       ↑
        每过一层 Norm，梯度可能衰减一次
        8 层后：梯度 ≈ (1/σ₁) × (1/σ₂) × ... × (1/σ₈) × 原始梯度
        如果每层 σ≈2，梯度衰减为 原始/256
```

Pre-Norm（MiniMind / LLaMA）的梯度路径：

```
x ───────────────────────────────────▶ output
│                                      ↑
│                                      │
▼                                      │
Norm(x)  ← Norm 在分支上，不在主干道     │
│                                      │
▼                                      │
Sublayer(Norm(x))                      │
│                                      │
▼                                      │
Add: x + Sublayer(Norm(x))  ← 残差直连  │
│                                      │
└──────────────────────────────────────┘
```

反向传播时：

```
output → Add ──→ x          ← 主干道：只有 Add，梯度 = 1，不衰减
           │
           └→ Sublayer → Norm  ← 分支：梯度可以忽略
```

8 层堆叠后：

```
output → Add₈ → ... → Add₁ → x
           ↑       ↑       ↑
        每过一层只有 Add（梯度 = 1）
        8 层后：梯度 = 1 × 1 × ... × 1 = 1（不衰减！）
```

**为什么 Norm 的 1/σ 会导致梯度衰减？**

```python
y = x / σ(x) × γ    # σ(x) = RMS(x) 或标准差
```

反向传播时，梯度经过 Norm 被 `1/σ` 缩放。如果某一层输入能量大（σ 大），梯度就被大幅缩小。

**具体数值对比**（假设 8 层，每层 σ ≈ 1.5）：

```
Post-Norm:  梯度衰减 = (1/1.5)⁸ = 1/25.6 ≈ 3.9%  → 第 1 层只收到原始的 3.9%
Pre-Norm:   梯度衰减 = 1⁸ = 100%                   → 第 1 层收到原始的 100%
```

**为什么 Post-Norm 需要 warm-up？**

训练初期参数随机初始化，输出能量大（σ 很大）。Post-Norm + 大学习率 → 梯度被 σ 大幅缩小但更新步长大 → 不稳定→ 发散。warm-up 先用小学习率让 σ 稳定下来，再逐步增大。Pre-Norm 天然不存在这个问题。

> 大白话：Post-Norm 就像每传一层快递都要过一次安检（Norm），安检可能扣东西（梯度衰减），传了 8 层可能什么都不剩了。Pre-Norm 的主干道是"直达快递"（残差直连），Norm 只在旁边的支路上，不影响主干道的传输。

**9. 推理时的 KV Cache 机制节省了什么？如何工作？**

答案：KV Cache 把每个 token 已计算过的 K 和 V 向量缓存起来，生成下一个 token 时不需要重新计算所有历史 token 的 K 和 V，只需要算新 token 的 Q 去和历史 K 做注意力。它节省的是 Attention 层的 K/V 投影计算量，将每步的计算量从 O(N²) 降到 O(N)。

**10. MiniMind Small 的 Attention 参数量 = 655,360。如果用 MHA（8 个 KV 头）而不是 GQA（2 个 KV 头），这个数字会变成多少？**

答案：MHA 下 K 投影和 V 投影的输入输出维度都变成 `512 × (8×64) = 512 × 512 = 262,144`。总共：Q(262,144) + K(262,144) + V(262,144) + O(262,144) = 1,048,576。GQA 节省了 (1,048,576 - 655,360) / 1,048,576 ≈ 37.5% 的 Attention 参数。但对 KV Cache 来说，显存节省更明显——从 8 组降到 2 组，减少 75%。

---

## 八、与其他文件的关系

```
model_minimind.py ←── 所有训练脚本 import 它来实例化模型
                 ←── eval_llm.py 用它做推理
                 ←── serve_openai_api.py 用它做 API 服务

model_lora.py     ←── train_lora.py import 它来注入/保存/加载 LoRA
                 ←── LoRA 模块挂载在 Attention 和 FeedForward 的 Linear 层上

tokenizer_config.json / tokenizer.json  ←── 分词器，决定 vocab_size=6400 和特殊 token ID
```

### 从训练脚本看模型的使用

```python
# 所有训练脚本的模式：
from scripts.Model.model_minimind import MiniMindConfig, MiniMindForCausalLM

lm_config = MiniMindConfig(hidden_size=512, ...)
model = MiniMindForCausalLM(lm_config)
# 如果有 LoRA:
from scripts.Model.model_lora import apply_lora, load_lora
apply_lora(model, rank=8)
```

---
