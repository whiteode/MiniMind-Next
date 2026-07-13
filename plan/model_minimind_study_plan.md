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

**7. MoE 的辅助损失（Auxiliary Loss）公式是什么？不用会怎样？**

答案：`aux_loss = Σ(Pᵢ × fᵢ) × α`，其中 Pᵢ 是专家 i 的平均门控概率，fᵢ 是专家 i 的实际负载比例（乘 n_experts 归一化，均衡时应 ≈ 1.0）。不用辅助损失会导致"路由崩溃"：模型在训练早期发现某个专家稍微好用一点，就把几乎所有 token 都路由给它，其他专家"饿死"，MoE 退化成普通 FFN。

**8. 为什么 Pre-Norm 比 Post-Norm 训练更稳定？**

答案：Post-Norm 的梯度需要穿过归一化层才能到达前面的网络层，梯度可能衰减。Pre-Norm 中残差连接绕过归一化层直连前面的层，梯度可以"无障碍"回传。这就是为什么 Pre-Norm 不需要学习率 warm-up，而原始 Transformer 的 Post-Norm 需要。

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
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

lm_config = MiniMindConfig(hidden_size=512, ...)
model = MiniMindForCausalLM(lm_config)
# 如果有 LoRA:
from model.model_lora import apply_lora, load_lora
apply_lora(model, rank=8)
```

---
