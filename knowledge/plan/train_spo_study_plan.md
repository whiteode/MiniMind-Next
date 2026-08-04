# train_spo.py 学习计划指引

🌟 **写在前面的话：什么是 SPO？**

想象你在训练一只小狗（大模型）。

**SFT（监督微调）阶段**：你手把手教它作揖。

**偏好对齐（RLHF 等）阶段**：它自己尝试作揖，你根据它的表现给零食（打分）。

过去，我们需要请一个专门的"裁判"（Critic 模型）来打分，或者让小狗一次性做 8 组动作来对比哪组好（GRPO）。而 **SPO（自博弈优化）** 提供了一种极其聪明且省钱的方法：它不需要请裁判，也不需要小狗做多组动作，只需要记住小狗最近表现的"平均及格线"，只要这次表现超过及格线，就给奖励。

它的名字"自博弈"就来源于此——模型**自己和自己过去的表现比**，而不是和别人的表现比。

---

## 一、文件定位

`train_spo.py` 是 MiniMind 项目的**自博弈优化**（Self-Play Optimization / SPO）训练脚本，属于大模型训练的最后一步：**偏好对齐阶段**。

它与大家常听说的 DPO、PPO、GRPO 属于同一家族，是目前非常新颖的第四个变体。

```
                                ┌─ DPO（最简单：直接拿着人类标好的好坏数据学）
                                ├─ GRPO（无裁判：自己生成8个回答互相比）
       训练管线（对齐阶段） ────┼─ PPO（传统老大哥：需要一个专门的裁判模型打分）
                                └─ SPO（极简极速：自己跟自己过去的平均水平比） ← 你在这里
```

### 为什么有了前面的老大哥，还要学 SPO？

这四种方法都在解决同一个问题：**如何评价模型当前的回答好不好？**

在强化学习中，我们通常用"优势 (Advantage, 简称 A)"来表示回答的好坏。

```
优势 A = 当前得分 R - 预期及格线 (Baseline)
```

| 方法 | 预期及格线 (Baseline) 怎么来？ | 需要几个模型？ | 每次提问生成几个回答？ | 优点 |
|------|-------------------------------|:-------------:|:--------------------:|------|
| DPO | 无及格线（直接对比现成的答案） | 2 | 0 | 最简单，不需要模型自己生成 |
| GRPO | 组内平均分（这次生成的 8 个回答的平均分） | 3 | 8 | 不需要裁判模型，对比很公平 |
| PPO | 裁判预测分（专门用一个模型预测及格线） | 5 | 1 | 裁判给分非常细致 |
| **SPO** | **历史滚动及格线（滑动窗口自适应）** | **3** | **1** | **极其省显存，且每次只需生成 1 个回答** |

**SPO 的核心创新**：它扔掉了复杂的裁判模型，也扔掉了每次生成 8 个回答的算力负担。它用一段代码（`AutoAdaptiveValueTracker`）在后台默默记录模型最近得分的"平均及格线"。只要这次得分 R 高于历史及格线，就是正向优化。

> 🐶 **大白话**：SPO 就像你教小狗握手——刚开始它乱做，你看到差不多就像样就给吃的（baseline 低）；后来它越做越好，你的标准也提高了（baseline 上升）。你不需要找另一个裁判来评判，也不需要让它做 8 次选最好的，只要它比**它自己以前的平均水平**好就行。

### 四种优势计算方式对比（一图胜千言）

```
DPO:   无优势，直接从对比对学习
GRPO:  A = (R - μ) / σ            ← 组内统计（和同组 8 个兄弟比）
PPO:   A = R - V(s)               ← Critic 估计（裁判打分）
SPO:   A = R - baseline            ← 滑动窗口基线（和自己过去的平均水平比）
```

---

## 二、核心概念

### 2.1 三模型架构（极简的舞台）

SPO 只需要 **3 个模型**（`train_spo.py` L298-313），让我们看看它们分别扮演什么角色：

```python
# ① Policy 模型（可训练）— 正在受训的主角
model, tokenizer = init_model(lm_config, base_weight, device=args.device)

# ② Reference 模型（冻结不更新）— 参照物（不忘初心）
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)

# ③ Reward 模型（冻结不更新）— 打分器（客观规律）
reward_model = AutoModel.from_pretrained(args.reward_model_path, ...)
reward_model = reward_model.to(args.device).eval().requires_grad_(False)
```

| 模型 | 可训练 | 通俗作用 |
|------|:-----:|---------|
| Policy | ✅ | **学生**：负责生成回答，并根据得分不断调整自己的参数 |
| Ref | ❌ | **好学生对照组**：防止 Policy 模型为了拿高分走火入魔，变成只会说漂亮话的废话生成器（计算 KL 散度） |
| Reward | ❌ | **自动阅卷机**：给 Policy 最终生成的回答打分（得出 R） |

和 GRPO 一样——**没有 Critic，没有 Old Actor**（省下了两份显存！）。

但与 GRPO 不同的是，SPO **每个 prompt 只生成 1 个回复**，这使得它的训练速度极快。

> 🐶 **大白话**：Policy 是你正在训练的小狗，Reward 是你手里的零食称（打分器），Reference 是它小时候的录像（防止它为了零食学坏）。

---

### 2.2 AutoAdaptiveValueTracker：动态的及格线（L27-66）

这是 SPO 最核心、最精妙的组件，它就是用来计算那个"预期及格线 (Baseline)"的。

#### 2.2.1 Beta 分布基线

如果给你连续抛硬币，你想预测下一次是正面的概率。最好的方法是记下"成功（正面）的次数 α"和"失败（反面）的次数 β"。

`AutoAdaptiveValueTracker` 在内部维护的就是这样一个 **Beta 分布** `(α, β)`：

```python
N_init = 1.0 / (1.0 - clip_lower)  # 相当于假设初始有 2 次抛硬币记录
self.alpha = 0.5 * N_init  # = 1.0 (成功 1 次)
self.beta  = 0.5 * N_init  # = 1.0 (失败 1 次)
```

初始时 `α=β=1`，意味着一开始我们什么都不知道，所以及格线 (baseline) 就是一半一半，设定为 0.5。

这里的 `clip_lower` 是什么？它是一个关键的超参数，**身兼二职**，但指向同一个设计思想——**控制基线记忆的"惯性"大小**：

1. **初始化先验强度**：`N_init = 1/(1 - clip_lower)` 的直觉来自"有效记忆步长"——如果模型最多能记住 N 步的历史，那初始时就应该假设看到了 N 个样本。`clip_lower=0.5` 意味着最多记住 2 步 -> 初始的等效样本量就是 2 -> 拆成 1 次成功 1 次失败，完全不偏不倚。

2. **rho（衰减因子）的下界裁剪**（L51）：后文会看到，`compute_rho` 算出的衰减率 `rho` 会被 `clip_lower` 卡住下限 0.5。这意味着**单次考试（一个 batch）最多只能影响一半的及格线**，防止某次偶然的高分或低分把基线带跑偏。

> 🐶 **大白话**：`clip_lower=0.5` 就像在说——"及格线的记忆至少保留两步以上，别因为一次表现好就觉得小狗永远会了，也别因为一次失误就觉得它啥也不会了。"

**计算当前的及格线**（L40-42）：

```python
def get_baselines(self, batch_size):
    baseline = self.alpha / (self.alpha + self.beta)   # 成功次数 / 总次数
    return torch.full((batch_size,), baseline)
```

取 Beta 分布的**均值** `α/(α+β)` 作为当前基线——相当于说"根据历史的成功率和失败率，平均来看大概能得多少分"。

**为什么要用 Beta 分布而不是直接做 EMA（指数移动平均）？**

Reward Model 的评分在 `[-3, 3]` 范围内，标准化到 `[0, 1]` 后可视为一个**伯努利试验的累计成功率**：
- 高分（接近 1）≈ 成功
- 低分（接近 0）≈ 失败

Beta 分布是伯努利试验的**共轭先验**——只需要维护 α（累计成功次数）、β（累计失败次数）两个参数，就能完整描述当前基线的不确定性。

> 🐶 **大白话**：为什么不用 EMA 而用 Beta？EMA 只能记住"平均分是多少"，Beta 能记住"成功了多少次、失败了多少次"。虽然当前代码只用了均值，但保留 α 和 β 两个参数给未来留了后路——比如你还可以看方差，知道"这个基线到底有多可靠"。

---

#### 2.2.2 滑动更新（L53-66）

```python
def update(self, rewards, cur_logprobs=None, response_masks=None):
    # 计算当前 batch 的平均 logprob（用于 rho 计算）
    mean_logprob = ((cur_logprobs * response_masks).sum() / response_masks.sum()).item()
    rho = self.compute_rho(mean_logprob)  # 自适应衰减率
    self.old_mean_logprob = mean_logprob

    scale = 3.0
    normalized_rewards = (rewards + scale) / (2 * scale)     # [-3, 3] → [0, 1]
    avg_normalized_reward = normalized_rewards.mean().item()  # 当前 batch 平均分数

    self.alpha = rho * self.alpha + avg_normalized_reward     # 指数滑动更新
    self.beta  = rho * self.beta  + (1 - avg_normalized_reward)

    return rho
```

> 💡 **代码解析：`mean_logprob` 是什么？**
>
> `mean_logprob = ((cur_logprobs * response_masks).sum() / response_masks.sum()).item()`
>
> | 步骤 | 表达式 | 含义 |
> |------|--------|------|
> | 1 | `cur_logprobs * response_masks` | 逐元素相乘。`response_masks` 是 0/1 掩码，prompt 位置为 0，response 位置为 1，所以乘完之后 prompt 的 logprob 被清零，只保留 response 部分 |
> | 2 | `.sum()` | 把所有 response token 的 logprob 加起来 |
> | 3 | `/ response_masks.sum()` | `response_masks.sum()` 就是 response token 的个数，除法得到**平均值** |
> | 4 | `.item()` | 从 tensor 提取为 Python 标量 |
>
> **一句话总结**：算出当前 batch 中，模型对 response 每个 token 给出的 log probability 的均值。
>
> **用途**：这个值不是用来算 loss 的，而是传给 `compute_rho()` 来衡量**模型变化有多大**（L187-188）——和上一次的 `old_mean_logprob` 做差得到 `kl`，kl 越大说明模型变化越剧烈，rho 就越小，意味着滑动平均的及格线应该更快地跟上新数据（旧数据权重降低）。

> 💡 **`cur_logprobs` 是什么？**
>
> `cur_logprobs` 就是 `per_token_logps.detach()`（`train_spo.py:192`），即**当前 policy 模型对每个 response token 给出的 log probability**。
>
> 它由 `get_per_token_logps()` 计算（`train_spo.py:149-156`）：
> 1. 模型前向得到 logits，取 `log_softmax` 得到每个 token 在词表上的 log 概率分布
> 2. 用 `torch.gather` 从分布中取出**实际生成的那个 token**的 log prob
> 3. 返回 shape 为 `[B, R]` 的 tensor（B=batch size, R=response 长度）
>
> 传入 `update()` 时加了 `.detach()`，是因为这里只需要数值，不需要梯度回传——`mean_logprob` 仅用于计算 rho（衰减率），不参与 loss 反向传播。

> 💡 **`normalized_rewards` 在干嘛？**
>
> `normalized_rewards = (rewards + scale) / (2 * scale)` 是 **min-max 归一化**，把 reward 从 `[-3, 3]` 映射到 `[0, 1]`：
>
> | rewards | 计算 | 结果 |
> |---------|------|------|
> | -3（最差） | (-3 + 3) / 6 | **0** |
> | 0（一般） | (0 + 3) / 6 | **0.5** |
> | 3（最好） | (3 + 3) / 6 | **1** |
>
> 这是标准 min-max 公式 `(x - min) / (max - min)` 的变体，min=-scale, max=scale，所以分母 `max-min = 2*scale`，分子 `x-min = x+scale`。
>
> 归一化后 `avg_normalized_reward` 就在 `[0, 1]` 之间，可以直接当概率用——贡献给 α（成功的权重）和 `1 - R̄` 贡献给 β（失败的权重），两者加起来正好是 1。

> 💡 **`rewards` 是什么？为什么要加 scale？**
>
> `rewards` 是 `calculate_rewards()` 的返回值（`train_spo.py:69-121`），由两部分加成：
> 1. **格式奖励**（reasoning 模式）：输出符合 `<think>...</think><answer>...</answer>` 格式就加分，每个正确标签 +0.25，格式完全匹配 +0.5
> 2. **Reward Model 打分**：用单独的奖励模型对 (prompt, response) 评分，score 被 clamp 到 `[-3, 3]`
>
> 所以 `rewards` 的取值范围是 `[-3, 3]`。
>
> **为什么要加 scale？** min-max 归一化的目标是把 `[-3, 3]` 映射到 `[0, 1]`：
> ```
> 标准公式：(x - min) / (max - min) = (x - (-3)) / (3 - (-3)) = (x + 3) / 6
> ```
> `+scale` 就是 `-min`，作用是**把最小值从 -3 抬到 0**。如果不加，-3 会映射到 -0.5，不在 `[0, 1]` 内，后面 α、β 就没法当非负权重用了。

每次 `update()` 时，用当前 batch 的平均 reward 做一次**指数滑动更新**：

```
α_new = ρ · α_old + R̄            （β 同理，用 1 - R̄）
```

其中 **ρ（rho）** 是衰减率——控制历史信息保留多久。新的 reward 贡献 R̄ 到 α，1 - R̄ 到 β。

所以：
- 如果 R̄ 高（这次考得好）→ α 增加得比 β 多 → 及格线上升
- 如果 R̄ 低（这次考砸了）→ β 增加得比 α 多 → 及格线下降

> 💡 **α 会不会越来越大？**
>
> 不会发散，但会**超过 1**。这里和标准 EMA 不同：
> - 标准 EMA：`α_new = (1-ρ)·α + ρ·R̄`（权重之和 = 1）
> - SPO 公式：`α_new = ρ·α + R̄`（权重之和 = ρ + 1 > 1）
>
> 收敛值推导：令 `α* = ρ·α* + R̄`，解得 `α* = R̄ / (1-ρ)`。
> 当 `ρ=0.9, R̄=0.5` 时，`α* = 5`。所以 alpha 会收敛到一个有限值，不会无限增长。
>
> **为什么不发散？** 因为 `ρ < 1`，旧值系数严格小于 1，递推是压缩映射，必收敛。
>
> **超过 1 有问题吗？** 没有——α 和 β 是 Beta 分布的**形状参数**（shape parameters），只需 > 0，无上限。α 越大，Beta 分布越集中在高 reward 方向，及格线越高。

> 🐶 **大白话**：每次训练后更新"小账本"。如果小狗这次做得好（R̄ 高），就在"成功"栏里多记一笔，下次的及格线就会抬高一点；如果做得不好，就在"失败"栏里多记一笔，下次的及格线就降低一点。`rho` 控制着"以前的老账本保留多少"。

---

#### 2.2.3 自适应 rho 计算（L44-51）

```python
def compute_rho(self, cur_mean_logprob):
    if self.rho_mode == 'constant':
        return self.rho_const            # 固定值 0.9
    if self.old_mean_logprob is None:
        return self.rho_const            # 首次返回固定值

    kl = abs(self.old_mean_logprob - cur_mean_logprob)      # logprob 的变化量
    rho = 2 ** (-kl / self.D_half)                           # 半衰期衰减
    return max(min(rho, self.clip_upper), self.clip_lower)   # 裁剪到 [0.5, 0.96]
```

**ρ 的自适应机制**——模型变化越快，历史基线越不可靠，就越应该相信新数据：

- **logprob 变化小**（模型收敛，输出趋于稳定）：kl 小 → rho 接近 1.0 → 历史保留多，基线稳如泰山
- **logprob 变化大**（模型还在活跃探索，输出还在剧烈变）：kl 大 → rho 变小 → 旧数据快速丢弃，基线快速跟上

公式解析：`rho = 2^(-kl / D_half)`，其中 `D_half=0.06` 是半衰期参数。

当 kl 多大时，历史信息的权重会衰减一半？

```
kl = 0.00 → rho = 2^(0)      = 1.00   历史几乎全保留
kl = 0.06 → rho = 2^(-1)     = 0.50   半衰期：历史权重减半
kl = 0.12 → rho = 2^(-2)     = 0.25   再衰减一半
kl = 0.60 → rho = 2^(-10)    ≈ 0.001  几乎完全丢弃历史
```

所以 `D_half=0.06` 的含义是：**当 logprob 变化 0.06 nats 时，历史信息的记忆只剩下一半**。

> 🐶 **大白话**：`compute_rho` 做了一件很聪明的事——它通过监控模型输出的变化幅度（logprob 变化），来判断模型现在的状态。如果模型的输出和上一步差不多（收敛了），那就慢慢更新基线。如果模型还在剧烈变化（刚开始训练），那就快速丢弃旧基线，用新数据重新建立。就像教小狗新把戏时，如果它学得很快，你就该及时更新标准。

> 💡 **代码逐行解读：**
>
> | 行 | 代码 | 含义 |
> |----|------|------|
> | 1-2 | `if rho_mode == 'constant': return rho_const` | 如果模式是固定值，直接返回 0.9，不做自适应 |
> | 3-4 | `if old_mean_logprob is None: return rho_const` | 第一次调用没有历史记录，也返回固定值 |
> | 5 | `kl = abs(old - cur)` | 计算两次 batch 之间 mean_logprob 的绝对差值，衡量模型变化幅度 |
> | 6 | `rho = 2^(-kl / D_half)` | 半衰期公式：kl=0.06 时 rho=0.5，kl=0 时 rho=1 |
> | 7 | `clip(rho, 0.5, 0.96)` | 裁剪防止极端：rho 最小 0.5（至少保留一半历史），最大 0.96（至少丢掉 4% 旧数据） |

---

#### 2.2.4 与 GRPO / PPO 的对比

| | GRPO | PPO | SPO |
|---|---|---|---|
| 基线 | `(R-μ)/σ` 组内统计 | `V(s)` 独立 Critic | `α/(α+β)` 滑动 Beta |
| 历史保留 | 无（每步独立） | Critic 权重自动保留历史 | ρ 控制的指数衰减 |
| 跨 batch 一致性 | ❌ 每步独立 | ✅ Critic 有记忆 | ✅ 通过 α,β 有记忆 |
| 额外模型 | 无 | 1 个 Critic | 无 |
| 额外生成 | 7 个额外回复 | 0 | 0 |

---

### 2.3 get_per_token_logps（L149-156）

```python
def get_per_token_logps(mdl, input_ids, n_keep):
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)
```

**逐行解读**：

| 行 | 作用 |
|----|------|
| `logits_to_keep=n_keep + 1` | 只需要最后 `n_keep+1` 个 token 的 logits（**省计算量**——前面的 prompt 部分已经算好了，不用重算） |
| `logits[:, :-1, :]` | 去掉最后一个位置的 logits，保持和 labels 长度对齐 |
| 逐行 for 循环 | 对 batch 内每个样本分别做 `gather`，兼容 flash_attn 的限制 |
| `log_softmax + gather` | 取出每个位置 target token 的 log_prob |

> 💡 **`logits[:, :-1, :]` 为什么要切片？**
>
> 这行做了三件事，从右往左读：
>
> 1. **`mdl(input_ids, logits_to_keep=n_keep + 1)`** — `logits_to_keep` 是优化参数，模型只计算**最后 `n_keep + 1` 个位置**的 logits，省显存省计算
> 2. **`.logits`** — 取出 logits tensor，shape `[B, seq_len, vocab_size]`
> 3. **`[:, :-1, :]`** — 去掉最后一个位置。因为语言模型中位置 `t` 的 logits 预测 `t+1` 位置的 token。如果保留了 `n_keep + 1` 个位置，去掉最后一个就剩 `n_keep` 个，正好对应预测 `input_ids[:, -n_keep:]`（response 的每个 token）
>
> 示例（response = `["我", "爱", "你"]`，n_keep=3）：
> ```
> 位置:       ... |  我  |  爱  |  你
> logits_to_keep=4: logit[我] logit[爱] logit[你] logit[?]
> [:-1] 去掉最后:    logit[我] logit[爱] logit[你]
>                      ↓        ↓        ↓
> 预测目标:          "我"      "爱"      "你"
> ```

> 💡 **`ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row` 是什么？**
>
> 这是一个**防御性拷贝**，防止 `ids_row`（token ID）意外携带计算图。
>
> - `detach()` — 从计算图中摘出来，不再跟踪梯度
> - `.clone()` — 复制一份新 tensor，避免共享内存
>
> token ID（如 `[1024, 345, 789]`）本质上是整数索引，不需要也不能有梯度。如果它不小心继承了计算图，后续用它做 `gather` 索引时可能出问题。`detach().clone()` 确保它是一份干净的、无梯度的整数 tensor。
>
> 注意：推理模式下 `torch.no_grad()` 已全局禁用梯度，detach 是冗余的；训练模式下才真正需要。但在本项目中 `get_per_token_logps` 只在训练时调用，`ids_row.is_inference()` 始终返回 `False`，所以这个 `if` 分支**永远不会触发**——这行实际上是死代码，可能是作者为通用性加的防御。

> 💡 **`torch.gather(..., 1, ids_row.unsqueeze(1)).squeeze(1)` 在干嘛？**
>
> 从模型的 log 概率分布中，取出每个位置**实际生成的那个 token** 的 log prob。逐步拆解：
>
> | 步骤 | 表达式 | shape | 含义 |
> |------|--------|-------|------|
> | 1 | `logits_row.log_softmax(dim=-1)` | `[R, vocab]` | 对 vocab 维度做 log_softmax，得到每个位置上所有词的 log 概率分布 |
> | 2 | `ids_row.unsqueeze(1)` | `[R] → [R, 1]` | 扩展 token ID 的维度，适配 gather 索引格式 |
> | 3 | `torch.gather(..., 1, ...)` | `[R, 1]` | 沿 vocab 维度，按索引取出每个位置实际 token 的 log prob |
> | 4 | `.squeeze(1)` | `[R]` | 去掉多余维度 |
>
> 示例（vocab = `["我", "爱", "你", "好"]`，response = `["我", "你"]`）：
> ```
> log_softmax = [[-0.5, -1.2, -2.0, -3.0],   # 位置 0
>                [-1.8, -2.5, -0.3, -1.0]]   # 位置 1
> ids_row = [0, 2]                            # "我"=0, "你"=2
>
> gather 按索引取: [-0.5, -0.3]  →  "我"的 log prob, "你"的 log prob
> ```

> 💡 **`torch.stack(per_token_logps)` 是什么？**
>
> for 循环中每次 append 一个 shape 为 `[R]` 的 tensor（一个样本的 log prob），循环后得到长度为 B 的列表。`torch.stack` 沿**新维度**堆叠，把 `[R], [R], ...` (共 B 个) 变成 `[B, R]`：
>
> ```
> 堆叠前: [tensor([-0.5, -1.2, -0.8]),    # 样本 0, shape [R]
>          tensor([-1.1, -0.3, -0.6])]    # 样本 1, shape [R]
>
> 堆叠后: [[-0.5, -1.2, -0.8],            # shape [B, R]
>          [-1.1, -0.3, -0.6]]
> ```
>
> 注意和 `torch.cat` 的区别：`cat` 在已有维度拼接，`stack` 创建新维度。

> 🐶 **大白话**：这个函数的作用是"算出模型在生成的每个词上有多自信"。它会遍历生成的每个 token，从模型的输出概率中提取出"你选这个词的概率有多大"。注意它按**逐个样本**循环处理，而不是整个 batch 一起算——这是一个底层兼容性优化，不影响原理理解。

**与 PPO/GRPO 的差异**：

| | PPO | GRPO | SPO |
|---|---|---|---|
| logp 获取方式 | 一次 forward + gather | 一次 forward + gather | **逐行循环** gather |
| 返回粒度 | 序列级（sum over tokens） | token 级 `[B, R]` | token 级 `[B, R]` |
| 为什么循环 | — | — | 兼容 flash_attn 的 `logits_to_keep` 限制 |

> 💡 **为什么用逐行循环而不是 batch gather？**
>
> 正常 batch gather 写法：
> ```python
> log_softmax = logits.log_softmax(dim=-1)                          # [B, R, vocab]
> per_token_logps = torch.gather(log_softmax, -1, ids.unsqueeze(-1)).squeeze(-1)  # [B, R]
> ```
>
> 但 `logits_to_keep` 结合 flash_attn 时，不同样本的"最后 n_keep 个位置"在 batch 内的绝对坐标可能不一致。如果 batch gather，会取错位置。
>
> 而逐行循环**在每个样本自己的行内**做 gather，无论 batch 对齐状态如何，都不会取错。
>
> 性能方面不用担心——R（response 长度）通常 100~1000，B 也不大，循环开销远小于模型 forward 本身，是**逻辑正确优先、性能无碍的防护性写法**。

---

### 2.4 优势计算（L169-177）

```python
baselines = value_tracker.get_baselines(len(prompts)).to(args.device)  # [B] 从 Beta 分布获取基线

scale = 3.0
unnormalized_baselines = baselines * (2 * scale) - scale  # 反标准化: [0,1] → [-3,3]
advantages = rewards - unnormalized_baselines              # 优势 = 奖励 - 基线

advantages = advantages.clamp(-5.0, 5.0)                   # 裁剪防止极端值
```

**三步处理**：

1. **获取基线**：`value_tracker.get_baselines()` 从 Beta 分布均值 `α/(α+β)` 得到 `[0,1]` 的标量
2. **反标准化**：`baseline * 6 - 3` 将 `[0,1]` 映射回 `[-3, 3]`（和 Reward Model 的评分范围一致）
3. **优势裁剪**：`clamp(-5, 5)` 防止个别极端的评分差异导致梯度爆炸

> 💡 **`get_baselines` 在做什么？**
>
> ```python
> def get_baselines(self, batch_size):
>     baseline = self.alpha / (self.alpha + self.beta)
>     return torch.full((batch_size,), baseline, dtype=torch.float32)
> ```
>
> `alpha / (alpha + beta)` 是 **Beta(α, β) 分布的均值**——数学期望公式。它代表了基于历史数据积累的"及格线"。返回一个全为这个值的 `[B]` tensor，batch 内每个样本共享同一个基线。
>
> **为什么要反标准化？** baseline 在 `[0, 1]`，但 rewards 在 `[-3, 3]`，单位不一致。反标准化是归一化 `y = (x + scale) / (2 * scale)` 的**逆运算**：`x = y * (2 * scale) - scale`。
>
> 最后 `advantages = rewards - unnormalized_baselines`：正值 = 超预期，负值 = 低于预期。

> 🐶 **大白话**：计算"优势"就是回答一个问题——**"这次比平时好多少？"** 先用 Beta 分布算出"平时水平（baseline）"，把它从 `[0,1]` 还原回 `[-3,3]` 的分数范围，然后 `奖励 R - 平时水平` 就是"这次超常发挥了多少"。最后 clamp(-5, 5) 是个保险——防止某次表现过于离谱把模型带偏。

**四方法优势计算对比**：

```
DPO:  无优势（直接用偏好对）
GRPO: A = (R - μ) / σ                ← 组内归一化，均值 μ，标准差 σ
PPO:  A = R - V(s)                   ← Critic 估计的价值
SPO:  A = R - (α/(α+β) · 6 - 3)     ← Beta 分布均值反标准化
```

---

### 2.5 完整 Loss（L184-188）

```python
# ① 逐 token KL 散度
kl_div = ref_per_token_logps - per_token_logps                    # [B, R]
per_token_kl = torch.exp(kl_div) - kl_div - 1                     # [B, R]

# ② 逐 token 损失：策略梯度 + KL 惩罚
per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl  # [B, R]

# ③ 序列内平均 + batch 平均
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

# ④ 总损失
loss = (policy_loss + aux_loss) / args.accumulation_steps
```

> 💡 **`per_token_logps` 和 `ref_per_token_logps` 是什么？**
>
> 两个都是通过 `get_per_token_logps()` 计算的 log probability，区别在于来自哪个模型：
>
> ```python
> # train_spo.py L159, L164
> per_token_logps     = get_per_token_logps(model, ...)        # policy 模型（可训练）
> ref_per_token_logps = get_per_token_logps(ref_model, ...)    # reference 模型（冻结不更新）
> ```
>
> - `per_token_logps` — policy 模型对每个 response token 的 log prob，shape `[B, R]`
> - `ref_per_token_logps` — 冻结的 reference 模型对**同样输入**的 log prob，shape `[B, R]`
>
> 两者相减得到 `kl_div = ref - policy`，衡量 policy 相对于 reference 的偏移。如果 policy 给 reference 也高概率的 token → kl_div ≈ 0 → 惩罚小；如果 policy 跑偏了 → kl_div 大 → 惩罚大。
>
> **本质作用**：防止 policy 在优化 reward 时跑太远——你要学着拿高分，但别忘了初心（reference 代表的原始行为）。

Loss 由两部分组成：

- **策略梯度 `-logp · A`**：如果优势为正（这次比平时好），就增大这些 token 的概率；为负则减小
- **KL 惩罚 `β · per_token_kl`**：不让模型离 Reference 模型太远，防止"为了高分说违心话"

> 💡 **逐项拆解**
>
> `per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl`
>
> **① 策略梯度项 `-per_token_logps * A`**
>
> | 条件 | Loss 方向 | 模型行为 |
> |------|-----------|---------|
> | A > 0（超预期） | 想让 Loss 小，`-logp` 必须变小 → logp 需**变大** | **强化**这类 token |
> | A < 0（低于预期） | 想让 Loss 小，`-logp` 必须变大 → logp 需**变小** | **抑制**这类 token |
>
> **② KL 惩罚项 `β · per_token_kl`**
>
> `per_token_kl = exp(kldiv) - kldiv - 1`，其中 `kldiv = ref - policy`。这个函数两边不对称：
>
> ```
> policy 概率  |  kldiv  |  per_token_kl  |  惩罚含义
> 远高于 ref   |  负值   |  快速增长      |  🚫 别太自信
> 略高于 ref   |  略负   |  几乎为 0      |  ✅ 没事
> 远低于 ref   |  正值   |  线性增长      |  ⚠️ 可适度降低
> ```
>
> 不对称是**刻意设计**的——过度自信比适度降权更危险，所以左半边惩罚更重。

> 🐶 **大白话**：Loss 就是在做两件事的平衡——**(1) 鼓励模型重复这次的好表现**（策略梯度），和 **(2) 别让它放飞自我**（KL 惩罚）。就像一个老师在鼓励学生"这次做得对，下次还这样做"，但同时说"但别为了拿高分去作弊或说违心话"。

**对比 PPO / GRPO 的 Loss**：

| 成分 | PPO | GRPO | SPO |
|------|:---:|:----:|:---:|
| 策略梯度 | ✅ 裁剪代理 `min(surr1, surr2)` | ✅ 未裁剪 `exp(x)·A` | ✅ 未裁剪 `-logp · A` |
| 价值 Loss | ✅ `MSE(V, R)` | ❌ | ❌ |
| KL 惩罚 | ✅ 序列级 `(actor-ref).mean()` | ✅ 逐 token `exp(kl)-kl-1` | ✅ 逐 token `exp(kl)-kl-1` |
| 裁剪 | ✅ clip_epsilon | ❌ | ❌ |
| 辅助 Loss | ✅ aux_loss | ✅ aux_loss | ✅ aux_loss |

SPO 的 Loss 结构和 GRPO **几乎一模一样**，唯一区别是**优势的来源不同**：
- GRPO 的 A 来自组内归一化
- SPO 的 A 来自滑动基线

#### 为什么 `-per_token_logps * advantages` 等价于 `ratio * advantages`？

对比 GRPO 的策略梯度：`per_token_loss = -per_token_logps * advantages`

如果你回忆 GRPO 的公式：`loss = -exp(per_token_logps - old_per_token_logps) * advantages`

GRPO 中 `old_per_token_logps = per_token_logps.detach()`，所以 `exp(x - x.detach()) = 1`，实际上就是 `-per_token_logps * advantages`。

SPO 更彻底——它**干脆没有 old_logp 的概念**，直接用当前的 `per_token_logps` 做策略梯度。因为 SPO 每步都用新数据，不需要重要性采样修正。

> 🐶 **大白话**：GRPO 有"旧的模型"的概念（保存了一份旧版本的参数），SPO 连这都省了。既然每次都是重新生成新数据，为什么要保留旧版本呢？这就是 SPO 一贯的设计哲学——**能省就省**。

---

### 2.6 completion_mask 构造（L179-182）

```python
is_eos = completion_ids == tokenizer.eos_token_id                # [B, R] 找到 eos 位置
eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), ...)    # 默认 R（序列末）
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]  # 找到 eos 位置
completion_mask = (torch.arange(...).expand(...) <= eos_idx.unsqueeze(1)).int()  # 从开头到 eos
```

> 💡 **逐层拆解：**
>
> 假设 batch=2, R=5, eos_idx=[2, 4]：
>
> | 步骤 | 代码 | 结果 |
> |------|------|------|
> | 1 | `torch.arange(R)` | `[0, 1, 2, 3, 4]` — 位置序号 |
> | 2 | `.expand(B, -1)` | `[[0,1,2,3,4],[0,1,2,3,4]]` — 复制成 batch 份 |
> | 3 | `<= eos_idx.unsqueeze(1)` | 广播比较，`[[0,1,2,3,4]<=2, [0,1,2,3,4]<=4]` |
> | 4 | `.int()` | `[[1,1,1,0,0],[1,1,1,1,1]]` — 0/1 掩码 |
>
> 样本 0（eos 在位置 2）：前 3 个有效，后 2 个 padding 遮掉
> 样本 1（eos 在位置 4）：全有效

与 GRPO 完全相同——构造一个 mask，标记从序列开头到 eos（含）之间的所有 token 为有效。eos 之后的 token 被排除在 loss 计算之外。

> 🐶 **大白话**：因为模型生成的回复长度不固定，长的短的需要 padding 到一样长才能一起算。这个 mask 就是告诉训练代码"只看生成的内容，padding 的空白部分不参与计算"。

---

### 2.7 调度器

```python
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # 默认 1e-7
scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
```

**比较四种方法的 LR**——为什么 SPO 用 1e-7？

| 方法 | 默认 LR | 说明 |
|------|:------:|------|
| DPO | 4e-8 | 静态数据，最保守，步子最小 |
| GRPO | 8e-8 | 组内归一化提供稳定性，步子可以大一点 |
| **SPO** | **1e-7** | 滑动基线提供一定稳定性，但无裁剪，介于中间 |
| PPO | 1e-6 | 裁剪机制兜底，最大胆 |

SPO 的 LR 比 GRPO 略大（1e-7 vs 8e-8），但没有 PPO 那么大（1e-6），因为：
- 滑动基线提供了一些稳定性（比 GRPO 的纯组内统计更平滑）
- 但没有裁剪机制（不如 PPO 安全），所以不能太激进

> 🐶 **大白话**：学习率就像训练时的步长——太小人学得慢，太大容易摔跤。SPO 的步长在 GRPO 和 PPO 之间，因为它的"安全气囊"（滑动基线）比 GRPO 好（GRPO 每步独立没有记忆），但比 PPO 的裁剪机制差（裁剪是直接限制了最大步长）。所以取个中间值。

---

## 三、关键公式一览

### AutoAdaptiveValueTracker
```
α₀ = β₀ = 1.0
baseline_t = α_t / (α_t + β_t)             当前基线（Beta 分布均值）
ρ_t = 2^(-|logprob_t - logprob_{t-1}| / D_half)   自适应衰减率
α_{t+1} = ρ_t · α_t + R̄_t                   滑动更新
β_{t+1} = ρ_t · β_t + (1 - R̄_t)             滑动更新
```

### 优势
```
A = R - baseline_unscaled
baseline_unscaled = baseline * 6 - 3
```

### 策略 Loss
```
L = -log π(a|s) · A + β · KL(π || π_ref)
KL = exp(log π_ref - log π) - (log π_ref - log π) - 1
```

---

## 四、学习目标检查清单

- [ ] 理解 SPO 在完整训练管线中的位置
- [ ] 理解 SPO 与 DPO / GRPO / PPO 的本质区别（基线来源不同）
- [ ] 理解 AutoAdaptiveValueTracker 的 Beta 分布设计思想
- [ ] 理解 get_baselines() 从 α, β 计算 baseline 的方式
- [ ] 理解 update() 中 α, β 的滑动更新公式
- [ ] 理解 compute_rho() 的自适应衰减机制和 D_half 的含义
- [ ] 理解 SPO 的 3 模型架构（与 GRPO 相同，与 PPO 不同）
- [ ] 理解 get_per_token_logps 为什么用逐行循环而不是 batch gather
- [ ] 理解 baseline 反标准化 `baseline * 6 - 3` 的意义
- [ ] 理解 `advantages.clamp(-5, 5)` 的作用
- [ ] 理解 SPO Loss 与 GRPO Loss 的异同
- [ ] 理解 completion_mask 的构造方式
- [ ] 比较 SPO / GRPO / PPO / DPO 四种对齐方式的优缺点

---

## 五、文件逐段精读计划

### 第 1 层：AutoAdaptiveValueTracker（L27-66）

**Q：为什么需要这个类？它替代了什么？**
A：替代了 PPO 的 Critic 模型和 GRPO 的组内多采样。它用两个标量 α, β 维护一个运行基线，不需要额外模型和额外生成。

> 🐶 这就相当于不需要请裁判打分（PPO），也不需要让小狗做 8 次再取平均（GRPO），只需要记一本"最近表现账本"就行。

**Q：初始 α 和 β 为什么设为 1.0？**
A：初始基线为 `α/(α+β) = 0.5`（反标准化后为 0），表示"初始时不知道 Reward Model 的打分习惯"。随着训练推进，α 和 β 会自适应调整。

**Q：`rho` 的作用是什么？为什么用半衰期公式？**
A：rho 控制历史信息的衰减速度。半衰期公式 `2^(-kl/D_half)` 使得 logprob 变化越大衰减越快——当模型变化快时，旧基线的参考价值降低，需要更快地丢弃。

> 🐶 就像你教小狗新把戏——如果它突然学会了一个新技能，之前的及格标准就应该快速更新，而不是还拿旧标准衡量。

**Q：`rho_mode='constant'` 和 `rho_mode='kl'` 的区别？**
A：constant 模式固定 rho=0.9，始终以 0.9 的速率衰减。kl 模式根据 logprob 变化自适应调整 rho。

**Q：为什么用 Beta 分布而不是简单的 EMA（指数移动平均）？**
A：Beta 分布天然适合 `[0,1]` 区间的累计成功/失败计数，能同时提供均值估计和不确定性信息。简单 EMA 只有一个标量，失去了分布信息。但当前代码中只用了均值做基线——相当于只用了 Beta 分布的"一阶矩"。

### 第 2 层：calculate_rewards（L69-128）

与 GRPO 和 PPO 的 `calculate_rewards` **完全一样**——格式奖励 + Reward Model 评分的双层设计。

**Q：和 GRPO 的 calculate_rewards 有什么不同？**
A：结构完全一致——格式奖励的 pattern、分值分配、推理模式下的加权方式（完整 40% + answer 60%）都相同。

### 第 3 层：训练循环主流程（L131-192）

**Q：`do_sample=True, temperature=0.8` 和 PPO 一样？**
A：是的，因为 SPO 也是每个 prompt 只生成 1 个回复。

**Q：`get_per_token_logps` 的 `logits_to_keep` 参数有什么作用？**
A：`logits_to_keep=n_keep + 1` 告诉模型只计算最后 `n_keep+1` 个 token 的 logits，前面的 prompt 部分直接复用 KV cache 不重新计算，大幅节省计算量。

> 🐶 就像批改作文时只看新写的那段，前面已经看过的部分就不用再看了。

**Q：为什么 `get_per_token_logps` 用 for 循环逐个处理 batch？**
A：代码注释没有明确说明，这是一个实现细节——配合 `flash_attn` + `logits_to_keep` 时，batch gather 可能触发兼容性问题。逐个处理是最稳妥的做法，虽然牺牲了一点并行效率。

**Q：为什么 advantages 要做 `clamp(-5, 5)`？**
A：SPO 没有 PPO 的裁剪机制，也没有 GRPO 的组内归一化。单个样本可能因为 Reward 模型在分布外的极端预测而产生非常大的优势值，`clamp` 是最后的防线。

> 🐶 就像给考试加分设置上限——不能因为一次超常发挥就加太多分，也不能因为一次失误就扣太多。

**Q：GRPO 也有 `per_token_kl` 的写法，和 SPO 一样吗？**
A：完全一样。`kl_div = ref_logp - logp` 是逐 token 的 log 概率差；`exp(kl_div) - kl_div - 1` 是其在 0 处泰勒展开的前两项，既保证了 KL 非负，又不会在负方向过度惩罚。

**Q：为什么 Loss 不是 `sum(dim=1)` 而是逐 token 乘 mask 再求平均？**
A：因为返回的就是 `[B, R]` 的 token 级 loss。先 `sum(dim=1)` 序列内求和，再除以 `completion_mask.sum(dim=1)` 做序列内平均，最后 `.mean()` 做 batch 平均。这和 GRPO 完全一致。

### 第 4 层：优化与调度（L194-199）

**Q：SPO 只有一个优化器？**
A：是的。SPO 只有 Policy 模型可训练（Ref 和 Reward 冻结），所以只用 `optimizer.AdamW(model.parameters(), ...)`，不像 PPO 需要两个优化器。

**Q：梯度累积的写法？**
A：每步 `loss.backward()` 都会累积梯度，每 `accumulation_steps` 步（默认 4）才执行一次 `optimizer.step()` + `scheduler.step()` + `zero_grad()`。

### 第 5 层：主函数与初始化（L244-354）

**Q：SPO 初始化了几个模型？**
A：3 个——Policy、Reference、Reward。和 GRPO 一样，比 PPO 少 2 个。

**Q：`value_tracker` 在什么时候创建？有什么参数？**
A：L315，创建的默认参数为 `rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96`。

**Q：为什么 `lm_config.max_seq_len` 被设为 `max_seq_len + max_gen_len`？**
A：L281：`max_seq_len=args.max_seq_len + args.max_gen_len`。因为 SPO 除了处理 prompt 还要处理生成的 completion，总长度需要同时容纳两者。

**Q：SPO 的从断点恢复逻辑和 GRPO/PPO 有什么不同？**
A：更简单——只恢复 `model`、`optimizer`、`scheduler`。不需要恢复 Critic（SPO 没有 Critic），也不需要恢复 Old Actor。

---

## 六、自测题

### 基础题

1. SPO 与 GRPO 最核心的区别是什么？（提示：不是模型数量，是优势来源）

2. `AutoAdaptiveValueTracker` 的 α 和 β 分别代表什么？初始值为什么是 1.0？

3. `get_baselines()` 返回的 baseline 范围是多少？为什么要做反标准化？

4. SPO 的优势和 PPO / GRPO 的优势分别是什么？

5. `compute_rho()` 的 `D_half` 参数是怎么工作的？如果 `D_half` 减小会怎样？

**参考答案**

> **第 1 题**：GRPO 用组内多采样的均值 μ 作基线（`A = (R-μ)/σ`），SPO 用跨 batch 的滑动 Beta 分布均值作基线（`A = R - baseline`）。GRPO 的基线随 batch 变化，SPO 的基线跨 batch 平滑更新。
>
> GRPO 每步需要 `num_generations=8` 个采样才能得到一个可靠的 μ；SPO 每步只需要 1 个采样，因为它通过 α, β 累积了跨 batch 的历史信息。
>
> | | GRPO | SPO |
> |---|---|---|
> | 基线来源 | 组内统计（当前 batch 内 8 个回复） | Beta 分布均值（跨 batch 历史累积） |
> | 每步生成 | 8 个回复 | 1 个回复 |
> | 跨 batch 一致性 | ❌ 每步独立 | ✅ 通过 α, β 保持 |
>
> > 🐶 **一句话**：GRPO 是"这一轮做 8 次取平均"，SPO 是"永远记得最近的平均水平"。
>
> **第 2 题**：α 是"累计有效成功次数"（奖励标准化到 [0,1] 后的累计值），β 是"累计有效失败次数"（1 - 奖励的累计值）。初始 α=β=1.0 对应 Beta(1,1) 均匀分布，基线 = 0.5，表示"没有任何先验知识时的中立基线"。
>
> **第 3 题**：`get_baselines()` 返回 `[0, 1]` 范围的标量（Beta 分布的均值）。因为 Reward Model 的评分范围是 `[-3, 3]`，所以需要反标准化：`baseline * 6 - 3` 映射到 `[-3, 3]`。
>
> **第 4 题**：
> - **SPO**：`A = R - (α/(α+β) * 6 - 3)`，来自滑动 Beta 分布均值
> - **PPO**：`A = R - V(s)`，来自 Critic 网络
> - **GRPO**：`A = (R - μ) / σ`，来自组内统计
>
> **第 5 题**：`rho = 2^(-kl/D_half)`。D_half 是半衰期参数——当 `kl = D_half` 时，rho = 0.5。D_half 越小，rho 对 kl 变化越敏感：同样 kl=0.06，D_half=0.03 → rho=0.25，D_half=0.06 → rho=0.5，D_half=0.12 → rho≈0.707。

### 进阶题

6. SPO 的 Loss 公式 `-per_token_logps * advantages + beta * per_token_kl` 和 GRPO 的 Loss 有什么异同？和 DPO 的 Loss 呢？

7. 如果把 SPO 的 baseline 改为固定值 0（即 `A = R`），会发生什么？和改为 `A = R - EMA(R)` 相比呢？

8. SPO 的 LR = 1e-7，介于 GRPO（8e-8）和 PPO（1e-6）之间。这个值的设计逻辑是什么？

9. `get_per_token_logps` 中 `input_ids.detach().clone()` 的 `is_inference()` 检查是做什么用的？

**参考答案**

> **第 6 题**：
>
> | 公式 | DPO | GRPO | SPO |
> |------|:---:|:----:|:---:|
> | 策略梯度 | — | `-logp · A` | `-logp · A` |
> | 基线/优势 | — | `(R-μ)/σ` | `R-baseline` |
> | KL 惩罚 | — | `per_token_kl` | `per_token_kl` |
> | 对比偏好 | `log(σ(β·Δlogp))` | — | — |
>
> GRPO 和 SPO 的 Loss 结构几乎一样，唯一的区别是优势 A 的计算方式不同。GRPO 的 `detach()` 技巧让 `exp(x - x.detach()) = 1`，所以 `-logp · A` 在数值上等价于 `-exp(x-x.detach())·A`。
>
> DPO 则完全不同——它没有在线采样，也没有优势/KL，而是直接从 `chosen/rejected` 对中学习偏好对比信号。
>
> **第 7 题**：
>
> - **固定 baseline=0**：`A = R`。训练会出现严重的"prompt 难度偏差"——简单 prompt 一直高分（正优势），难 prompt 一直低分（负优势），模型偏向只回答简单 prompt。GRPO/PPO/SPO 引入基线就是为了消除这个偏差。
>
> - **baseline = EMA(R)**：`A = R - EMA(R)`。这是滑动平均基线，比 SPO 的 Beta 分布更简单。优点是实现更简单，缺点是 EMA 只有一个标量，没有分布信息。实际上 SPO 的 Beta 均值在"只取均值"的情况下等价于 EMA（当 rho 固定时），但 Beta 分布保留了 α,β 两个参数，理论上可以估算置信区间。
>
> **第 8 题**：LR 的设计逻辑：
>
> | 方法 | LR | 原因 |
> |------|:---:|------|
> | PPO | 1e-6 | 裁剪机制兜底，不怕大更新 |
> | **SPO** | **1e-7** | 滑动基线提供一定稳定性，但无裁剪，折中 |
> | GRPO | 8e-8 | 组内归一化仅去偏不去噪，需要保守 |
> | DPO | 4e-8 | 静态数据，最保守 |
>
> SPO 的 LR 在 PPO 和 GRPO 之间，反映了：滑动基线提供了比 GRPO 更平滑的梯度信号（baseline 跨 batch 一致，不像 GRPO 的组内统计每步跳变），所以可以比 GRPO 略大；但没有 PPO 的裁剪，不能像 PPO 那么激进。
>
> **第 9 题**：`is_inference()` 是 PyTorch 中判断 tensor 是否处于推理模式的检查。当 tensor 在推理模式下（如 `torch.inference_mode()` 上下文），它不允许原地修改。`detach().clone()` 创建一个新的、可修改的副本。在训练模式下，这步是冗余的；在推理模式下，这是必要的。这是一种防御性编程。

### 深入题

10. `AutoAdaptiveValueTracker` 只用到了 Beta 分布的均值。如果你要做改进，可能怎么利用 Beta 分布的方差信息？

11. SPO 的 baseline 是全局统一的（所有 prompt 共享一个 baseline），而 PPO 的 Critic 是每步每个 prompt 单独的 `V(s)`。这个差异会带来什么问题？

12. SPO 和 GRPO 在"每 prompt 只生成 1 个回复" vs "每 prompt 生成 8 个回复"之间做出了不同选择。在什么场景下 SPO 的选择更优？

**参考答案**

> **第 10 题**：Beta 分布的方差为 `αβ / ((α+β)²(α+β+1))`。方差信息可以用来：
> - **自适应学习率**：方差大说明基线不确定，应降低更新步长
> - **自适应 KL 系数**：方差大说明基线不可靠，应加强 KL 约束防止跑偏
> - **探索策略**：方差大时鼓励更多探索（不确定性高）
>
> 当前实现只用了均值，相当于只取了 Beta 分布的一点信息，还有改进空间。
>
> **第 11 题**：全局统一 baseline 的问题：
>
> 不同 prompt 的难度不同：难 prompt 平均 0.2，易 prompt 平均 0.8。全局 baseline 如果是 0.5：
> - 难 prompt：`R=0.2, baseline=0.5 → A=-0.3`，**所有**难 prompt 都是负优势
> - 易 prompt：`R=0.8, baseline=0.5 → A=+0.3`，**所有**易 prompt 都是正优势
>
> 结果：模型偏向回答简单问题，回避难题。
>
> PPO 的 Critic 按 prompt 的 `V(s)` 解决了这个问题——简单题 V 高、难题 V 低，A 的正负只取决于回答质量，不取决于难度。GRPO 的组内归一化也解决了这个问题——同一 prompt 内多个回复互相比较。
>
> SPO 的全局 baseline 在这点上最弱。只有当 Reward Model 的评分分布稳定且 prompt 难度差异不大时，SPO 的全局基线才够用。
>
> **第 12 题**：SPO 策略更优的场景：
>
> | 场景 | SPO 更好 | GRPO 更好 |
> |------|:---------:|:---------:|
> | 生成计算成本高（显存/时间） | ✅ 只需生成 1 个回复 | ❌ 需生成 8 个，显存和时间 ×8 |
> | 实时更新（每步需快速迭代） | ✅ 一次生成即可 | ❌ 需等待 8 个回复 |
> | prompt 难度差异小 | ✅ 全局基线够用 | ✅ 同样好 |
> | prompt 难度差异大 | ❌ 全局基线有偏差 | ✅ 组内对比消除难度影响 |
> | 奖励信号极其稀疏 | ❌ 单个回复信息少 | ✅ 多个回复互相弥补 |
>
> > 🐶 **一句话**：生成成本高 → SPO；难度差异大或奖励稀疏 → GRPO。

---

## 七、关联文件

```
train_spo.py
 ├─ scripts/Model/model_minimind.py               ← Policy、Ref 共享的模型定义
 ├─ dataset/lm_dataset.py                 ← RLAIFDataset（与 GRPO/PPO 相同的数据集）
 ├─ scripts/Trainer/trainer_utils.py              ← 工具函数（init_model, SkipBatchSampler 等）
 ├─ scripts/Trainer/train_grpo.py                 ← 对比学习：GRPO（组内归一化，无 Critic）
 ├─ scripts/Trainer/train_ppo.py                  ← 对比学习：PPO（有 Critic，有裁剪）
 ├─ plan/train_grpo_study_plan.md         ← 前置学习：GRPO 学习计划
 └─ plan/train_ppo_study_plan.md          ← 前置学习：PPO 学习计划
```
