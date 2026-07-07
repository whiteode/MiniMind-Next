# train_spo.py 学习计划指引

## 一、文件定位

`train_spo.py` 是 MiniMind 项目的**自博弈优化**（Self-Play Optimization / SPO）训练脚本，
属于偏好对齐阶段。与 DPO / GRPO / PPO 同属**强化学习/偏好优化家族**的第四个变体。

```
                                ┌─ DPO（静态偏好对比）
                                ├─ GRPO（无 Critic，组内归一化）
       训练管线（对齐阶段） ────┼─ PPO（有 Critic 网络，标准 RLHF）
                                └─ SPO（无 Critic，自适应基线） ← 你在这里
```

### 为什么学完 DPO / GRPO / PPO 还要学 SPO？

四种方法构成一条完整的光谱：

| 方法 | 基线来源 | 模型数 | 每 prompt 生成数 | 裁剪 | 优点 |
|------|---------|:------:|:----------------:|:----:|------|
| DPO | 无（静态偏好对） | 2 | 0 | 无 | 最简单，离线数据 |
| GRPO | 组内均值 μ | 3 | 8 | 无 | 无额外模型，组内对比天然无偏 |
| PPO | Critic V(s) | 5 | 1 | ✅ | Critic 给出细粒度价值估计 |
| **SPO** | **滑动窗口自适应基线** | **3** | **1** | **无** | 简单且稳定，仅用 1 次生成 |

**SPO 的核心创新**：用 `AutoAdaptiveValueTracker` 维护一个跨 batch 的运行基线来替代：
- Critic 模型（PPO 的做法）—— 省掉一个模型
- 组内多采样（GRPO 的做法）—— 每 prompt 只生成 1 个回复

它的直觉是：**Reward Model 的评分分布变化很慢**，不需要每步用 Critic 重新学，也不需要用 8 个采样来估计——只需要一个滑动窗口的去偏 Beta 分布追踪即可。

### 四种优势计算方式对比

```
DPO:   无优势，直接从对比对学习（log π(chosen) - log π(rejected)）
GRPO:  A = (R - μ) / σ            ← 组内统计
PPO:   A = R - V(s)               ← Critic 估计
SPO:   A = R - baseline            ← 滑动窗口基线
```

---

## 二、核心概念

### 2.1 三模型架构（与 GRPO 完全一致）

SPO 只需要 **3 个模型**（`train_spo.py` L298-313）：

```python
# ① Policy 模型（可训练）— 要优化的策略
model, tokenizer = init_model(lm_config, base_weight, device=args.device)

# ② Reference 模型（冻结）— KL 散度的基线
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)

# ③ Reward 模型（冻结）— 给回复打分
reward_model = AutoModel.from_pretrained(args.reward_model_path, ...)
reward_model = reward_model.to(args.device).eval().requires_grad_(False)
```

| 模型 | 可训练 | 作用 |
|------|:-----:|------|
| Policy | ✅ | 生成回复 + 策略梯度更新 |
| Ref | ❌ | KL 散度计算的基线 |
| Reward | ❌ | 给最终回复打分 |

和 GRPO 一样——**没有 Critic，没有 Old Actor**。

但与 GRPO 不同的是，SPO **每个 prompt 只生成 1 个回复**（和 PPO 一样）。

### 2.2 AutoAdaptiveValueTracker 设计（L27-66）

这是 SPO 最核心的组件，替代了 PPO 的 Critic 和 GRPO 的组内统计。

#### 2.2.1 Beta 分布基线

`AutoAdaptiveValueTracker` 在内部维护一个 **Beta 分布** `(α, β)`：

```python
N_init = 1.0 / (1.0 - clip_lower)  # = 1.0 / 0.5 = 2.0
self.alpha = 0.5 * N_init  # = 1.0
self.beta  = 0.5 * N_init  # = 1.0
```

初始时 `α=β=1`，Beta 分布在 `[0, 1]` 上均匀分布，baseline = 0.5。

**baseline 的计算**（L40-42）：

```python
def get_baselines(self, batch_size):
    baseline = self.alpha / (self.alpha + self.beta)   # Beta 分布的均值
    return torch.full((batch_size,), baseline)
```

取 Beta 分布的**均值** `α/(α+β)` 作为当前基线。

**为什么要用 Beta 分布而不是直接做 EMA？**

Reward Model 的评分在 `[-3, 3]` 范围内，标准化到 `[0, 1]` 后可视为一个**伯努利试验的累计成功率**：
- 高分（接近 1）≈ 成功
- 低分（接近 0）≈ 失败

Beta 分布是伯努利试验的**共轭先验**——只需要维护 `α`（累计成功次数）、`β`（累计失败次数）两个参数，就能完整描述当前基线的不确定性。

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

每次 `update()` 时，用当前 batch 的平均 reward 做一次**指数滑动更新**：

```
α_new = ρ · α_old + R̄            （β 同理，用 1 - R̄）
```

其中 **ρ（rho）** 是衰减率——控制历史信息保留多久。新的 reward 贡献 `R̄` 到 α，`1-R̄` 到 β。

所以：
- 如果 R̄ 高 → α 增加得比 β 多 → 基线上升
- 如果 R̄ 低 → β 增加得比 α 多 → 基线下降

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

**ρ 的自适应机制**：

- **logprob 变化小**（模型收敛）：kl 小 → rho 接近 1.0 → 历史保留多，基线稳定
- **logprob 变化大**（模型还在活跃探索）：kl 大 → rho 变小 → 历史丢弃快，基线快速适应

公式解析：`rho = 2^(-kl / D_half)`，其中 `D_half=0.06` 是半衰期参数。

```
kl = 0.00 → rho = 2^(0)      = 1.00   历史几乎全保留
kl = 0.06 → rho = 2^(-1)     = 0.50   半衰期：历史权重减半
kl = 0.12 → rho = 2^(-2)     = 0.25
kl = 0.60 → rho = 2^(-10)    ≈ 0.001  几乎完全丢弃历史
```

所以 `D_half=0.06` 的含义是：**当 logprob 变化 0.06 nats 时，历史权重衰减一半**。

#### 2.2.4 与 GRPO / PPO 的对比

| | GRPO | PPO | SPO |
|---|---|---|---|
| 基线 | `(R-μ)/σ` 组内统计 | `V(s)` 独立 Critic | `α/(α+β)` 滑动 Beta |
| 历史保留 | 无（每步独立） | Critic 权重自动保留历史 | ρ 控制的指数衰减 |
| 跨 batch 一致性 | ❌ 每步独立 | ✅ Critic 有记忆 | ✅ 通过 α,β 有记忆 |
| 额外模型 | 无 | 1 个 Critic | 无 |
| 额外生成 | 7 个额外回复 | 0 | 0 |

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
| `logits_to_keep=n_keep + 1` | 只需要最后 `n_keep+1` 个 token 的 logits（节省计算量） |
| `logits[:, :-1, :]` | 去掉最后一个位置的 logits，保持和 labels 长度对齐 |
| 逐行 for 循环 | 对 batch 内每个样本分别做 `gather`，避免 `flash_attn` + `logits_to_keep` 在 batch 维度上的兼容性问题 |
| `log_softmax + gather` | 取出每个位置 target token 的 log_prob |

**与 PPO/GRPO 的差异**：

| | PPO | GRPO | SPO |
|---|---|---|---|
| logp 获取方式 | 一次 forward + gather | 一次 forward + gather | **逐行循环** gather |
| 返回粒度 | 序列级（sum over tokens） | token 级 `[B, R]` | token 级 `[B, R]` |
| 为什么循环 | — | — | 兼容 flash_attn 的 `logits_to_keep` 限制 |

这种逐行 gather 的方式是 PR 级别的优化细节——`logits_to_keep` 配合 flash_attn 时可能不支持 batch 维度的 gather，所以逐个样本处理。

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
3. **优势裁剪**：`clamp(-5, 5)` 防止个别极端的评分差异导致梯度爆炸（没有 PPO 的 clip 机制，所以用 clamp 兜底）

**四方法优势计算对比**：

```
DPO:  无优势（直接用偏好对）
GRPO: A = (R - μ) / σ                ← 组内归一化，均值 μ，标准差 σ
PPO:  A = R - V(s)                   ← Critic 估计的价值
SPO:  A = R - (α/(α+β) · 6 - 3)     ← Beta 分布均值反标准化
```

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

### 2.6 completion_mask 构造（L179-182）

```python
is_eos = completion_ids == tokenizer.eos_token_id                # [B, R] 找到 eos 位置
eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), ...)    # 默认 R（序列末）
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]  # 找到 eos 位置
completion_mask = (torch.arange(...).expand(...) <= eos_idx.unsqueeze(1)).int()  # 从开头到 eos
```

与 GRPO 完全相同——构造一个 mask，标记从序列开头到 eos（含）之间的所有 token 为有效。eos 之后的 token 被排除在 loss 计算之外。

### 2.7 调度器

```python
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # 默认 1e-7
scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
```

**比较四种方法的 LR**：

| 方法 | 默认 LR | 说明 |
|------|:------:|------|
| DPO | 4e-8 | 静态数据，需稳定 |
| GRPO | 8e-8 | 在线采样，组内归一化提供稳定性 |
| PPO | 1e-6 | 裁剪机制兜底，可激进 |
| **SPO** | **1e-7** | 介于 GRPO 和 PPO 之间，滑动基线提供一定稳定性，但无裁剪 |

SPO 的 LR 比 GRPO 略大（1e-7 vs 8e-8），但没有 PPO 那么大（1e-6），因为：
- 滑动基线提供了一些稳定性（比 GRPO 的纯组内统计更平滑）
- 但没有裁剪机制（不如 PPO 安全），所以不能太激进

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

**Q：初始 α 和 β 为什么设为 1.0？**
A：使得初始基线为 `α/(α+β) = 0.5`（对应反标准化后的 0），表示"初始时不知道 Reward Model 的打分习惯"。随着训练推进，α 和 β 会自适应调整。

**Q：`rho` 的作用是什么？为什么用半衰期公式？**
A：rho 控制历史信息的衰减速度。半衰期公式 `2^(-kl/D_half)` 使得 logprob 变化越大衰减越快——当模型变化快时，旧基线的参考价值降低，需要更快地丢弃。

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

**Q：为什么 `get_per_token_logps` 用 for 循环逐个处理 batch？**
A：代码注释没有明确说明，这是一个实现细节——配合 `flash_attn` + `logits_to_keep` 时，batch gather 可能触发兼容性问题。逐个处理是最稳妥的做法，虽然牺牲了一点并行效率。

**Q：为什么 advantages 要做 `clamp(-5, 5)`？**
A：SPO 没有 PPO 的裁剪机制，也没有 GRPO 的组内归一化。单个样本可能因为 Reward 模型在分布外的极端预测而产生非常大的优势值，`clamp` 是最后的防线。

**Q：GRPO 也有 `per_token_kl` 的写法，和 SPO 一样吗？**
A：完全一样。`kl_div = ref_logp - logp` 是逐 token 的 log 概率差；`exp(kl_div) - kl_div - 1` 是其在 0 处泰勒展开的前两项，既保证了 KL 非负，又不会在负方向过度惩罚。

**Q：为什么 Loss 没有做 `sum(dim=1)` 而是逐 token 乘 mask 再求平均？**
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
> GRPO 和 SPO 的 Loss 结构几乎一样，唯一的区别是优势 A 的计算方式不同。GRPO 的 `detach()` 技巧让 `exp(x - x.detach()) = 1`，所以 `-logp · A` 在数值上等价于 `-exp(x-x.detach())·A·x·A`。
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
> | 推理成本极高（API 调用收费） | ✅ 只需 1 次调用 | ❌ 需要 8 次 |
> | 实时更新（每步需快速迭代） | ✅ 一次生成即可 | ❌ 需等待 8 个回复 |
> | prompt 难度差异小 | ✅ 全局基线够用 | ✅ 同样好 |
> | prompt 难度差异大 | ❌ 全局基线有偏差 | ✅ 组内对比消除难度影响 |
> | 奖励信号极其稀疏 | ❌ 单个回复信息少 | ✅ 多个回复互相弥补 |
>
> **一句话**：生成成本高 → SPO；难度差异大或奖励稀疏 → GRPO。

---

## 七、关联文件

```
train_spo.py
 ├─ model/model_minimind.py               ← Policy、Ref 共享的模型定义
 ├─ dataset/lm_dataset.py                 ← RLAIFDataset（与 GRPO/PPO 相同的数据集）
 ├─ trainer/trainer_utils.py              ← 工具函数（init_model, SkipBatchSampler 等）
 ├─ trainer/train_grpo.py                 ← 对比学习：GRPO（组内归一化，无 Critic）
 ├─ trainer/train_ppo.py                  ← 对比学习：PPO（有 Critic，有裁剪）
 ├─ plan/train_grpo_study_plan.md         ← 前置学习：GRPO 学习计划
 └─ plan/train_ppo_study_plan.md          ← 前置学习：PPO 学习计划
```
