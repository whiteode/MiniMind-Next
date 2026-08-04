# train_ppo.py 学习计划指引

## 一、文件定位

`train_ppo.py` 是 MiniMind 项目**近端策略优化**（Proximal Policy Optimization）训练脚本，
是模型对齐阶段的另一种实现方式，与 `train_grpo.py` 同属 **强化学习家族**。

```
                            ┌─ DPO（静态偏好对比）
      训练管线（对齐阶段） ──┼─ GRPO（无 Critic，组内归一化）
                            └─ PPO（有 Critic 网络，标准 RLHF） ← 你在这里
```

### 为什么学完 GRPO 还要学 PPO？

| | GRPO | PPO |
|---|---|---|
| Critic 网络 | ❌ 不需要 | ✅ 需要，额外训练一个价值网络 |
| 优势计算 | 组内归一化 `(R - μ)/σ` | Critic 输出 baseline `A = R - V(s)` |
| 每组采样数 | `num_generations=8` | 每个 prompt 只生成 1 个回复 |
| 旧策略保存 | 无（用 `detach()` 技巧） | 有，`old_actor_model` 每 K 步同步一次 |
| 裁剪机制 | 无 | `clip_epsilon` 标准 PPO 裁剪 |
| 梯度来源 | 策略梯度 + KL | 策略梯度 + 价值 loss + KL |

**一句话**：GRPO 是 PPO 的"轻量版"——用组内多个采样替代了 Critic 网络的开销和复杂度。
学 PPO 能帮你理解"为什么可以去掉 Critic"以及"去掉 Critic 付出了什么代价"。

### 到底哪个好？

没有绝对的好坏，取决于你的**资源瓶颈在哪**。

```
资源瓶颈是"显存 / 模型复杂度" → GRPO 好
    └── 不需要额外训练一个 Critic，省 1 个模型的开销

资源瓶颈是"生成速度 / 推理成本" → PPO 好
    └── 每个 prompt 只生成 1 个回答，GRPO 要生成 8 个
```

详细对比：

| 维度 | PPO | GRPO | 谁赢 |
|------|-----|------|:----:|
| **显存占用** | 多一个 Critic 模型 | 少一个模型 | **GRPO** |
| **每步生成量** | 1 个 / prompt | 多个（默认 8）/ prompt | **PPO** |
| **优势估计质量** | 依赖 Critic 训练质量（可能不收敛） | 组内统计天然无偏，采样越多越准 | **GRPO**（稀疏奖励任务） |
| **训练稳定性** | 有裁剪 + Critic baseline 双重保障 | 组内归一化 + 二次归一化，也很稳 | 平手 |
| **实现复杂度** | 5 个模型、两个优化器、同步逻辑 | 3 个模型，简单清晰 | **GRPO** |

**业界实践**：DeepSeek 在 DeepSeekMath 论文中发现，对于**数学推理**这类奖励信号稀疏且明确的任务，GRPO 优于 PPO。原因是 Critic 很难对"推理到一半"的状态给出准确价值估计。

而对于**对话 / 写作**这类奖励信号连续且密集的任务，一个训练良好的 Critic 能提供更细粒度的优势估计，PPO 仍占有优势。

**一句话结论**：能承受多轮生成成本 → **GRPO**；主要瓶颈在生成延迟 → **PPO**。

---

## 二、核心概念

### 2.1 五模型架构

PPO 需要 **5 个模型**（`train_ppo.py` L29-41, L305-335）：

```python
# ① Actor 模型 — 要训练的 policy，从 reason/full_sft 权重初始化
actor_model, _ = init_model(lm_config, base_weight, device=args.device)

# ② Old Actor 模型 — Actor 的延迟同步副本，用于重要性采样
old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
old_actor_model = old_actor_model.eval().requires_grad_(False)

# ③ Reference 模型 — Actor 的冻结副本，用于 KL 散度约束
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)

# ④ Critic 模型 — 价值网络，估计状态值 V(s)，与 Actor 共享主体但有多出来的 value_head
critic_model = CriticModel(lm_config)
critic_model.load_state_dict(state_dict, strict=False)  # value_head 随机初始化

# ⑤ Reward 模型 — 外部评分模型，独立加载，完全冻结
reward_model = AutoModel.from_pretrained(args.reward_model_path, ...)
reward_model = reward_model.to(args.device).eval().requires_grad_(False)
```

| 模型 | 可训练 | 作用 | 独特之处 |
|------|-------|------|---------|
| Actor | ✅ | 生成回复 + 策略梯度更新 | — |
| Old Actor | ❌ | **延迟**固定副本，提供重要性采样的分母 `π_old` | 每 K 步同步一次最新权重 |
| Ref | ❌ | KL 散度计算的基线 | 全程冻结，不与 Actor 同步 |
| Critic | ✅ | 给每个状态估计价值 `V(s)` | 共享 Actor 主体 + 独立的 `value_head` |
| Reward | ❌ | 给最终回复打分 | 完全外部加载，结构可不同 |

与 GRPO 的区别：
- GRPO：3 模型（policy + ref + reward），**没有 Critic 和 old_actor**
- PPO：5 模型，**多了一个 Critic 和一个 old_actor**

### 2.2 CriticModel 设计（L29-41）

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)  # 替换 lm_head

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, ...)          # 共享主体
        hidden_states = self.model.norm(outputs[0])             # 取最后一层 hidden
        values = self.value_head(hidden_states).squeeze(-1)     # 投影为标量
        return values  # [B, T] 每个 token 一个价值估计
```

关键要点：
- **继承自 `MiniMindForCausalLM`**，所以共享了完整的 Transformer 主体
- 用 `value_head`（`Linear(hidden_size, 1)`）替换了 `lm_head`（`Linear(hidden_size, vocab_size)`）
- 输入 `[B, T]` 的 token 序列，输出 `[B, T]` 的标量价值序列
- 初始化时 `strict=False`——`value_head` 因为没有在 checkpoint 中找到对应权重而**随机初始化**
- 这意味着：Critic 不是从头训练的，而是**在语言模型基础上加了一个随机初始化的价值头**

#### value_head 随机初始化了，Critic 怎么估计价值？

虽然 `value_head` 的权重是随机的，但它的**输入不是随机噪音**——是预训练语言模型的 `hidden_states`，已经蕴含了丰富的语义和世界知识。Critic 等于在用有意义的特征做一个简单的线性回归，不是从零开始学。

而且 Critic 在训练中每步都通过 `MSE(values, rewards)` 更新，学习速度极快：

| 原因 | 说明 |
|------|------|
| ① 输入特征好 | 共享主体已能提取高质量语言表示，value_head 只需学一个线性映射 |
| ② 监督信号直接 | 每个样本只需逼近一个标量 `rewards`，任务极简单 |
| ③ 学习率足够 | Critic 的 LR=1e-6，单层 Linear 收敛只需几十步 |

**数值示例**：
```
Step 0:  value_head 随机初始化 → V = 0.12（随机猜），R = 0.85
         advantage = 0.85 - 0.12 = 0.73（噪音大，但方向对）

Step 10: 经过 MSE 训练 → V = 0.70（接近真实评分）
         advantage = 0.85 - 0.70 = 0.15（噪音大幅降低）

Step 100: V ≈ 0.83（基本学会预测）
```

**对比 GRPO**（加深理解）：
```
PPO:   V 从随机→学会，需要训练成本，但每个 prompt 只生成 1 个回复
GRPO: 不用学 V，直接用组内均值 μ 替代，但每个 prompt 要生成 8 个回复

PPO 的 Critic 承担的角色 = GRPO 的"额外 7 个采样"
```

### 2.3 优势计算（L140-144）

```python
values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
last_indices = (full_mask * torch.arange(...)).argmax(dim=1)            # 找最后一个非 pad 位置
values = values_seq[torch.arange(B), last_indices]                      # [B] 最终价值
advantages = rewards - values.detach()                                  # [B] 优势
```

这是 PPO 最核心的设计决策之一：**优势 = 实际奖励 - Critic 估计的价值**。

对比 GRPO 的优势计算：
```
PPO:   A = R - V(s)          ← 需要一个 Critic 模型来估计 V(s)
GRPO:  A = (R - μ_group)/σ   ← 用组内统计替代 Critic
```

所以 GRPO 的本质是：**用每组多个采样的均值 μ 来近似 V(s)**。这意味着 GRPO 在组内样本足够多时（如 8 个）无需训练 Critic 也能得到合理的优势估计。

### 2.4 PPO 裁剪机制（L169-172）

```python
ratio = torch.exp(actor_logp - old_logp)  # [B] 重要性采样权重
surr1 = ratio * advantages                  # 未裁剪
surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

裁剪的核心逻辑：
```
如果 advantage > 0（好回复）：
  ratio 提升 → surr1 变大 → 但 surr2 限制了上限（1+ε）
  → 防止 actor 为了一次好运气而过度上调概率

如果 advantage < 0（差回复）：
  ratio 降低 → surr1 变小 → 但 surr2 限制了下限（1-ε）
  → 防止 actor 为了一次坏运气而过度下调概率
```

`clip_epsilon=0.1` 意味着 ratio 被限制在 [0.9, 1.1] 范围内，即单步更新最多改变 10% 的概率。
GRPO 没有这个裁剪，因为它的 ratio 就是 `exp(0) = 1`（恒为 1）。

### 2.5 旧策略同步机制（L222-227）

```python
if (step + 1) % args.update_old_actor_freq == 0:
    raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
    raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
    state_dict = raw_actor.state_dict()
    old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
    old_actor_model.to(args.device)
```

这是标准 PPO 的 **延迟策略更新**（deferred policy update）：
- `old_actor_model` 不是全程冻结的，而是**每 K 步同步一次**
- 在这 K 步内，`old_actor_model` 不变，Actor 在持续更新
- ratio = `exp(actor_logp - old_logp)` 反映了**在这 K 步内**策略的累积变化
- K 越大，ratio 越可能偏离 1，裁剪越频繁被触发

对比 GRPO：GRPO 根本没有 old_actor，因为它**每步重新采样**，ratio 恒为 1。

### 2.6 完整的 PPO Loss（L167-174）

```python
# ① KL 散度（参考模型，用于 loss）
kl_ref = (actor_logp - ref_logp).mean()

# ② PPO 裁剪代理
ratio = torch.exp(actor_logp - old_logp)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
policy_loss = -torch.min(surr1, surr2).mean()

# ③ 价值损失
value_loss = F.mse_loss(values, rewards)

# ④ 总损失
loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss)
```

PPO loss 共 **4 项**，每项解决一个不同的子问题。下面逐项拆解。

#### ① KL 惩罚 `(actor_logp - ref_logp).mean()`

```python
kl_ref = (actor_logp - ref_logp).mean()
```

`actor_logp` 和 `ref_logp` 在前面代码中经过 `resp_mask` 屏蔽 prompt 部分、`sum(dim=1)` 压缩为每个样本一个标量（shape `[B]`），所以这里的 `.mean()` 是对 batch 取平均。

| `actor_logp - ref_logp` | 含义 | PPO 反应 |
|:---:|---|---|
| > 0 | Actor 比 Ref **更自信**（概率更高） | 惩罚，不让偏离太远 |
| = 0 | 一样 | 无惩罚 |
| < 0 | Actor 比 Ref **更不自信**（概率更低） | 同样惩罚 |

对比 GRPO 的 KL：
```
PPO:   kl_ref = (actor - ref).mean()           ← 序列级 KL（一个样本一个值）
GRPO:  per_token_kl = exp(kl_div) - kl_div - 1 ← token 级 KL（每个 token 一个值）
```
PPO 的 KL 更"粗"但够用，因为有裁剪机制兜底。

#### ② 裁剪代理 — 让策略梯度不走极端

```python
ratio = torch.exp(actor_logp - old_logp)  # ① 重要性采样权重
surr1 = ratio * advantages                  # ② 未裁剪代理
surr2 = torch.clamp(ratio, 0.9, 1.1) * advantages  # ③ 裁剪后的代理
policy_loss = -torch.min(surr1, surr2).mean()  # ④ 取小 + 取负
```

**① `ratio`：真正的"重要性采样"** — 和 GRPO 不同！

| | 分母来源 | ratio 是否恒为 1？ |
|---|---|---|
| PPO | `old_actor_model`（**上一个同步点**的冻结 Actor） | ❌ 否，Actor 更新后 ratio ≠ 1 |
| GRPO | `detach()`（同一次前向的副本） | ✅ 是，恒为 1 |

```
ratio > 1.0：Actor 比上次同步时更倾向于这个 token（概率提升了）
ratio < 1.0：Actor 比上次同步时更不倾向于这个 token（概率降低了）
```

**②-③ 裁剪的作用**：用一个具体的例子看四个场景：

| 场景 | ratio | advantage | surr1 | surr2 | min 选谁 | 效果 |
|:---:|:---:|:---------:|:-----:|:-----:|:--------:|------|
| 好回答 + 过度乐观 | 1.3 | +1.0 | 1.30 | **1.10** | surr2 | 限速！不让一步更新 30% |
| 好回答 + 正常 | 1.05 | +1.0 | **1.05** | 1.10 | surr1 | 正常更新 |
| 差回答 + 过度悲观 | 0.7 | -0.5 | -0.35 | **-0.45** | surr2 | 限速！不让一步降 30% |
| 差回答 + 正常 | 0.95 | -0.5 | **-0.475** | -0.45 | surr1 | 正常更新 |

##### 为什么 min 这么选？先把"保守"放一边，看一条简单规则：

**`min(surr1, surr2)` 选的一律是 ratio 落在 `[0.9, 1.1]` 内的那个版本。**

表格里重新看：

```
好回答 + 过度乐观 (ratio=1.3):
  surr1 用原始 ratio=1.3（超出上界）
  surr2 用 clamp=1.1（落在界内）← min 选这个

好回答 + 正常 (ratio=1.05):
  surr1 用原始 ratio=1.05（落在界内）← min 选这个
  surr2 用 clamp=1.1（落在界内，但不用它）

差回答 + 过度悲观 (ratio=0.7):
  surr1 用原始 ratio=0.7（超出下界）
  surr2 用 clamp=0.9（落在界内）← min 选这个

差回答 + 正常 (ratio=0.95):
  surr1 用原始 ratio=0.95（落在界内）← min 选这个
  surr2 用 clamp=0.9（落在界内，但不用它）
```

**和 advantage 正负无关！** clip 先 clamp ratio，min 再选落在界内的那个。advantage 正负只影响最终数字的大小，不影响 min 选谁。

##### 那为什么差回答 + 过度悲观时，clip 后的 surr2(-0.45) 比 surr1(-0.35) 更负？

这里有个反直觉的点——因为 advantage 是负数（-0.5），ratio 乘以负数后大小关系**反转**了：

```
ratio = 0.7 (低级)    → surr1 = 0.7 × (-0.5) = -0.35
ratio = 0.9 (被拉高)  → surr2 = 0.9 × (-0.5) = -0.45（更负！）

            数字越大 → 乘以负数 → 越小
```

类比——**负数乘法会反转大小关系**：
```
正常人："3 比 2 大"
乘以 -1 后："-3 比 -2 小"  ← 顺序完全反过来
```

所以 clip **提高**了 ratio（0.7 → 0.9），但因为是乘以 **负数** advantage，结果反而更大了。这就是为什么差回答场景下，"safe" clip 产生的 loss 反而比原始值还大。

##### 所以"保守"到底是什么意思？

保守不是指 loss 大小，而是指 **更新步长有上限**：

```
差回答 + 过度悲观 (ratio=0.7 → clip=0.9):
  如果不 clip: loss = -(-0.35) = +0.35，但 ratio 本该是 0.7，还在持续下跌
  如果 clip:    loss = -(-0.45) = +0.45，但 ratio 被卡在 0.9 下不来
  保守体现在：ratio 不会再低了！"更新幅度被控制住了"
```

| 场景 | clip 在保护什么？ | 通俗说法 |
|------|------------------|---------|
| 好回答 + ratio 飙升 | 别让 ratio 涨到 1.3，卡在 1.1 | "油门限速" |
| 差回答 + ratio 暴跌 | 别让 ratio 跌到 0.7，卡在 0.9 | "刹车限速" |

都是 **把 ratio 拉回 [0.9, 1.1] 区间内**，限制单步更新的最大幅度。

##### 外层负号：为什么最大化问题要取负？

```python
policy_loss = -min(surr1, surr2).mean()    # ← 这个负号
```

PPO 和 GRPO 的目标函数都是**最大化**（让好回答概率更高、差回答概率更低），但 `optimizer.step()` 内部固定做 `θ -= lr × grad`——**永远是梯度下降**。

想让下降变上升，只需在目标前加个负号：

```
最大化目标： J = E[ min(surr1, surr2) ]         越大越好
实际传的 loss： L = -J                           越小越好

L.backward()  → ∇L = -∇J
step()        → θ = θ - lr × (-∇J) = θ + lr × ∇J  ← 实际在增大 J！
```

| | 没取负号 | 取了负号 |
|---|---|---|
| loss | `min(surr1, surr2)` | `-min(surr1, surr2)` |
| `step()` 的效果 | `θ -= lr × ∇J` ↓ 减小 J ❌ | `θ += lr × ∇J` ↑ 增大 J ✅ |

这就是外层负号的全部意义——**把 PyTorch 的"最小化"骗成"最大化"**。PPO、GRPO、乃至所有强化学习的策略梯度代码都有这个负号。

#### ③ 价值损失 `MSE(V, R)`

```python
value_loss = F.mse_loss(values, rewards)
```

Critic 的本质是**回归模型**——学习预测"这个回复能得多少分"：

```
输入 state 的 hidden states → Critic → 预测 V = 0.72
真实 reward = 0.85
MSE = (0.85 - 0.72)² = 0.0169 → 反向传播 → 更新 value_head + 共享主体
```

Critic 训练得好不好直接影响策略梯度质量：
- Critic 准 → `A = R - V(s)` 准确 → 策略梯度信号好
- Critic 差 → A 中噪音大 → 策略梯度信号差

#### ④ 总 Loss 合并

```python
loss = (policy_loss               # Actor: 学偏好
        + vf_coef * value_loss    # Critic: 学会评分
        + kl_coef * kl_ref        # Actor: 别跑偏
        + aux_loss)               # MoE 负载均衡
```

三个目标在同一个 loss 里共存，系数调节各自权重。

超参数对比（和 GRPO）：

| 参数 | PPO | GRPO | 作用对象 |
|------|:---:|:----:|---------|
| `vf_coef` | ✅ 有 | ❌ 无 | Critic 的学习权重 |
| KL 系数 | `kl_coef` | `beta`（0.02） | KL 约束强度 |
| `clip_epsilon` | 0.1 | ❌ 无 | 策略更新步长上限 |

PPO 多一个 `vf_coef` 要调——Critic 学太快会过拟合当前 batch 的噪音，学太慢则优势估计质量差。

### 2.7 学习率与优化器

```python
actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)       # 默认 1e-6
critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)  # 默认 1e-6

actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_steps, eta_min=lr/10)
critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_steps, eta_min=critic_lr/10)
```

Actor 和 Critic 有**独立的优化器和学习率调度器**。PPO Actor 的 LR=1e-6 比 GRPO 的 8e-8 大得多（12.5 倍），因为 PPO 的裁剪机制提供了额外的稳定性保障。

#### CosineAnnealingLR 是什么？

**余弦退火**（Cosine Annealing）调度器。学习率按半个余弦波的形状从初始值 `lr` 下降到最低值 `eta_min`：

```
学习率
  ↑
  |   lr ───╮
  |          ↘
  |           ↘
  |            ↘
  |             ↘
  |    eta_min ───────→  训练步数
  |
  └────────────────────→  T_max
```

**关键参数：**

| 参数 | 值 | 含义 |
|------|:---:|------|
| `T_max` | `total_steps` | 半个余弦波的周期长度，走到 `T_max` 步时 LR 降到 `eta_min` |
| `eta_min` | `lr / 10` | 学习率的最低值 |

**为什么用余弦退火？**

训练前期 LR 大，模型大步探索；后期 LR 小，精细调整。余弦曲线比阶梯式下降更平滑：

| 调度器 | 形状 | 特点 |
|--------|:----:|------|
| StepLR | 阶梯下降 | 每到固定步数 LR 砍半，有冲击 |
| CosineAnnealingLR | 平滑曲线 | 每步缓慢下降，无冲击 |
| LinearLR | 直线下降 | 简单线性衰减 |

**当前代码效果：**
```
Actor LR：  1e-6 ─→ 1e-7（lr/10）
Critic LR： 1e-6 ─→ 1e-7
```
两个调度器步伐一致（`T_max` 相同、`eta_min` 比例相同）。

---

## 三、关键公式一览

### PPO 裁剪
```
ratio = π_θ / π_θ_old                         重要性采样权重
surr1 = ratio × A                              未裁剪代理
surr2 = clip(ratio, 1-ε, 1+ε) × A             裁剪代理  
L_policy = -E[ min(surr1, surr2) ]             策略 Loss
```

### 优势函数
```
V(s) = Critic(input_ids)                       每个状态的价值估计
A = R - V(s)                                   简单优势（非 GAE）
```

### 价值损失
```
L_value = MSE(V(s), R)                         让 Critic 学会预测奖励
```

### 总 Loss
```
L_total = L_policy + vf_coef × L_value + kl_coef × KL(π_θ || π_ref) + aux_loss
```

---

## 四、学习目标检查清单

- [ ] 理解 PPO 的 5 模型架构及其各自的作用
- [ ] 理解 CriticModel 的设计（共享主体 + value_head）
- [ ] 理解 Critic 随机初始化的含义
- [ ] 理解 `A = R - V(s)` 的优势计算方式
- [ ] 理解 PPO 裁剪机制 `min(surr1, surr2)` 的作用
- [ ] 理解旧策略同步机制（每 K 步同步一次）
- [ ] 理解 PPO 的 3 成分 Loss（策略 + 价值 + KL）
- [ ] 对比 PPO 与 GRPO 的优势计算方式差异
- [ ] 对比 PPO 与 GRPO 的模型数量差异
- [ ] 理解 Actor 与 Critic 独立优化器、独立调度器的设计
- [ ] 理解 PPO 在完整管线中的位置

---

## 五、文件逐段精读计划

### 第 1 层：CriticModel 定义（L29-41）

**Q：为什么继承 `MiniMindForCausalLM` 而不是从头写一个新类？**
A：为了共享完整的 Transformer 主体，Critic 直接用预训练好的 hidden states 做输入，不需要从头学特征提取。

**Q：`value_head` 的输入输出维度是什么？**
A：`Linear(hidden_size, 1)`。输入最后一层 hidden states `[B, T, hidden_size]`，输出每个 token 位置的标量值 `[B, T]`。

**Q：`strict=False` 加载权重的含义是什么？**
A：允许 checkpoint 中缺少 `value_head` 的权重（原始语言模型没有这个层），所以 `value_head` **随机初始化**。输入是预训练好的 hidden states + 监督信号是标量 reward，单层 Linear 几步就能收敛。

**Q：对比 Actor 的 `lm_head`，两者输出目标有什么不同？**
A：Actor 输出 vocabulary 上的概率分布（分类问题），Critic 输出标量价值（回归问题）。

### 第 2 层：calculate_rewards（L44-116）

**Q：与 GRPO 的 `calculate_rewards` 有哪些不同？**
A：结构基本一致——都是格式奖励 + Reward Model 评分的双层设计。格式奖励的正则、分值分配（格式完整 +0.5，每个标记 +0.25）完全相同。推理模式下的加权方式也一样：完整回答 40% + answer 部分 60%。

**Q：Reward Model 的调用方式有何特点？**
A：先解析 prompt 中的 `<|im_start|>` 标签构造 messages 列表，然后调用 `reward_model.get_score(reward_tokenizer, tmp_chat)`。

### 第 3 层：训练循环主流程（L123-144）

**Q：`do_sample=True, temperature=0.8` 的含义是什么？**

A：`do_sample=True` 表示非贪婪解码——每次从概率分布中采样，而不是选概率最高的 token。`temperature` 控制采样的"随机程度"：值越小越接近贪婪解码，值越大越随机。

PPO 选 `temperature=0.8` 的原因：既要生成多样化的回复让 Critic 有东西可比较，又不能太随机导致全是噪音。每个 prompt 只生成 **1 个**回复（GRPO 是 8 个）。

类比点唱机：
```
do_sample=False（贪婪）→ 永远只播排行榜第一的歌
do_sample=True, temp=0.8 → 偶尔也播前十的歌，但主要还是榜首
do_sample=True, temp=2.0 → 随机乱播
```

**Q：Critic 为什么要在完整序列上算价值，而不是只算 completion 部分？**

A：Critic 需要同时看到 prompt 和 completion 才能准确估计价值。只看 completion 不知道上下文，无法判断回复好不好。比如只看"是中国的首都"，你没法知道这是满分回答"北京是中国的首都"还是0分回答"东京是中国的首都"。

**Q：为什么只用最后一个 token 的价值？其他 token 的价值呢？**

A：PPO 把整条回复视为一个"单一动作"，而不是把每个 token 生成都当作一个独立动作。最后一个 token 的价值代表整条回复完成时的预期奖励。中间 token 的价值在单步优势中不需要——就像评委只给最终端上桌的菜打分，而不是给切菜、炒菜、调味的每个中间步骤单独打分。

**Q：`last_indices` 的 argmax 技巧是怎么找到最后一个非 padding 位置的？**

A：输入序列用了左侧 padding，所以 mask 中 padding 位置为 0、真实 token 位置为 1。`mask × arange` 的结果中，padding 位置全部为 0，真实 token 位置保留各自的索引值。最后一个非 padding 位置的索引值最大，`argmax(dim=1)` 正好取到这个位置。

数值示例：
```
mask      = [0, 0, 1, 1, 1, 1, 1]     （左侧 2 个 padding）
arange    = [0, 1, 2, 3, 4, 5, 6]
mask × arange = [0, 0, 2, 3, 4, 5, 6]
                              ↑ 最后一个非 padding 位置
                 argmax = 6  ← 取到最大值所在的索引
```

**Q：优势为什么是 `R - V(s)`？减去 V 解决了什么问题？**

A：`A = R - V(s)` 的核心思想是**看实际奖励是否超出预期**，而不是看绝对分数。

类比考试：
```
你考了 85 分（R）。是好是坏？要看平时水平（V）：
  平时 60 分水平 → 超出预期 25 分，A = +25（正优势，鼓励）
  平时 95 分水平 → 低于预期 10 分，A = -10（负优势，惩罚）
```

如果直接用 R 做优势（即 A = R），不同 prompt 的难度差异会导致偏差——简单题天然高分，难题天然低分，模型会偏向只回答简单题。减去 V 后：
```
简单题 R=0.9，V=0.85 → A=+0.05（不算什么大惊喜）
难  题 R=0.2，V=0.15 → A=+0.05（同样值得鼓励！难度被去掉了）
```

这和 GRPO 的 `(R - μ)/σ` 在功能上等价，只是基线来源不同：

| | PPO | GRPO |
|---|---|---|
| 基线来源 | Critic 训练出来的 `V(s)` | 同一 prompt 多采样的均值 `μ` |
| 做的事情 | 减去 prompt 难度差异 | 减去 prompt 难度差异 |
| 额外成本 | 每步多一次 Critic 前向+反向 | 每步多生成 7 个回复 |

**Q：`values` 为什么要 `.detach()`？**

A：`advantages = rewards - values.detach()` 中的 `detach()` 切断了梯度从 advantage 流向 Critic 的路径。如果不 detach，Critic 会同时收到两个互相矛盾的梯度信号：
- 来自 `value_loss = MSE(values, rewards)` 的梯度（训练 Critic 拟合 R）
- 来自 `policy_loss` 中 `-min(surr1, surr2)` 的梯度（通过 advantage 反传到 values）

detach 后，advantage 中的 `V(s)` 只作为常数参考值，Critic 的更新完全由 `value_loss` 控制。这是一个**职责分离**的设计——Actor 的 loss 不影响 Critic，Critic 的 loss 不影响 Actor。

### 第 4 层：Loss 计算（L167-174）

**Q：`ratio = exp(actor_logp - old_logp)` 和 GRPO 的 `exp(x - x.detach())` 有什么本质区别？**
A：PPO 的 `old_logp` 来自 `old_actor_model`（上一个同步点的冻结 Actor），是**真正的**重要性采样权重，ratio 会随 Actor 更新而偏离 1。GRPO 的 `exp(x - x.detach())` 恒为 1。

**Q：裁剪机制在什么时候触发？被裁剪后梯度行为是怎样的？**
A：在 ratio 超出 `[1-ε, 1+ε]` 时触发。`min(surr1, surr2)` 选到 clip 后的版本，ratio 超出区间的梯度被切断，防止单步更新过大。

**Q：为什么 GRPO 不需要这个裁剪？**
A：GRPO 的 ratio 恒为 1，永远在裁剪区间内，没有裁剪的必要。

**Q：`kl_ref = (actor_logp - ref_logp).mean()` 是逐 token 平均还是序列平均？**
A：前面代码中 `actor_logp` 已经通过 `resp_mask` 屏蔽 prompt + `sum(dim=1)` 压缩为 `[B]`，所以 `.mean()` 是 batch 平均——序列级 KL。

**Q：`loss.backward()` 会同时更新 Actor 和 Critic 吗？**
A：会。Actor 和 Critic 的参数在同一个计算图中，backward 向两者同时传播梯度。然后 Actor 和 Critic 各自的优化器分别调用 `step()`。

### 第 5 层：旧策略同步与训练管理（L222-227）

**Q：为什么 PPO 需要维护一个 Old Actor 模型？GRPO 为什么不需要？**

A：核心原因是**旧策略快照**的需求不同。

PPO 的 Actor 每步都在更新，但旧策略需要在 K 步内保持固定（延迟策略更新）。如果只用一个 Actor，它每步都变，重要性采样的分母 `π_old` 就丢了。所以 PPO 维护一个 Old Actor 作为"评价时的固定参考点"。Old Actor 每 K 步同步一次，在这 K 步内它保持冻结，而 Actor 持续更新——更新幅度通过 `ratio = exp(actor_logp - old_logp)` 来度量。

GRPO 没有这个需求——它每步重新采样，且用 `detach()` 技巧让同一份 logp 兼任分子和分母，ratio 恒为 1。

类比——**手机相机的实时滤镜 vs 拍立得**：
```
PPO 的做法：
  先拍一张（采样）→ 存一张原图（Old Actor 快照）
  然后调滤镜、加特效（Actor 更新）
  每次都用原图来比较"调了多少"（ratio）
  每调 K 次就重拍一张原图换上（同步 Old Actor）

GRPO 的做法：
  没有原图比较——每次直接拍新照片、在新的基础上再调
```

**Q：`update_old_actor_freq` 增大/减小分别有什么效果？**

A：`update_old_actor_freq` 控制"多久拍一次新照片"：

| 频率 | 含义 | 效果 |
|:----:|------|------|
| 1（每步） | 每步都同步 | ≈ online 学习，Old Actor ≈ Actor，ratio≈1，裁剪几乎不触发 |
| 4（默认） | 每 4 步同步 | ratio 在 4 步内累积变化，裁剪偶尔触发 |
| 10 | 每 10 步同步 | ratio 偏离更远，裁剪频繁触发，可能过度裁剪导致梯度消失 |

直觉类比——**GPS 导航更新地图**：
```
每步更新：每秒钟刷新路况 → 信息最新，但浪费 CPU（接近 online）
每 4 步更新：每 2 分钟刷新一次 → 路况稍有滞后，但够用 ✓
每 100 步更新：一天刷新一次 → 路况严重过时，导航会导到堵车路段
```

**Q：`load_state_dict()` 是完整替换还是部分替换？**

A：**完整替换**。`old_actor_model.load_state_dict(actor_model.state_dict())` 把 Actor 的所有权重完整复制到 Old Actor。这意味着 Old Actor 和 Actor 在同步瞬间是完全相同的模型。

**Q：同步后 ratio 的变化曲线是什么样的？**

A：画出来是一条"锯齿波"：

```
ratio
  ↑
 1.10 ┤        ╱╲                ╱╲
 1.05 ┤       ╱  ╲              ╱  ╲
 1.00 ┤──────╱────╲────────────╱────╲──
 0.95 ┤     ╱      ╲          ╱      ╲
 0.90 ┤    ╱        ╲        ╱        ╲
      └───┴──────────┴───────┴──────────┴──→ step
          同步        同步       同步
         ratio=1.0   ratio=1.0  ratio=1.0
```

同步瞬间 ratio=1.0（因为 `old_logp = actor_logp`），然后随着 K 步内 Actor 持续更新、Old Actor 保持不变，ratio 逐渐偏离 1。到下一次同步时，Old Actor 跳跃式追上 Actor，ratio 瞬间跳回 1.0。

**Q：为什么 `old_actor_model` 不参与 DDP 包装？**

A：DDP 的作用是在多 GPU 间同步**梯度**。`old_actor_model` 只在 `no_grad` 下做前向推理，不参与梯度计算和反向传播，所以不需要 DDP 包装。

类比——**操场跑步**：
```
Actor 和 Critic 是参赛选手（需要同步配速/梯度）→ 用 DDP 包装
Old Actor 和 Ref 是场边拿秒表的计时员（只记录，不跑步）→ 不需要 DDP
```

**Q：`freqs_cos/freqs_sin` 为什么被排除 DDP 同步？**

A：这些是 RoPE 位置编码用到的**预计算缓存表**（cos/sin 值），属于模型的 buffer 而不是 parameter。

- 没有梯度：它们不需要反向传播，所以也不需要 DDP 的梯度同步
- 每个 rank 内容相同：所有 GPU 上算出来的 cos/sin 表完全一样，多此一举去同步
- 节省通信带宽：模型很大时，排除这些 buffer 可以减少 DDP 初始化时的广播开销

代码里这样写：
```python
actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
```
这行代码告诉 DDP："这两个键是本地 buffer，不需要广播到其他 rank。"

### 第 6 层：主函数与模型初始化（L252-381）

**Q：Actor 与 Critic 为什么有独立的优化器？**
A：两者参数不同（Actor 更新 lm_head + 共享主体，Critic 更新 value_head + 共享主体），可能需要不同的学习率或优化策略。

**Q：PPO Actor 的 LR（1e-6）是 GRPO（8e-8）的 12.5 倍，为什么 PPO 敢用这么大的学习率？**
A：PPO 有裁剪机制提供第二道防线——即使一步更新很大，`min(surr1, surr2)` 也会将有效更新幅度限制在 `[1-ε, 1+ε]` 内。GRPO 没有裁剪，必须用极小 LR 维持每步稳定性。类比：PPO 有限速器可以深踩油门，GRPO 没有限速器只能轻踩。

**Q：`torch.compile` 为什么只对 Actor 启用，不对 Critic 启用？**
A：Critic 是一个简单的前向（Transformer 主体 + 单层 Linear），编译加速收益不大，还可能增加显存开销。

**Q：Critic 的 `value_head` 随机初始化，训练初期怎么办？**
A：输入是预训练好的 hidden states（已有语义特征），监督信号是一个标量，学习率 1e-6 足够大，几步之内就能收敛——详见 2.2 节的分析。

---

## 六、自测题

### 基础题

1. PPO 的训练流程分为哪几个主要步骤？用自己的话描述。

2. PPO 需要哪 5 个模型？各自的角色和可训练状态是什么？与 GRPO 的 3 模型架构对比。

3. `CriticModel` 的设计思路是什么？它和 Actor 模型共享了什么部分？独有了什么部分？

**参考答案**

> **第 1 题**：PPO 每个训练 step 的主要步骤：
>
> 1. **分词** — 将 prompt 列表分词 padding 到等长
> 2. **生成回复** — 用 Actor 模型的 `generate(do_sample=True)` 给每个 prompt 生成 1 个回复（与 GRPO 的 8 个不同）
> 3. **奖励计算** — 用 Reward Model + 格式奖励给每个回复打分
> 4. **Critic 前向 + 优势计算** — 在完整序列上跑 Critic，取最后一个 token 的值作为 `V(s)`，优势 `A = R - V(s)`
> 5. **Actor 前向（当前策略）** — 在生成序列上算当前 Actor 的 `actor_logp`
> 6. **Old Actor / Ref 前向（固定基线）** — `no_grad` 下算 `old_actor_logp` 和 `ref_logp`
> 7. **Loss 计算** — 三项组合：裁剪策略梯度 + 价值 MSE + KL 惩罚
> 8. **反向传播** — 梯度同时流向 Actor 和 Critic
> 9. **梯度累积 + 优化器步进** — 每 `accumulation_steps` 步更新一次参数
> 10. **旧策略同步** — 每 `update_old_actor_freq`（默认 4）步将 Actor 权重同步到 Old Actor
> 11. **日志 + 保存 checkpoint**
>
> **第 2 题**：5 个模型：
>
> | # | 模型 | 可训练 | 角色 | 特点 |
> |:-:|------|:-----:|------|------|
> | ① | **Actor** | ✅ | 生成回复 + 策略梯度更新 | 主模型 |
> | ② | **Old Actor** | ❌ | 提供重要性采样分母 `π_old` | 每 K 步同步一次 |
> | ③ | **Ref** | ❌ | KL 散度的基线 | 全程冻结，永不同步 |
> | ④ | **Critic** | ✅ | 估计状态价值 `V(s)` | 共享 Actor 主体 + value_head |
> | ⑤ | **Reward** | ❌ | 给回复打分 | 外部加载，结构可不同 |
>
> 与 GRPO 的 3 模型（policy + ref + reward）对比：PPO **多了一个 Critic 和一个 Old Actor**。GRPO 用组内多采样的均值替代了 Critic 的 `V(s)`，用 `detach()` 替代了显式的 Old Actor。
>
> **第 3 题**：CriticModel **继承自 `MiniMindForCausalLM`**，共享完整的 Transformer 主体。独有的是用 `value_head`（`Linear(hidden_size, 1)`）替换了 Actor 的 `lm_head`（`Linear(hidden_size, vocab_size)`），输出从 `[B, T, vocab_size]` 的分类分布变成了 `[B, T]` 的标量价值序列。初始化时 `strict=False`，所以 `value_head` 因为 checkpoint 中找不到对应权重而**随机初始化**。

### 进阶题

4. PPO 的 `A = R - V(s)` 与 GRPO 的 `A = (R - μ)/σ` 在功能上是等价的吗？
   各有什么优劣？（提示：Critic 的训练成本 vs 组内多采样的成本）

5. PPO 的裁剪机制 `min(surr1, surr2)` 在什么情况下会触发？
   如果 `clip_epsilon=0` 会发生什么？如果 `clip_epsilon=∞` 呢？

6. PPO 有 `old_actor_model` 并每 K 步同步一次，而 GRPO 没有。
   这会导致训练行为上的什么差异？

**参考答案**

> **第 4 题**：两者在功能上是**等价的**——都是提供一个基线（baseline）来降低策略梯度的方差，让模型知道"相对好坏"而不是"绝对好坏"。
>
> | 维度 | PPO `A = R - V(s)` | GRPO `A = (R - μ)/σ` |
> |------|-------------------|----------------------|
> | 需要额外训练 | ✅ Critic 模型，每次前向+反向 | ❌ 无需训练，直接算统计量 |
> | 每步生成数 | 1 个 / prompt | 多个（默认 8）/ prompt |
> | 基线质量 | 依赖 Critic 训得好不好（可能不收敛） | 采样越多越接近真实期望 |
> | 奖励信号 | 稀疏奖励时 Critic 难学 | 组内对比天然有效 |
>
> **PPO 的优势场景**：生成成本昂贵（如大模型推理费时），不愿为每个 prompt 生成多个回复。这时宁可训练一个 Critic 来替代多次采样。
>
> **GRPO 的优势场景**：奖励稀疏（如数学题只有对/错），Critic 很难学到中间状态的价值。这时宁可多采几个样做组内对比。
>
> **第 5 题**：裁剪在 ratio 超出 `[1-ε, 1+ε]` 时触发：
>
> | 情况 | 触发条件 | 后果 |
> |------|---------|------|
> | `ratio > 1+ε` | Actor 对这个 token 概率涨了 >10% | surr2(1.1) 被选 → 限制更新幅度 |
> | `ratio < 1-ε` | Actor 对这个 token 概率跌了 >10% | surr2(0.9) 被选 → 限制更新幅度 |
> | ratio 在界内 | 变化在 ±10% 以内 | 不触发裁剪，正常更新 |
>
> - **`clip_epsilon=0`**：`clamp(ratio, 1.0, 1.0)` 把 ratio 强行钉死在 1.0，`surr2 = 1.0 × adv = adv`（常数）。此时 min 的选取规律：
>
> | | adv | ratio 位置 | 对模型来说 | min 选谁 | 梯度？ |
> |:-:|:---:|:---------:|:----------:|:--------:|:-----:|
> | 好回答 | +0.5 | **1.3**（涨了，对的 direction） | ✅ 要继续保持 | surr2（常数 0.50） | ❌ 截断 |
> | | +0.5 | 0.8（跌了，错的 direction） | ❌ 需要回调 | surr1（0.40） | ✅ 放行 |
> | 差回答 | -0.5 | **0.7**（跌了，对的 direction） | ✅ 要继续保持 | surr2（常数 -0.50） | ❌ 截断 |
> | | -0.5 | 1.3（涨了，错的 direction） | ❌ 需要回调 | surr1（-0.65） | ✅ 放行 |
>
> 规律：min 永远截断"对的 direction"（ratio 偏离 1 的方向），放行"错的 direction"（ratio 回 1 的方向）。结果 ratio 只能回 1 不能离 1，涨不上去也跌不下来——**模型学不动**。
>
> - **`clip_epsilon=∞`**：`clamp(ratio, -∞, +∞)` 是恒等映射，`surr2 = ratio × adv = surr1`。`min(surr1, surr2) = surr1`，裁剪消失，退化为**无裁剪的原始策略梯度**（REINFORCE）。
>
> **第 6 题**：核心差异在于 **ratio 能否偏离 1**：
>
> | | PPO | GRPO |
> |---|---|---|
> | ratio 来源 | `exp(actor_logp - old_logp)`，old 每 K 步冻结 | `exp(x - x.detach())`，恒为 1 |
> | ratio 是否偏离 1 | ✅ 会，随着 Actor 在 K 步内不断更新而偏离 | ❌ 不，每步重新采样 |
> | 裁剪是否触发 | ✅ 频繁触发，因为 ratio 会偏离 | ❌ 不触发 |
>
> **PPO 的优点**：可以**多 epoch 复用数据**。在同一次采样的数据上跑多个 epoch，因为 Old Actor 冻结着，ratio 能反映多次 epoch 的累积变化。GRPO 做不到，每步必须重新采样。
>
> **PPO 的代价**：需要多维护一个 Old Actor 模型 + 同步逻辑。如果 K 设得太大，ratio 可能偏离过远，裁剪频繁触发导致梯度消失。

### 深入题

7. PPO 的 Loss 由三部分组成（策略梯度 + 价值 Loss + KL 惩罚），
   GRPO 只有两部分（策略梯度 + KL 惩罚）。多出来的价值 Loss 对训练有什么影响？
   如果 `vf_coef=0` 会怎样？

8. PPO 的 `kl_ref = (actor_logp - ref_logp).mean()` 计算的是**序列平均**的 KL，
   而 GRPO 的 `per_token_kl = exp(kl_div) - kl_div - 1` 计算的是**逐 token** 的 KL。
   这两种计算方式会导致什么差异？

9. PPO 中 Actor 的 LR=1e-6 而 GRPO 的 LR=8e-8，PPO 的步长是 GRPO 的 12.5 倍。
   为什么 PPO 敢用更大的学习率？（提示：与裁剪机制有关）

**参考答案**

> **第 7 题**：价值 Loss 的作用是训练 Critic 模型学会预测 `V(s)`。Critic 预测越准，`A = R - V(s)` 中的噪音越小，策略梯度信号质量越高。
>
> 如果 `vf_coef=0`：
> - Critic 不再通过 MSE 学习，`V(s)` 永远停留在随机初始化的水平
> - 优势 `A = R - V(s)` 中的 `V(s)` 是随机的，等价于 `A = R + noise`
> - 这等于给奖励加了大量噪音，策略梯度方向受污染，训练可能发散
> - 即便共享主体通过 Actor 的梯度在更新，`value_head` 本身不更新，永远输出随机值
>
> 所以 `vf_coef` 不能设为 0（除非你有其他方式提供价值估计）。PPO 多出的这个超参数需要调节：太大 → Critic 过拟合当前 batch 的噪音；太小 → Critic 跟不上 Actor 的变化。
>
> **第 8 题**：两种 KL 计算方式的差异：
>
> | | PPO（序列级） | GRPO（token 级） |
> |---|---|---|
> | 先求和再平均 | `actor_logp` 是先对回复所有 token 的 log_prob 求和，再 batch 平均 | 不求和，每个 token 独立算 KL |
> | 细粒度 | 粗——偏离大的 token 和偏离小的 token 混在一起平均 | 细——偏离大的 token 受到更大的逐 token 惩罚 |
> | 为什么各自够用 | PPO 有裁剪 + Critic 双重稳定，不需要细粒度 KL | GRPO 没有裁剪，依赖细粒度 KL 做逐 token 约束 |
>
> 举例：一个 10 个 token 的回答，其中 1 个 token 的偏离巨大（kl_div=5），其余 9 个完美对齐（kl_div≈0）。
> - PPO 的序列级 KL：`(5 + 0×9)/10 = 0.5` → 偏离被平均稀释了
> - GRPO 的 token 级 KL：`per_token_kl` 矩阵中那个偏离大的 token 单独受罚，不会被稀释
>
> **第 9 题**：PPO 之所以敢用 12.5 倍大的学习率，核心原因就是**裁剪机制**：
>
> ```
> PPO 的更新链条：
>   大 LR → Actor 大步更新 → ratio 可能偏离很大
>   → clip 在 |ratio - 1| > ε 时触发
>   → min(surr1, surr2) 选到裁剪后的版本
>   → 梯度被截断，单步有效更新幅度被限制
>
> 所以：LR 大 ≠ 有效更新幅度大，clip 是第二道防线
>
> GRPO 的更新链条：
>   小 LR → Actor 小步更新 → 没有 clip 保护
>   → 每步必须谨慎
> ```
>
> 这就像一个油门（LR）和限速器（clip）的关系：
> - PPO 的油门可以踩很深（LR 大），但限速器会把速度限制在安全范围
> - GRPO 没有限速器，所以油门只能轻轻踩（LR 小）

---

## 七、关联文件

```
train_ppo.py
 ├─ scripts/Model/model_minimind.py               ← Actor、Old Actor、Ref 和 Critic 共享的模型定义
 ├─ dataset/lm_dataset.py                 ← RLAIFDataset（与 GRPO 相同的数据集）
 ├─ scripts/Trainer/trainer_utils.py              ← 工具函数（init_model, SkipBatchSampler 等）
 ├─ scripts/Trainer/train_grpo.py                 ← 对比学习：GRPO 的无 Critic 实现
 └─ plan/train_grpo_study_plan.md         ← 前置学习：GRPO 学习计划
```
