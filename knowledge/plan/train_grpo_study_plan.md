# train_grpo.py 学习计划指引

## 一、文件定位

`train_grpo.py` 是 MiniMind 项目**组相对策略优化**（Group Relative Policy Optimization）训练脚本，
也是完整训练管线的**最后阶段**。在 reasoning 训练让模型学会推理格式的基础上，
GRPO 通过**在线采样 + 组内相对奖励**进一步优化推理质量。

```
train_pretrain.py（预训练 ✅）
    ↓ 产出 pretrain.pth
train_full_sft.py（指令微调 ✅）
    ↓ 产出 full_sft.pth
train_lora.py（LoRA 高效微调 ✅）
    ↓ 产出 lora_xxx.pth
train_dpo.py（偏好对齐 ✅）
    ↓ 产出 dpo.pth
train_reason.py（推理能力训练 ✅）
    ↓ 产出 reason.pth
train_grpo.py（GRPO ← 你在这里）
    ↓ 产出 grpo.pth
```

### 为什么需要 GRPO？

DPO 和 Reason 都是**静态数据**训练：数据是预先准备好的，模型只是"模仿"或"加权学习"。
GRPO 让模型**自己生成多个回答**，用 Reward Model 打分后再从组内比较中学习：
- 好的回答（高于组内平均）→ 增大概率
- 差的回答（低于组内平均）→ 减小概率

这种"自己探索 + 外部评价"的方式是 RL（强化学习）的核心思想。

| 阶段 | 数据来源 | 学习方式 | 目标 |
|------|---------|---------|------|
| SFT | 静态指令数据 | 模仿 | 学会对话格式 |
| DPO | 静态偏好对 | 对比 | 偏好好回答 |
| Reason | 静态推理数据 | 加权模仿 | 学会推理格式 |
| **GRPO** | **模型在线生成** | **探索+奖励** | **优化推理质量** |

---

## 二、与新学概念（前置知识：DPO study plan §2.6 已详细对比了 PPO / GRPO / DPO）

建议先复习 `plan/train_dpo_study_plan.md` 第 2.6 节，那里用大量篇幅和图表对比了三个方法的完整流程。

### 2.1 GRPO 核心流程（4 步）

GRPO 的每个训练 step 分 4 步：

**Step 1 — 在线采样**
```
对每个 prompt，让 policy 生成 num_generations（默认 8）个回答
比如 prompt="1+1=?" → 生成 ["2","2","3","2","1","2","4","2"]
```
代码 L107-109：
```python
outputs = model_for_gen.generate(
    **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
    num_return_sequences=args.num_generations, ...)
```

**Step 2 — 奖励计算（calculate_rewards）**
```
对每个回答计算总奖励 = 格式奖励 + Reward Model 评分
```
详见 2.3 节。

**Step 3 — 组内相对优势**
```
grouped_rewards = rewards.view(B, num_gen)        # 按 prompt 分组
mean_r = grouped_rewards.mean(dim=1)               # 组内均值
std_r = grouped_rewards.std(dim=1)                 # 组内标准差
advantages = (rewards - mean_r) / (std_r + 1e-4)   # 相对优势
```
代码 L133-137。

**Step 4 — 策略梯度更新**
```
用 advantage 加权更新 policy：好的回答（advantage > 0）→ 增大概率，反之减小
同时用 KL 散度惩罚防止 policy 偏离 ref 太远
```
详见 2.4 节。

### 2.2 三模型架构

GRPO 需要 **3 个模型**（train_grpo.py:252-266）：

```python
# ① Policy 模型 — 要训练的模型，从 reason/full_sft 权重初始化
model, tokenizer = init_model(lm_config, base_weight, device=args.device)

# ② Reference 模型 — Policy 的冻结副本，用于 KL 散度约束
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)

# ③ Reward 模型 — 外部评分模型，独立加载，完全冻结
reward_model = AutoModel.from_pretrained(args.reward_model_path, ...)
reward_model = reward_model.to(args.device).eval().requires_grad_(False)
reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, ...)
```

| 模型 | 可训练 | 作用 | 初始权重来源 |
|------|-------|------|------------|
| Policy | 是 | 生成回答 + 策略更新 | `reason`（推理模型）或 `full_sft` |
| Ref | 否 | KL 散度计算基线 | 与 Policy 相同 |
| Reward | 否 | 给生成结果打分 | 独立路径（`--reward_model_path`） |

注意与 DPO 的区别：
- DPO：policy 和 ref **初始权重完全相同**
- GRPO：policy 和 ref **初始权重也相同**，但 ref 全程冻结，reward 从**外部独立加载**

### 2.3 奖励函数设计（calculate_rewards, L27-92）

GRPO 的奖励由**两部分**组成：

**① 格式奖励（reasoning_model_reward, L29-53）**
```
完整格式匹配（<think>...</think><answer>...</answer>）：+0.5
每个特殊标记单独计数（<think>/</think>/<answer>/</answer> 各出现 1 次）：各 +0.25
格式奖励总分范围：[0, 1.5]
```
这部分**只有 `--reasoning=1` 时才启用**，确保模型坚持推理格式。

**② Reward Model 评分（L59-90）**
```python
score = reward_model.get_score(reward_tokenizer, tmp_chat)
score = max(min(score, scale), -scale)   # 裁剪到 [-3, 3]
```
如果启用了 reasoning，还会**拆分回答**：
```python
# 完整回答（含推理过程）打分占 40%
# 提取 <answer> 部分单独打分占 60%
score = score * 0.4 + answer_score * 0.6
```
这意味着：**推理过程和最终答案都会被评判，但最终答案的权重更高**。

### 2.4 策略梯度与 KL 散度（L113-149）

**① 计算 per_token_logps（L113-120）**

与 DPO 的 `logits_to_log_probs` 类似，用 `log_softmax` + `gather` 取出每个 token 的 log_prob。
但比 DPO 多一个 trick：只保留生成部分（`completion_ids`）的 log_prob，不计算 prompt 部分。

**② 分别计算 policy 和 ref 的 logps**
```python
per_token_logps = get_per_token_logps(model, outputs, n_keep)    # [B*num_gen, R]
ref_per_token_logps = get_per_token_logps(ref_model, outputs, n_keep)  # [B*num_gen, R]
```

**③ 关键公式（L144-147）**
```python
kl_div = ref_per_token_logps - per_token_logps                    # ①
per_token_kl = torch.exp(kl_div) - kl_div - 1                     # ②
per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach())
                   * advantages.unsqueeze(1)
                   - args.beta * per_token_kl)                    # ③
policy_loss = ((per_token_loss * completion_mask).sum(dim=1)
               / completion_mask.sum(dim=1)).mean()              # ④
```

逐行拆解：

| 行 | 公式 | 含义 |
|----|------|------|
| ① | `kl_div = ref_logps - policy_logps` | Policy 比 ref **高/低**了多少（正值 = policy 概率更高） |
| ② | `per_token_kl = exp(kl_div) - kl_div - 1` | KL 散度的近似形式，kl_div=0 时值为 0，偏离越大惩罚越大 |
| ③ | `exp(policy_logps - sg(policy_logps)) × adv - β × kl` | 第一项数值为 `1× adv`（detach 防梯度抵消，见第 6 题详解）；第二项是 KL 惩罚 |
| ④ | `mean over sequence then over batch` | 先对序列平均（去除 padding），再对 batch 平均 |

其中 `per_token_logps - per_token_logps.detach()` 是**停止梯度**技巧：两者数值相同（来自同一次前向），因此 `exp(0)=1` 恒成立；但 `detach()` 截断了右半的梯度链，防止左右梯度相互抵消（否则 `d/dθ(x-x)=0`），使 advantage 信号能顺利流向 policy 参数。

**④ EOS 位置标记（L139-142）**
```python
is_eos = completion_ids == tokenizer.eos_token_id         # 找到 EOS
eos_idx = is_eos.int().argmax(dim=1)                      # 每个序列第一个 EOS 位置
completion_mask = (arange <= eos_idx.unsqueeze(1)).int()   # EOS 之后的位置 mask 掉
```
这确保 EOS 之后的 token 不参与 loss 计算、不贡献梯度。

### 2.5 数据准备

GRPO 使用 `RLAIFDataset`（dataset/lm_dataset.py:242-276），返回结构：
```python
{'prompt': "对话历史（不含assistant回复）", 'answer': "标准答案（用于参考）"}
```

与 SFTDataset/DPODataset 的关键区别：
- 不需要 labels，因为 GRPO 是**在线生成**，label 来自模型自身的输出
- `prompt` 是原始字符串（`apply_chat_template` 后的纯文本），由 GRPO 训练循环自行 tokenize
- 使用 `RLAIFDataset.create_chat_prompt` 时 `add_generation_prompt=True`（让模型知道该生成了）

### 2.6 学习率与优化器

```python
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)    # 默认 8e-8
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr/10)
```

对比全流程 LR 演进：
```
Pretrain:  5e-4  (从零训练，大步长)
SFT:       1e-6  (微调对话能力)
DPO:       4e-8  (微调偏好，极小)
Reason:    1e-6  (学习推理格式)
GRPO:      8e-8  (在线优化，需稳定)
```

GRPO 的 LR=8e-8 介于 DPO 和 Reason 之间：
- 需要比 DPO 大一点，因为 policy 要"探索"新的生成策略
- 不能太大，否则在线采样不稳定，reward 波动大

### 2.7 与 DPO 的核心差异

| 维度 | DPO | GRPO |
|------|-----|------|
| 数据来源 | 静态偏好对（chosen/rejected） | **模型在线生成**（`model.generate()`） |
| 模型数量 | 2 个（policy + ref） | **3 个**（policy + ref + reward） |
| 奖励来源 | 偏好对直接提供"好坏" | **Reward Model 打分** |
| 优势计算 | chosen vs rejected 直接对比 | **组内相对优势**（同 prompt 内比较） |
| 探索能力 | 无（被动学习已有数据） | **有**（`do_sample=True` 生成多样回答） |
| Loss 类型 | 对比 loss（-log σ） | **策略梯度**（advantage × log_prob） |
| KL 约束 | 隐含在 loss 公式中 | **显式**（`β × per_token_kl` 惩罚项） |
| 学习率 | 4e-8 | 8e-8 |
| 训练开销 | 低（纯前向） | **高**（每步生成 8 个回答） |

---

## 三、关键公式一览

### 格式奖励
```
R_format = match_bonus(0.5) + count_bonus(max 1.0)
         = 0.5 + (think✓ + /think✓ + answer✓ + /answer✓) × 0.25
         ∈ [0, 1.5]
```

### 总奖励
```
R_total = R_format + clip(R_reward_model, -3, 3)      # 非 reasoning 模型
R_total = R_format + 0.4×R_full + 0.6×R_answer        # reasoning 模型（拆分评分）
```

### 组内相对优势
```
μ_group = mean(R_group)
σ_group = std(R_group)
A = clamp((R - μ_group) / (σ_group + ε), -10, 10)    # 组内归一化
A = (A - mean(A)) / std(A)                            # 二次归一化
```

### 策略梯度 Loss
```
L_policy = -E[ exp(π/sg(π)) × A - β × (exp(Δ) - Δ - 1) ]
           数值恒为 1×A（detach 防梯度抵消，见第 6 题）  KL 散度惩罚
```

---

## 四、学习目标检查清单

- [ ] 理解 GRPO 的完整流程（采样 → 奖励 → 优势 → 更新）
- [ ] 理解在线采样与静态数据训练的本质区别
- [ ] 理解 3 模型架构（policy + ref + reward）各司其职
- [ ] 理解格式奖励 + Reward Model 评分的双层奖励设计
- [ ] 理解组内相对优势的计算过程和意义
- [ ] 理解 KL 惩罚项 `exp(kl_div) - kl_div - 1` 的作用
- [ ] 理解 `per_token_logps - per_token_logps.detach()` 停止梯度的 trick
- [ ] 理解 EOS mask 的作用
- [ ] 对比 GRPO 与 DPO 的差异（数据、模型、loss、开销）
- [ ] 理解 GRPO 在完整训练管线中的位置和作用

---

## 五、文件逐段精读计划

### 第 1 层：导入与参数定义

- `from dataset.lm_dataset import RLAIFDataset` → 仅返回 prompt 字符串，无 labels
- `--num_generations`（默认 8）：每个 prompt 生成的回答数，GRPO 的核心超参
- `--beta`（默认 0.02）：KL 惩罚系数
- `--reasoning`（默认 1）：是否启用推理格式奖励
- `--max_gen_len`（默认 1536）：回答最大生成长度
- `--learning_rate`（默认 8e-8）：策略梯度学习率
- `--reward_model_path`：独立 Reward Model 的路径

### 第 2 层：calculate_rewards（L27-92）

- `reasoning_model_reward`：正则匹配完整格式 + 逐个计数奖励
- Reward Model 的 `get_score` 调用，裁剪到 [-3, 3]
- reasoning 模式下回答拆分：完整回答 40% + answer 部分 60%

### 第 3 层：grpo_train_epoch 的在线采样（L95-111）

- `model.generate(..., num_return_sequences=args.num_generations)`
- `completion_ids = outputs[:, prompt_len:]` 截取生成部分
- 模型调用需要 DDP 拆包（`model.module`）

### 第 4 层：get_per_token_logps（L113-120）

- 用 `log_softmax` + `gather` 提取逐 token log_prob
- `logits_to_keep` 参数控制只计算最后 N 个位置（生成部分）的 logits

### 第 5 层：优势计算与策略梯度（L127-149）

- ref 和 policy 各算一次 per_token_logps
- 组内归一化 + 二次归一化
- KL 惩罚 + `exp(π/sg(π))×A`（数值恒为 1×A，detach 防梯度抵消，详见第 6 题）
- completion_mask 只保留 EOS 之前的 token

---

## 六、自测题

### 基础题

1. GRPO 的每个训练 step 分为哪 4 步？用自己的话描述每一步在做什么。

2. GRPO 需要哪 3 个模型？各自的角色和可训练状态是什么？

3. `num_generations=8` 的含义是什么？增大/减小这个值会有什么影响？

**参考答案**

> **第 1 题**：GRPO 每步的 4 个阶段：
>
> 1. **在线采样** — 对 batch 中的每个 prompt，policy 模型通过 `model.generate(do_sample=True)` 生成 `num_generations`（默认 8）个回答。这里的关键是 `do_sample=True`，让采样有随机性，产生多样化的回答供后续比较。
> 2. **奖励计算** — 对每个生成的回答，调用 `calculate_rewards` 计算总奖励。奖励 = 格式奖励（匹配 <think>/<answer> 格式）+ Reward Model 评分。如果是 reasoning 模式，还会将完整回答和 answer 部分按 4:6 加权。
> 3. **组内相对优势** — 将同一个 prompt 的 `num_generations` 个回答的奖励值放在一起，计算组内均值 μ 和标准差 σ，再算出每个回答的相对优势 `A = (R - μ) / (σ + ε)`。这回答了"在这个 prompt 的所有回答中，这个回答相对是好是坏"。
> 4. **策略梯度更新** — 用 advantage 加权更新 policy 参数。advantage > 0 的回答概率被推高，< 0 的被压低。同时用 KL 散度惩罚阻止 policy 偏离 reference 模型太远。
>
> **第 2 题**：三个模型分别为：
>
> | 模型 | 可训练 | 角色 |
> |------|-------|------|
> | **Policy** (model) | ✅ 是 | 生成回答 + 接受梯度更新，是唯一参数变化的模型 |
> | **Reference** (ref_model) | ❌ 否 | Policy 的冻结副本，提供 KL 散度的基线，防止 policy 崩坏 |
> | **Reward** (reward_model) | ❌ 否 | 外部评分模型，给生成回答打质量分，完全独立加载 |
>
> Policy 和 Ref 初始权重相同（都来自 `reason.pth` 或 `full_sft.pth`），但 Ref 全程冻结不更新。Reward 从 `--reward_model_path` 独立加载，与 Policy/Ref 结构可以完全不同。
>
> **第 3 题**：`num_generations=8` 表示**每个 prompt 同时生成 8 个回答**。这些回答构成一个"组"，用于计算相对优势。
>
> - **增大**（如 16、32）→ 组内统计更稳定，优势估计方差更小；但显存和计算开销线性增长（每个回答都要跑完整前向+奖励计算）。
> - **减小**（如 4、2）→ 开销降低，但组内样本太少时均值和标准差的估计不可靠，优势值的噪音增大，训练可能不稳定。
> - 极端情况 `num_generations=1` → 组内只有一个样本，标准差为 0，优势恒为 0，梯度更新失效，GRPO 退化为 random walk。

### 进阶题

4. 组内相对优势 `(rewards - mean_r) / (std_r + 1e-4)` 解决了什么问题？
   为什么要对同一个 prompt 的多个回答计算相对优势，而不是直接用 Reward Model 的绝对分数？

5. 对比 `exp(kl_div) - kl_div - 1` 在 kl_div=0、kl_div=0.5、kl_div=1、kl_div=-0.5 时的值。
   这个惩罚函数有什么性质？（提示：对称性，最小值在何处）

6. `per_token_logps - per_token_logps.detach()` 中 `detach()` 的作用是什么？
   如果不加 detach 会怎样？

**参考答案**

> **第 4 题**：组内相对优势解决的是 **Reward Model 评分不可跨 prompt 比较**的问题。
>
> 不同 prompt 的难度不同：简单问题的 Reward Model 评分可能普遍偏高，难题普遍偏低。如果直接用绝对分数，"简单题的高分回答"会被错误地放大，"难题的低分回答"会被错误地打压。组内相对优势**只关心"相对于同一个 prompt 的其他回答，这个回答怎么样"**，消除了 prompt 之间的难度差异。
>
> 举个例子：
> - Prompt A（"1+1=?"）：生成 8 个回答，Reward 评分 [2.9, 2.8, 2.9, 2.7, 2.8, 2.9, 2.8, 2.9] → 绝对分都很高，但优势接近 0（大家都差不多好）
> - Prompt B（"证明黎曼猜想"）：生成 8 个回答，Reward 评分 [-2.0, 0.5, -1.5, 0.8, -1.0, 0.3, -2.5, 1.2] → 绝对分很低，但优势区分了"相对较好"的回答（0.8、1.2）和"相对较差"的回答（-2.5、-2.0）
>
> 此外，GRPO 还做了**二次归一化**（`A = (A - mean(A)) / std(A)`），确保整个 batch 的优势值均值为 0、方差为 1，进一步稳定训练。
>
> **第 5 题**：计算 `f(x) = exp(x) - x - 1`：
>
> | x (kl_div) | f(x) | 含义 |
> |-----------|------|------|
> | 0 | exp(0) - 0 - 1 = 0 | policy 和 ref 完全相同，惩罚为 0 |
> | 0.5 | e⁰·⁵ - 0.5 - 1 ≈ 1.648 - 1.5 = 0.148 | policy 比 ref 概率高 0.5，轻微惩罚 |
> | 1.0 | e¹ - 1 - 1 ≈ 2.718 - 2 = 0.718 | policy 比 ref 概率高 1.0，较大惩罚 |
> | -0.5 | e⁻⁰·⁵ + 0.5 - 1 ≈ 0.606 + 0.5 - 1 = 0.106 | policy 比 ref 概率低 0.5，也有惩罚 |
>
> **性质**：
> 1. **最小值在 x=0 处，f(0)=0** — policy 与 ref 无差异时无惩罚。
> 2. **非对称正函数**：f(x) ≥ 0 恒成立，且仅当 x=0 时取等。这保证了 KL 惩罚始终是"扣分"项，不会变成"加分"。
> 3. **对正偏离惩罚比负偏离更重**：f(1)=0.718 > f(-1)=0.367（因为 `exp(x) - x - 1` 对 x>0 增长更快）。这意味着 policy **盲目提高概率比降低概率受到的惩罚更大**，鼓励谨慎探索。
>
> 这种非对称性设计是故意的：policy 为了提高 reward 倾向于把某些 token 概率推得很高（exploitation），而 KL 惩罚确保它不会偏离初始化的 ref 太远。
>
> **第 6 题**：先看**实际代码**（`train_grpo.py` L543-546）：
>
> ```python
> # 在这里权重 π_θ/π_old 被简化为 exp(per_token_logps - per_token_logps.detach())
> per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach())
>                    * advantages.unsqueeze(1) - args.beta * per_token_kl)
> ```
>
> **关键事实：`per_token_logps` 和 `per_token_logps.detach()` 来自同一次前向传播**（L505），所以：
> ```
> per_token_logps - per_token_logps.detach() = 0 （每个元素都是 0）
> torch.exp(0) = 1                            （恒为 1）
> ```
>
> 那么这行代码写的 `exp(Δ) × adv` 实际上就是 `1 × adv`。**所以 ratio 恒为 1，不是真正的"重要性采样权重"。**
>
> ### 那 `.detach()` 到底在做什么？
>
> 关键在**梯度计算**而非数值本身：
>
> | 写法 | 数值 | 梯度 `d(loss)/d(x)` | 效果 |
> |------|------|-------------------|------|
> | `exp(x - x)` | 1 | 0 ❌ | 梯度抵消，policy 学不到 |
> | **`exp(x - x.detach())`** | **1** | **adv** ✅ | 梯度正确流通 |
>
> 数学原因：`x` 对 `x` 求导 = 1，`x` 对 `x` 求导也是 1，所以 `d/dx[x - x] = 1 - 1 = 0`。
> 而 `d/dx[x - x.detach()] = 1 - 0 = 1`（detach 把后半截的梯度截断了）。
>
> **`.detach()` 的作用：防止梯度在相减时自抵消。** 没有它，整个 `exp(x - x)` 梯度为 0，advantage 信号完全丢失。
>
> ### 那"旧策略"在哪里？
>
> 在**真实的 Reference 模型**（`train_grpo.py` L510-512）：
> ```python
> with torch.no_grad():                                  # ← 完全冻结
>     ref_per_token_logps = get_per_token_logps(ref_model, ...)
> ```
> 这才是真正的"旧策略基线"——**独立的 ref 模型**，初始与 policy 相同但全程不更新。`exp(per_token_logps - per_token_logps.detach())` 和重要性采样无关，真正约束 policy 的是 `ref_per_token_logps` 参与的 KL 惩罚项：
> ```python
> kl_div = ref_per_token_logps - per_token_logps          # 真实的策略偏离度
> per_token_kl = torch.exp(kl_div) - kl_div - 1           # 真实的 KL 惩罚
> ```
>
> ### 一句话总结
>
> - `exp(per_token_logps - per_token_logps.detach())` = **恒为 1**，不是重要性采样权重
> - `.detach()` 防止梯度自抵消，让 advantage 信号能通过梯度更新 policy
> - 真正防止 policy 跑偏的是 **`ref_model` + KL 惩罚项**（代码用 `torch.no_grad()` 冻结）
> - 源码注释自己也说"被**简化**为"——它就是简化版，不是标准 PPO 的重要性采样

### 深入题

7. GRPO 的 `calculate_rewards` 中，`reasoning_model_reward` 给出了最高 1.5 分的格式奖励，
   而 Reward Model 的评分范围是 [-3, 3]。如果格式奖励和模型评分冲突（格式完美但内容错误），
   训练会怎样收敛？

8. 对比 GRPO 和 DPO 的 KL 散度使用方式：
   - DPO 的 β 作用在 loss 公式的 logits 层面（隐式约束）
   - GRPO 的 β 作用在 per_token_kl 惩罚项（显式约束）
   各有什么优劣？如果 GRPO 去掉 KL 惩罚（β=0）会发生什么？

9. GRPO 加载的 `RLAIFDataset` 返回 `{'prompt': str, 'answer': str}`，
   训练过程中 `answer` 字段是否被使用？它的作用是什么？

**参考答案**

> **第 7 题**：在格式完美但内容错误的情况下，最终奖励 = 格式奖励(正) + Reward Model 评分(负)。训练会沿着**正负叠加的合力方向**收敛：
>
> - **短期**：格式奖励的正值会部分抵消 Reward Model 的负分，使得这种"格式好、内容差"的回答仍然有一定概率（不是直接被压到 0）。但如果 Reward Model 的负分绝对值 > 格式奖励，总的 advantage < 0，这类回答的概率仍会降低。
> - **长期**：在训练过程中，policy 会发现"仅格式好不够"——那些 content 也好、reward 更高的回答获得更大的正 advantage。因此模型会**同时学习两方面**：既要保持格式合规（否则格式奖励为负），又要提升内容质量（否则 Reward Model 打低分）。
> - **收敛结果**：理想情况下，模型学会在格式合规的前提下，尽可能提升内容质量。格式奖励起到了**约束搜索空间**的作用——它不让模型为了内容质量而丢弃格式（丢掉格式奖励的 1.5 分太亏了）。
>
> 这种"双层奖励"设计本质上是一个**多目标优化**：格式奖励保证输出结构的可控性，Reward Model 保证内容质量，二者缺一不可。
>
> **第 8 题**：
>
> **DPO 的隐式 KL 约束**：
>
> DPO 的 loss 公式（复习自 `train_dpo_study_plan.md`）：
> $$
> \mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[ \log \sigma\big( \beta \cdot ( \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} ) \big) \right]
> $$
>
> 这里的 $\beta$ 身兼两职：
> 1. **缩放偏好信号**：$\beta$ 越大，chosen 和 rejected 的 log-prob 差距被放大，sigmoid 输入值的绝对值越大，梯度步长越大
> 2. **KL 约束强度**：$\beta$ 也控制着 $\pi_\theta$ 相对 $\pi_{\text{ref}}$ 的偏离代价——$\beta$ 大意味着每一点偏离都会被 loss 中的 log-ratio 项"放大计分"
>
> 问题在于这两个角色**绑死在同一个 $\beta$ 上**。举个例子：
> ```
> 场景：数据质量高，希望大步伐学习偏好，同时又想严格控制 policy 不跑偏
>   → 需要 "大学习幅度 + 强 KL 约束"
>   → 但 β 增大同时加大了学习幅度和 KL 约束 → 可能 OK
>
> 场景：数据有噪声，希望小步伐学习，但 KL 约束不能太松（防止过拟合噪声）
>   → 需要 "小学习幅度 + 强 KL 约束"
>   → 减小 β 会同时减弱 KL 约束 ❌
>   → 增大 β 会同时放大噪声学习 ❌
>   → 无解！
> ```
>
> **数学本质**：在 DPO 中，$\beta$ 出现在 log-ratio 项的外层，是**乘法系数**而不是**加法项**。把公式展开：
> $$
> \mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta \cdot \Delta)]
> $$
> 其中 $\Delta = (\log\pi_\theta(y_w) - \log\pi_{\text{ref}}(y_w)) - (\log\pi_\theta(y_l) - \log\pi_{\text{ref}}(y_l))$。
> $\Delta$ 本身已经隐含了 $\pi_\theta$ 偏离 $\pi_{\text{ref}}$ 的代价（每一项都是 $\log\pi_\theta - \log\pi_{\text{ref}}$ 的形式），$\beta$ 只是统一缩放这个差值，无法区分"我想放大 preference 信号"和"我想收紧 KL 约束"。
>
> 💡 这部分用文字讲解比较抽象，推荐配合**比喻集**阅读：
> → [`knowledge/metaphor/dpo_grpo_beta.md`](../knowledge/metaphor/dpo_grpo_beta.md)（厨师做菜 / 水龙头 / 成绩单 / DJ 调音台）
>
> **对比 GRPO**：显式惩罚是**加法项** $\beta \times \text{per\_token\_kl}$，$\beta$ 仅控制 KL 强度，不碰策略梯度部分。两者完全解耦。
>
> **在线场景下的第二个缺陷**：
>
> DPO 是**离线算法**——数据是静态的 $(y_w, y_l)$ 对。但在线场景（模型自己生成数据）下会出现：
>
> | 偏差来源 | 表现 | DPO 能否区分？ |
> |---------|------|:------------:|
> | **探索偏差** | policy 采样时随机性大，生成了低概率 token，导致 $\pi_\theta$ 相对 $\pi_{\text{ref}}$ 暂时偏离 | ❌ 无法和你下一类区分 |
> | **过拟合偏差** | policy 真的在某个模式上过度置信了 | ❌ 同上 |
>
> 因为 DPO 只看到 **一次静态的 log-ratio 快照**，无法判断这个偏离是"模型在探索/采样噪音"还是"模型真的学歪了"。而 GRPO 的 ref 模型在每个 batch 都提供**实时的 KL 基线**，配合 $\beta$ 独立调节，可以精细控制"允许探索多少，禁止过拟合多少"。
>
> **GRPO 的显式 KL 惩罚**：
> - 优点：β 独立控制 KL 约束强度，可灵活调节；KL 项与策略梯度项完全解耦，方便调试和消融实验；在在线采样场景下能有效防止 policy 在 reward 噪声大的区域过度偏离。
> - 缺点：多了一个超参数 β 需要调优；KL 计算增加前向开销；如果 β 设置不当，要么约束过强（policy 学不动），要么约束过弱（policy 崩坏）。
>
> **GRPO 去掉 KL 惩罚（β=0）的后果**：
>
> policy 会**快速崩溃**。原因：
> 1. **Reward Hacking** — 模型会发现某些 token 模式能稳定获得高 reward，于是疯狂提升这些 token 的概率，偏离合理语言分布。
> 2. **模式坍塌** — 多样性消失，所有回答趋于相同的高 reward 模板。
> 3. **灾难性遗忘** — 失去了 ref 的锚定，policy 可能遗忘预训练/SFT 学到的语言能力，输出变得不自然甚至语无伦次。
>
> 这就是为什么 RLHF/GRPO 中 KL 惩罚不是可选项，而是**必需项**——它是连接"奖励最大化"和"语言建模"两个目标的桥梁。
>
> **第 9 题**：`answer` 字段在训练过程中**不直接参与 loss 计算**（GRPO 的 loss 只依赖 `per_token_logps` 和 advantages），但它有 3 个重要用途：
>
> 1. **Reward Model 评分参考**（`calculate_rewards` 中，如果 `reward_model.get_score` 的实现需要）：某些 Reward Model 的实现会参考标准答案来判断回答的正确性（例如数学题比较最终答案是否与标准答案一致）。
> 2. **日志和评估**：训练过程中可以用 `answer` 与模型生成的回答做对比，计算准确率等指标，用于监控训练进度。
> 3. **调试和人工分析**：当某个 prompt 的 rewards 出现异常时，`answer` 提供了标准答案作为对照，帮助判断是 Reward Model 打分不合理，还是模型生成的回答确实有问题。
>
> 这与 GRPO"在线生成、自探索"的设计理念一致：**标准答案不直接指导梯度更新，仅作为外部评判的参考依据**。

---

## 七、关联文件

```
train_grpo.py
 ├─ model/model_minimind.py              ← Policy 和 Ref 共用的模型定义
 ├─ dataset/lm_dataset.py                ← RLAIFDataset（prompt 数据集）
 ├─ trainer/trainer_utils.py             ← 工具函数（init_model, SkipBatchSampler 等）
 ├─ trainer/train_dpo.py                 ← 前置对比：DPO 的实现（2.6 节有 PPO/GRPO/DPO 详细对比）
 ├─ trainer/train_reason.py              ← 前置依赖：产出的 reason.pth 是 GRPO 的默认起点
 └─ plan/train_dpo_study_plan.md         ← 前置学习：DPO 偏好对齐（含 GRPO vs DPO vs PPO 对比）
```
