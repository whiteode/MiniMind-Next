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
| ③ | `exp(policy_logps - sg(policy_logps)) × adv - β × kl` | 第一项是重要性采样权重 × 优势值；第二项是 KL 惩罚 |
| ④ | `mean over sequence then over batch` | 先对序列平均（去除 padding），再对 batch 平均 |

其中 `per_token_logps - per_token_logps.detach()` 是**停止梯度**技巧：
`detach()` 让计算图中的分母部分不参与梯度传播，避免梯度估计方差过大。

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
           重要性采样权重            KL 散度惩罚
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
- KL 惩罚 + 重要性采样权重
- completion_mask 只保留 EOS 之前的 token

---

## 六、自测题

### 基础题

1. GRPO 的每个训练 step 分为哪 4 步？用自己的话描述每一步在做什么。

2. GRPO 需要哪 3 个模型？各自的角色和可训练状态是什么？

3. `num_generations=8` 的含义是什么？增大/减小这个值会有什么影响？

### 进阶题

4. 组内相对优势 `(rewards - mean_r) / (std_r + 1e-4)` 解决了什么问题？
   为什么要对同一个 prompt 的多个回答计算相对优势，而不是直接用 Reward Model 的绝对分数？

5. 对比 `exp(kl_div) - kl_div - 1` 在 kl_div=0、kl_div=0.5、kl_div=1、kl_div=-0.5 时的值。
   这个惩罚函数有什么性质？（提示：对称性，最小值在何处）

6. `per_token_logps - per_token_logps.detach()` 中 `detach()` 的作用是什么？
   如果不加 detach 会怎样？

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
