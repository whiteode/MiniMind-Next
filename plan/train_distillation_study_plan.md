# train_distillation.py 学习计划（知识蒸馏）

## 一、写在前面：什么是知识蒸馏？

### 1.1 从"老师教学生"说起

你已经学过了 MiniMind 的所有训练方法：

| 阶段 | 比喻 | 学了吗 |
|------|------|:------:|
| 预训练 | 婴儿学认字——大量阅读，建立语感 | ✅ |
| SFT 微调 | 老师手把手教"用规范句式回答问题" | ✅ |
| DPO | 给模型看"好答案 vs 差答案"，学会偏好 | ✅ |
| GRPO/PPO/SPO | 模型自己尝试答题，根据分数自我改进 | ✅ |

**蒸馏（Distillation）** 是完全不同的一类方法——它的核心思想是 **"让一个小模型（学生）模仿一个大模型（老师）的行为"**。

> 🐶 **大白话**：你有一个 GPT-4（老师），但它太贵太大跑不动。你想训练一个小模型（学生），让它学老师的"思考方式"，而不是只学标准答案。蒸馏就是**把大模型的"知识"迁移到小模型里**。

### 1.2 为什么需要蒸馏？

```
原始方案：训一个大模型（768维，16层）→ 部署，但推理慢、显存大
蒸馏方案：训一个小模型（512维，8层）→ 部署，推理快、显存省
          但小模型直接训效果不好 → 让大模型当老师带它
```

具体好处：
- **推理速度更快**：小模型 forward 更快
- **显存占用更少**：更小的 hidden_size 和 num_layers
- **效果比直接训小模型好**：老师的 soft label 比 hard label 信息更丰富
- **适合端侧部署**：手机、嵌入式设备

### 1.3 蒸馏和之前的方法有什么本质区别？

| 维度 | 之前的方法（SFT/DPO/GRPO/PPO/SPO） | 蒸馏 |
|------|--------------------------------------|------|
| 目标 | **提高模型的任务能力** | **让小模型模仿大模型** |
| 监督信号 | Hard Label（标准答案） + Reward | Soft Label（老师输出的概率分布） |
| 模型数 | 1~5 个（policy/ref/reward/critic） | **2 个（Teacher + Student）** |
| 老师可训练？ | Ref 冻结、Reward 冻结 | **Teacher 冻结** |
| 核心公式 | 策略梯度 / 对比损失 | **KL 散度 + CE** |

---

## 二、核心概念（循序渐进）

### 2.1 Softmax 与温度（Temperature）

#### 先复习：普通 Softmax

```python
# 对一个 logits 向量 [z1, z2, ..., zN]，softmax 输出概率：
p_i = exp(z_i) / Σ_j exp(z_j)
```

假设模型对三个词预测的 logits 是 `[2.0, 1.0, 0.1]`：

```
softmax([2.0, 1.0, 0.1]) = [0.659, 0.242, 0.099]
```

结果是"猫"有 65.9% 概率，"狗"有 24.2%。

#### 带温度的 Softmax

```python
# 加了一个温度参数 T：
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

**温度 T 的作用**（T > 0）：

| T | 效果 | 概率分布 |
|:-:|------|----------|
| T < 1 | 使分布更**尖锐** | 最大概率更大，小概率更小 |
| T = 1 | 普通 softmax | 保持不变 |
| T > 1 | 使分布更**平滑** | 最大概率变小，小概率变大 |

**例：logits = [2.0, 1.0, 0.1]**

```
T=0.5: softmax([4.0, 2.0, 0.2]) = [0.864, 0.117, 0.019]  ← 尖锐
T=1.0: softmax([2.0, 1.0, 0.1]) = [0.659, 0.242, 0.099]  ← 正常
T=2.0: softmax([1.0, 0.5, 0.05]) = [0.422, 0.341, 0.237] ← 平滑
```

> 🐶 **大白话**：温度 T 控制模型的"自信程度"。T 越小，模型越自信（大的更大，小的更小）；T 越大，模型越谦虚（大家都差不多，概率分布更均匀）。

#### 为什么蒸馏要用温度？

**核心洞察**：老师模型输出的 hard label（只有一个词是 1，其他都是 0）信息量太少。比如：

- Hard label：`[1, 0, 0]` → 只知道"猫"是对的
- Soft label（T=2.0）：`[0.422, 0.341, 0.237]` → 还知道"狗"也有一定可能，"鸟"不太可能

这个**分布里隐藏了老师对类别之间相似度的理解**——这就是"知识"。

> 🐶 **大白话**：普通训练只告诉学生"正确答案是猫"；蒸馏告诉学生"正确答案是猫，但狗也有一点点可能，因为它们都是宠物"。第二种教学方式的信息量大多了。

### 2.2 Soft Target vs Hard Target

| 类型 | 是什么 | 例子 | 信息量 |
|------|--------|------|:------:|
| **Hard Target** | 数据集中的**原始标签**（one-hot） | "猫" → `[1, 0, 0]` | 低——只告诉哪个对 |
| **Soft Target** | 老师模型输出的**概率分布** | `[0.422, 0.341, 0.237]` | 高——还告诉"猫和狗有多像" |

在 `train_distillation.py` 中：

```python
# Hard Target：数据集的 labels（标准答案）
ce_loss = F.cross_entropy(student_logits, shift_labels, ...)  # L8-74

# Soft Target：老师的输出分布
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)  # L26
```

### 2.3 KL 散度（这节课的 KL 和之前有什么不同？）

你已经见过很多次 KL 散度了：

| 位置 | KL 的作用 | 公式 |
|------|-----------|------|
| GRPO loss | 防止 policy 离 ref 太远 | `exp(kl) - kl - 1` |
| SPO/PPO loss | 同上 | `exp(kl) - kl - 1` |
| **蒸馏 loss** | **让学生分布模仿老师分布** | `F.kl_div(student_log_probs, teacher_probs)` |

**在蒸馏中，KL 散度衡量的是"学生的概率分布和老师的概率分布有多像"**。

```
KL(老师 || 学生) = Σ 老师(词) × log(老师(词) / 学生(词))
```

- 如果学生分布和老师**一模一样** → KL = 0
- 如果学生分布和老师**不一样** → KL > 0
- KL **不对称**：KL(老师||学生) ≠ KL(学生||老师)

> 🐶 **大白话**：KL 散度就是算"学生的答案分布和老师的答案分布有多大差距"。差距越小，学生学得越好。

#### F.kl_div 的参数含义

```python
F.kl_div(
    student_log_probs,   # 学生的 log 概率 [log(p1), log(p2), ...]
    teacher_probs,       # 老师的概率（不是 log）[q1, q2, ...]
    reduction='batchmean'  # 对整个 batch 求平均
)
```

注意：PyTorch 的 `F.kl_div` 要求第一个参数是 **log 概率**，第二个参数是 **概率**（不是 log）。公式：`Σ q_i × (log(q_i) - log(p_i)) = Σ q_i × log(q_i/p_i)`

### 2.4 为什么蒸馏 loss 要乘 `temperature^2`？

这是蒸馏论文（Hinton et al., 2015）中的关键细节：

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return (temperature ** 2) * kl  # ← 这里乘了 T²
```

**为什么需要 T²？**

数学原因：当温度 T 较大时，softmax 的输出值都变小（被 T 除），梯度也会变小。乘回 T² 是为了**保持梯度量级不变**，让 T 的选择不影响学习率。

简单理解：

```
梯度 ∝ 1/T²          # 温度使梯度变小
(T²) × 梯度 ∝ 1      # 乘 T² 补偿回来
```

如果不乘 T²，温度越高，loss 越小，梯度也越小，学生就学不动了。

> 🐶 **大白话**：温度把老师的答案"稀释"了，梯度也变弱了。乘回 T² 就是"把梯度调回正常大小"，这样你调温度的时候只需要关心分布平滑度，不用操心学习率要不要跟着改。

### 2.5 加权损失：`alpha * CE + (1-alpha) * Distill`

```python
loss = alpha * ce_loss + (1 - alpha) * distill_loss
```

| alpha 取值 | 含义 | 效果 |
|:----------:|------|------|
| alpha = 1.0 | 只用 CE，不用蒸馏 | 退化成普通 SFT 训练 |
| alpha = 0.5 | CE 和蒸馏各一半 | **默认，平衡模式** |
| alpha = 0.0 | 只用蒸馏，忽略标准答案 | 可能学不到正确内容 |

**为什么两个都需要？**

- **CE（Hard Target）**：保证学生学到**标准答案**——"回答的内容是对的"
- **Distill（KL）**：保证学生学到**老师的风格**——"回答的方式像老师"

两者都重要。只用 CE = 普通 SFT；只用 Distill = 可能学到错误知识（老师也不总是对的）。

---

## 三、代码结构总览

```
train_distillation.py（235 行，所有训练脚本中最短的一个）
│
├── 导入与全局配置（L1-21）
├── distillation_loss()（L24-35）          ← 核心①：蒸馏损失函数
├── train_epoch()（L38-133）               ← 核心②：训练循环
│   ├── 学生前向 + 老师前向（L54-63）
│   ├── CE Loss 计算（L67-77）
│   ├── Distillation Loss 计算（L80-87）
│   ├── 加权融合（L90）
│   └── 反向传播 + 日志 + 保存（L92-133）
└── main（L136-235）
    ├── 参数定义（L138-165）
    ├── 环境初始化（L168-196）
    ├── 双模型加载（L193-201）
    ├── 数据加载（L202-205）
    └── 训练循环（L222-232）
```

这是你见过的**最简洁的训练脚本**——没有复杂的优势计算，没有多个模型互相作用，只有"老师教学生"。

---

## 四、核心代码逐行解读

### 4.1 `distillation_loss()`（L24-35）

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl
```

**逐行解读：**

| 行 | 代码 | 含义 |
|----|------|------|
| L25-26 | `with torch.no_grad():` + `teacher_probs = ...` | 老师冻结，不计算梯度 |
| L26 | `teacher_logits / temperature` | 用温度软化 logits |
| L26 | `.softmax(dim=-1).detach()` | 转成概率分布 + 彻底断开梯度 |
| L28 | `student_log_probs = F.log_softmax(...)` | 学生 logits 也做温度软化，取 log |
| L30-34 | `F.kl_div(student_log_probs, teacher_probs)` | **注意顺序**：第一个参数是**学生 log 概率**，第二个是**老师概率** |
| L35 | `(temperature ** 2) * kl` | 温度补偿，让梯度量级不受 T 影响 |

### 4.2 双模型架构（L193-201）

```python
# 学生模型（可训练）
model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)

# 教师模型（冻结不更新）
teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
teacher_model.eval()
teacher_model.requires_grad_(False)
```

**核心差异：**

| | 学生（Student） | 老师（Teacher） |
|--|-----------------|----------------|
| 模型大小 | 小（默认 512维/8层） | 大（默认 768维/16层） |
| 是否训练 | ✅ 训练 | ❌ 冻结 |
| 权重来源 | `from_student_weight`（默认 full_sft） | `from_teacher_weight`（默认 full_sft） |
| 是否求梯度 | 是 | `requires_grad_(False)` |
| 角色 | 学东西的学生 | 输出 soft target 的老师 |

> 🐶 **大白话**：老师（大模型）知识渊博但不更新，学生（小模型）认真听讲并更新自己的参数。学生不仅要学标准答案（CE），还要学老师的答题风格（Distill）。

### 4.3 前向传播（L54-63）

```python
# 学生前向
with autocast_ctx:
    res = model(input_ids)
    student_logits = res.logits[..., :-1, :].contiguous()

# 老师前向（no_grad）
if teacher_model is not None:
    with torch.no_grad():
        teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
        vocab_size_student = student_logits.size(-1)
        teacher_logits = teacher_logits[..., :vocab_size_student]  # 截到学生词表大小
```

**注意第 63 行**：`teacher_logits = teacher_logits[..., :vocab_size_student]`

为什么截断？因为老师模型可能词表更大（不同 size 的模型用不同 tokenizer），但蒸馏只需要在学生词表范围内做 KL。超过学生词表的部分老师再怎么认为有可能，学生也输出不了，不算。

### 4.4 损失计算（L67-90）

**CE Loss：**

```python
ce_loss = F.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction='none'
)
ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
```

和 SFT 一模一样的 CE 计算——用 `ignore_index=-100` 忽略 prompt 部分，只对 response 算 loss。

**Distillation Loss：**

```python
distill_loss = distillation_loss(
    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
    temperature=temperature
)
```

注意这里**只取 response 部分**（`loss_mask_flat == 1`）做蒸馏——prompt 部分不需要模仿老师。

**加权融合：**

```python
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
```

### 4.5 核心超参数

| 参数 | 默认值 | 含义 |
|------|:------:|------|
| `--student_hidden_size` | 512 | 学生模型隐藏层维度 |
| `--student_num_layers` | 8 | 学生模型层数 |
| `--teacher_hidden_size` | 768 | 老师模型隐藏层维度 |
| `--teacher_num_layers` | 16 | 老师模型层数 |
| `--alpha` | 0.5 | CE 损失权重；总损失 = alpha×CE + (1-alpha)×Distill |
| `--temperature` | 1.5 | 蒸馏温度，推荐范围 1.0~2.0 |
| `--from_student_weight` | full_sft | 学生模型基于哪个 checkpoint 初始化 |
| `--from_teacher_weight` | full_sft | 老师模型基于哪个 checkpoint 初始化 |
| `--learning_rate` | 5e-6 | 比 SFT（5e-5）小一个数量级 |

---

## 五、完整训练流程

```
┌─────────────────────────────────────────────┐
│  ① 初始化环境和随机种子                       │
│     - DDP / 分布式设置                        │
│     - 设置随机种子                            │
├─────────────────────────────────────────────┤
│  ② 加载学生模型 + 老师模型                    │
│     - 学生：小模型，可训练                     │
│     - 老师：大模型，冻结                       │
│     - 两者用同一份 SFT 数据                   │
├─────────────────────────────────────────────┤
│  ③ 对每个 batch：                            │
│     a. 学生前向 → student_logits             │
│     b. 老师前向（no_grad）→ teacher_logits    │
│     c. 算 CE Loss（学生 vs 标准答案）          │
│     d. 算 Distill Loss（学生 vs 老师分布）     │
│     e. loss = alpha × CE + (1-alpha) × KL    │
│     f. 反向传播，更新学生参数                 │
├─────────────────────────────────────────────┤
│  ④ 保存学生模型                              │
│     - 学生是我们要部署的，老师只是辅助          │
└─────────────────────────────────────────────┘
```

---

## 六、蒸馏 vs 之前所有方法的对比

### 6.1 模型数量对比

| 方法 | 训练中涉及模型数 | 需冻结的模型 |
|------|:----------------:|:------------:|
| SFT | **1**（policy） | 无 |
| DPO | **2**（policy + ref） | ref |
| GRPO | **3**（policy + ref + reward） | ref + reward |
| PPO | **5**（actor + ref + reward + critic + old_actor） | ref + reward + old_actor |
| SPO | **3**（policy + ref + reward） | ref + reward |
| **蒸馏** | **2**（student + teacher） | **teacher** |

### 6.2 Loss 对比

```
SFT:     loss = CE(student, hard_label)
DPO:     loss = -log σ(β*(ref-chosen - ref-rejected))  ← 偏好对
GRPO:    loss = -logp × A + β × KL(policy || ref)       ← 组内优势
PPO:     loss = -min(surr1, surr2) + MSE(V, R) + β×KL   ← Critic 估值
SPO:     loss = -logp × A + β × KL(policy || ref)       ← 滑动基线
蒸馏:    loss = α × CE(student, hard_label) + (1-α) × KL(student || teacher)
```

**蒸馏是唯一一个 loss 中不包含 reward / advantage / 偏好对的**——它只关心"学生像不像老师"。

### 6.3 适用场景

| 场景 | 用蒸馏 | 用其他方法 |
|------|:------:|:----------:|
| 想把大模型压缩成小模型 | ✅ **最佳选择** | ❌ |
| 部署到手机/嵌入式设备 | ✅ | ❌ 模型太大 |
| 提高模型任务能力 | ❌ 蒸馏不提高能力上限 | ✅ GRPO/PPO/SPO |
| 小模型效果差，想提升 | ✅ 用大模型带 | ❌ 小模型自己训不好 |

---

## 七、自测题

### 基础

**1. 蒸馏的 Teacher 模型是训练的还是冻结的？为什么？**

答案：冻结的。因为老师的作用是提供稳定的 soft target，如果老师也在更新，学生就不知道跟谁学，训练不稳定。

**2. 温度 T 的作用是什么？T=1.0 和 T=2.0 有什么区别？**

答案：T 控制概率分布的平滑度。T > 1 使分布更平滑（大概率变小，小概率变大），T < 1 使分布更尖锐（大概率更大）。

**3. 为什么蒸馏 loss 要乘 T²？**

答案：温度使梯度变小（梯度 ∝ 1/T²），乘 T² 补偿回来，让调温度时不需要改学习率。

**4. `distillation_loss()` 中为什么用 `no_grad()` 包老师？**

答案：老师冻结，不需要梯度。`no_grad()` 省显存、省计算。

**5. 第 63 行为什么要把老师的 logits 截断到学生词表大小？**

答案：老师模型可能词表更大（不同 size 模型用不同 tokenizer），但 KL 散度只需要在学生能输出的范围内算。超出部分的概率学生根本学不了。

### 进阶

**6. `alpha` 参数的作用是什么？alpha=0.0 和 alpha=1.0 各是什么效果？**

答案：
- `alpha=1.0`：只用 CE，退化为普通 SFT
- `alpha=0.0`：只用蒸馏，可能学到老师的错误知识
- `alpha=0.5`：默认，两者平衡

**7. 为什么蒸馏的 learning_rate（5e-6）比 SFT（5e-5）小？**

答案：蒸馏时学生已经有了不错的初始化（from_student_weight），不需要大学习率；而且 KL 散度本身梯度信号比 CE 弱，大学习率容易破坏已有知识。

**8. 蒸馏的数据集用的是 SFTDataset，和 SFT 训练的数据集一样吗？为什么？**

答案：一样。蒸馏和 SFT 都用同一份 SFT 数据，只是因为蒸馏的监督信号来自老师的 soft target 而非硬标签。数据不需要特殊构造。

**9. 如果老师模型比学生还小（student_hidden_size > teacher_hidden_size），会发生什么？**

答案：学生比老师还大 → 学生学不到新知识，因为老师的能力上限低于学生。蒸馏通常要求**老师 > 学生**。

**10. 蒸馏和之前学过的 KL 惩罚（GRPO/SPO 中的 `exp(kl) - kl -1`）有什么不同？**

答案：两个 KL 的作用完全不同：

| | 蒸馏 KL | 对齐 KL 惩罚 |
|--|---------|-------------|
| 比较对象 | 学生 vs 老师 | policy vs ref |
| 目的 | **让学生模仿老师分布** | **限制 policy 偏离 ref** |
| 方向 | 单向（学生学老师） | 双向（抑制任何方向的大幅偏离） |
| 是否要接近 | **是，越接近越好** | 否，不能偏离太远但也不需要完全一致 |
| 公式 | `Σ q_teacher × log(q_teacher/p_student)` | `exp(log_ref - log_policy) - (log_ref - log_policy) - 1` |
| 不对称性 | ✅ 有（学生低估老师高估 惩罚不对称） | ✅ 有（policy 比 ref 更自信时惩罚重） |

**11. `F.kl_div` 的两个参数顺序能互换吗？**

答案：不能。PyTorch 的 `F.kl_div(input, target)` 要求：
- `input` = **学生的 log 概率**（`log_softmax` 的输出）
- `target` = **老师的概率**（`softmax` 的输出，不需要 log）

公式计算的是 `Σ target × (log(target) - log(input))`。如果顺序互换，会计算 `KL(input || target)`，含义变为"老师学学生"，方向反了。

### 深入

**12. 如何理解"soft label 比 hard label 信息量更大"？给一个具体例子。**

答案：假设老师对"今天天气怎么样？"的 token 预测分布：

```
Hard label： "好"=1.0, "坏"=0.0, "热"=0.0     ← 只知最好
Soft label： "好"=0.60, "坏"=0.15, "热"=0.15, ...  ← 还知哪些相关哪些不相关
```

学生从 hard label 只知道"好"是对的；从 soft label 还学到了"坏"和"热"和天气也相关，它们的概率差不多。这个**分布间的相对关系**就是"知识"。

**13. 为什么蒸馏只用 response 部分的 KL（`loss_mask_flat == 1`），prompt 部分不用？**

答案：prompt 部分对所有模型都一样（输入），不需要学。只有 response 部分是模型要生成的，才需要学生模仿老师的风格。

**14. 如果不是做模型压缩，只是想提升小模型效果，可以用蒸馏吗？**

答案：可以，但蒸馏**不提高能力上限**。如果老师本身在某些任务上不行，学生也学不到。蒸馏做的是"知识迁移"而非"能力创造"。

---

## 八、与其他文件的关系

```
train_distillation.py
 ├─ model/model_minimind.py          ← Student 和 Teacher 共享的模型定义
 ├─ dataset/lm_dataset.py            ← SFTDataset（和 SFT 训练用同一份数据）
 ├─ trainer/trainer_utils.py         ← 工具函数（init_model, Logger, lm_checkpoint 等）
 └─ plan/train_full_sft_study_plan.md ← 前置知识：SFT 数据格式和 CE Loss
```

**前置知识要求**：
- 已掌握 SFT 训练流程（train_full_sft.py）
- 理解 CE Loss 和 `ignore_index=-100` 的标签屏蔽机制
- 理解 SFTDataset 的数据格式（chat template 和 label 生成）

---

## 九、推荐学习路径

1. **先看理论**：仔细阅读本文第二、三节（核心概念 + 代码结构总览）
2. **通读代码**：打开 `train_distillation.py`，对照本文第四节逐行看
3. **动手实验**：
   - 试跑一次默认参数：`python train_distillation.py`
   - 改温度：T=1.0 vs T=3.0，观察 distill loss 变化
   - 改 alpha：alpha=0.2 vs alpha=0.8，观察 CE/Distill 平衡
   - 对比：直接 SFT 一个小模型 vs 用大模型蒸馏，效果差异
4. **回答问题**：完成第七节自测题
5. **更新 checklist**：回到 `learning_checklist.md` 打勾
