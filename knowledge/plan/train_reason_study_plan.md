# train_reason.py 学习计划指引

## 一、文件定位

`train_reason.py` 是 MiniMind 项目**推理能力训练**（Reasoning Distillation）脚本。
在 DPO 对齐后的模型基础上，让模型学会"先思考再回答"——输出格式化的推理过程。

```
train_pretrain.py（预训练 ✅）
    ↓
train_full_sft.py（指令微调 ✅）
    ↓
train_lora.py（LoRA 高效微调 ✅）
    ↓
train_dpo.py（偏好对齐 ✅）
    ↓
train_reason.py（推理能力训练 ← 你在这里）
    ↓
train_grpo.py（组相对策略优化，后续阶段）
```

### 为什么需要 Reasoning？

SFT 让模型学会对话，DPO 让模型学会偏好，但两者都是"直接给答案"。
Reasoning 训练让模型学会**在回答前先推理**——输出 `...` 再给 ``。

这与 OpenAI o1 / DeepSeek R1 的思路一致：用更多的推理 token 换取更准确的答案。

---

## 二、核心新概念（与 SFT / DPO 对比）

### 2.1 特殊标记

训练数据中包含四个特殊标记：
- `<think>`：开始推理
- `</think>`：结束推理
- `<answer>`：开始答案
- `</answer>`：结束答案

模型生成时的完整格式：
```
<think>
这个问题需要分步推理：首先...然后...因此...
</think>
<answer>
最终答案是 42
</answer>
```

### 2.2 加权 Loss —— 核心创新点

train_reason.py 与普通 SFT 最大的区别在 loss 计算（L38-53）：

```python
# 1. 和普通 SFT 一样，用 CrossEntropyLoss(reduction='none') 算每个 token 的 loss
loss = loss_fct(shift_logits, shift_labels)   # shape = (B, L)

# 2. 标准 mask：prompt 区域为 0，response 区域为 1
loss_mask = (shift_labels != -100).float()

# 3. ⭐ 特殊标记权重提升 10 倍
sp_ids = torch.isin(shift_labels, special_tokens)  # 找出 <think> 等 token 的位置
loss_mask_flat[sp_ids] = 10                         # 这些位置的权重从 1 → 10
```

**为什么特殊标记要加权？**
- `<think>`、`</think>`、`<answer>`、`</answer>` 是推理格式的**骨架**，
  模型必须准确学会在正确的位置插入这些标记。
- 如果权重为 1，这些标记只占序列的极小比例（1~2%），模型可能学会内容
  但学不会格式——生成时 `<think>` 位置不对、缺少 `</think>` 等。
- 权重 10 相当于告诉模型：**宁可把推理内容写错一点，也要保证格式正确**。

这个技巧不是 train_reason.py 独有的——很多结构生成任务
（代码生成、JSON 生成）都会对语法标记加权。

### 2.3 模型架构

与 DPO 不同，reason 训练**只有一个模型**（不是双模型，也不是 LoRA）：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
```

默认 `--from_weight='dpo'`，即基于 DPO 产出的权重继续训练。

| 方面 | DPO | Reason |
|------|-----|--------|
| 模型数 | 2 个 (policy + ref) | 1 个 |
| 损失函数 | DPO loss | CrossEntropy（加权） |
| 学习率 | 4e-8 | 1e-6 |
| 数据集 | DPODataset（偏好对） | SFTDataset（文本+特殊标记） |
| 特殊标记 | 无 | `<think>` / `<answer>` |
| 重点 | 拉开好坏差距 | 强化推理格式 |

### 2.4 数据集

默认使用 `r1_mix_1024.jsonl`，这是从 DeepSeek R1 蒸馏得到的推理数据，
每条包含完整的 `<think>`...`</think>` + `<answer>`...`</answer>` 格式。

`max_seq_len=720` 比 SFT（340）和 DPO（1024）都要长——推理需要更多 token。

---

## 三、与普通 SFT 的 Loss 对比

| 步骤 | 普通 SFT（train_full_sft.py） | Reason SFT（train_reason.py） |
|------|----|----|
| loss 函数 | CrossEntropyLoss（默认 reduction='mean'） | CrossEntropyLoss（reduction='none'） |
| loss mask 方式 | labels 里 prompt 位置标 -100，CE 自动忽略 | 手动算 mask，prompt=0，response=1 |
| 特殊标记处理 | 无 | `<think>` 等位置权重 ×10 |
| aux_loss | 有（MoE） | 有（MoE） |

普通 SFT 用 label=-100 让 CrossEntropyLoss 自动忽略 prompt token。
Reason 训练用 `reduction='none'` 拿到每个 token 的 loss，再手动乘 mask，
目的是**在乘 mask 之前还能修改个别位置的权重**（特殊标记乘 10）。

### 核心对比：`reduction='none'` + 手动 mask vs `label=-100`

| 方面 | `label=-100`（普通 SFT） | `reduction='none'` + 手动 mask（Reason） |
|------|------------------------|------------------------------------------|
| 权重分配 | 二值：忽略(0) 或 同等参与(1) | 任意值：0、1、10、甚至负权重均可 |
| 灵活度 | 低 — 不能对某个 token 单独加权 | 高 — 平均前可调整每个位置的权重 |
| 实现方式 | loss 函数内部硬编码忽略 -100 | 先算出逐 token loss，再手动乘 mask 后 `.sum() / mask_sum` |
| 典型场景 | 只需区分"算 loss / 不算 loss" | 需要差异化加权（如语法标记 ×10） |

本质：`label=-100` 把忽略逻辑藏在 loss 函数内部，你碰不到中间结果；`reduction='none'` 把逐 token loss **暴露给你**，给了你在平均前干预每个位置权重的能力。如果只用 `label=-100`，做不到"这个 token 比那个 token 重要 10 倍"。

---

## 四、学习目标检查清单

- [ ] 理解 reasoning 训练的目标（让模型学会"先思考再回答"）
- [ ] 理解 `<think>` / `<answer>` 四个特殊标记的作用和格式
- [ ] 理解加权 loss 的核心原理（特殊标记权重 10 倍的原因）
- [ ] 理解 `reduction='none'` + 手动 mask 相比 label=-100 的灵活之处
- [ ] 理解为什么特殊标记需要更大的权重（格式优先于内容）
- [ ] 理解推理数据集的来源（r1_mix_1024.jsonl 是蒸馏数据）
- [ ] 对比 reason 训练和普通 SFT 的 loss 计算差异
- [ ] 理解 train_reason.py 的模型架构（单模型，基于 DPO）
- [ ] 对比推理时 enable_thinking 开关的作用逻辑（eval_llm.py 侧）
- [ ] 理解 reasoning 与 GRPO 的关系（GRPO 依赖 reasoning 格式做奖励）

---

## 五、文件逐段精读计划

### 第 1 层：导入与参数定义

- `from scripts.Dataset.lm_dataset import SFTDataset` → 复用 SFT 的数据集（用 SFTDataset 处理推理数据）
- `--from_weight` 默认 `dpo` → 基于 DPO 产出继续训练
- `--max_seq_len` 默认 `720` → 比 SFT（340）长，推理需要更多 token
- `--data_path` 默认 `../dataset/r1_mix_1024.jsonl` → R1 蒸馏推理数据
- `--learning_rate` 默认 `1e-6` → 和 full SFT 相同

### 第 2 层：train_epoch 的 loss 计算（L23-55）

关键代码逐行解读：

```python
# L28: reduction='none' → 拿到每个 token 的 loss，而不是整体平均
loss_fct = nn.CrossEntropyLoss(reduction='none')

# L40-42: 标准 shift 操作（取 logits[:-1] 和 labels[1:]）
shift_logits = res.logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)).view(shift_labels.size())
# loss shape = (B, L) — 每个 token 位置的交叉熵

# L44-53: 构建加权 mask
loss_mask = (shift_labels != -100).float()       # prompt=0, response=1

sp_ids = torch.isin(shift_labels.view(-1),        # 找出特殊 token 位置
                    torch.tensor(special_ids).to(args.device))
loss_mask_flat = loss_mask.view(-1)
loss_mask_sum = loss_mask_flat.sum()               # 非 prompt 的总 token 数（作为分母）
loss_mask_flat[sp_ids] = 10                        # ⭐ 特殊标记权重×10
loss_mask = loss_mask_flat.view(shift_labels.size())

logits_loss = (loss * loss_mask).sum() / loss_mask_sum
# 分子：每个 token loss × 权重（特殊标记 ×10），然后求和
# 分母：只算 response token 数（不含 prompt），保证平均 loss 有意义
# 注意：因为分子中的特殊标记位置被乘了 10，logits_loss 会比不加权时偏大。
# 但特殊 token 占响应 token 的比例 p 很小（通常 < 5%），实际放大倍数 ≈ 1 + 9p（约 1.45x 以内）。
# 这个偏移对所有样本一致，优化器会自动适应绝对值范围，不影响收敛方向。
```

### 第 3 层：损失计算实例

假设一条数据：
```
input:  "1+1=?"  → labels: [prompt..., <think>, 推, 理, 过, 程, </think>, <answer>, 2, </answer>]
mask:   [0,0,... 0,   10,     1,  1,  1,  1,  10,      10,      1,   10      ]
```

如果模型把 `<answer>` 预测成了 `<think>`，这个 token 的 loss 要乘 10——
模型会被强有力地惩罚，从而快速学会标记的正确位置。

### 第 4 层：其他结构与 SFT 一致

- DDP、AMP、梯度累积、checkpoint 保存等与 full SFT 完全一致
- optimizer: `AdamW(model.parameters(), lr=1e-6)` — 全参数更新
- 保存完整模型 weight（和 full SFT / DPO 一样）

### 第 5 层：eval_llm.py 中的 enable_thinking

推理侧（`eval_llm.py`）的 `enable_thinking` 参数控制是否显示推理过程：

```python
if enable_thinking:
    # 显示 <think> 内容
    print(f"🧠 {think_content}")
    # 再显示 <answer> 内容
    print(f"🤖 {answer_content}")
else:
    # 直接跳过 <think> 区域，只输出 <answer> 内容
```

这个开关让用户可以选择"看推理过程"或"只看最终答案"。

---

## 六、自测题

### 基础题
1. train_reason.py 和 train_full_sft.py 使用相同的数据集类（SFTDataset），
   但 loss 计算方式有一个关键区别，是什么？
2. 为什么特殊标记（`<think>` / `<answer>`）的 loss 权重是 10 而不是 1？
3. `reduction='none'` 和 `reduction='mean'` 的区别是什么？为什么这里要用 'none'？

### 进阶题
4. 如果把特殊标记的权重改为 2（而不是 10），会有什么影响？
5. train_reason.py 的 `--from_weight` 默认是 `dpo`，如果改成 `full_sft` 会怎样？
6. loss_mask_sum 计算的是所有响应 token 的总数（分母），
   但加权后分子增大了（特殊标记 ×10），loss 会不会异常偏大？为什么？

### 深入题
7. 对比 train_reason.py 的加权 loss 与 train_full_sft.py 的 label=-100 策略：
   如果要在 full SFT 中也实现"对某个 token 加权"，应该怎么改？
8. 推理数据 `r1_mix_1024.jsonl` 是蒸馏数据（从 DeepSeek R1 输出蒸馏得到），
   如果换成普通的 SFT 数据但手动加上 `<think>` 标记，效果会一样吗？
9. train_grpo.py 中的 `reasoning_model_reward` 函数检查回答的格式——
   这个奖励函数与 train_reason.py 的加权 loss 是什么关系？

---

## 七、关联文件

```
train_reason.py
 ├─ scripts/Model/model_minimind.py          ← 模型定义
 ├─ scripts/Dataset/lm_dataset.py            ← SFTDataset（复用，无需新数据集类）
 ├─ scripts/Trainer/trainer_utils.py         ← 工具函数
 ├─ scripts/Trainer/train_full_sft.py        ← 对比参考：普通 SFT 的 loss 计算
 ├─ scripts/Trainer/train_grpo.py            ← 后续阶段：依赖 reasoning 格式做奖励
 ├─ eval_llm.py                      ← enable_thinking 推理时控制
 └─ plan/train_dpo_study_plan.md     ← 前置学习：DPO 偏好对齐
```

---

## 八、自测题参考答案

### 基础题

**1.** 普通 SFT 用 `label=-100` 由 CrossEntropyLoss 内部自动忽略 prompt 区域，loss 默认 `reduction='mean'` 做整体平均。Reason 训练用 `reduction='none'` 拿到逐 token loss，再手动构造 mask（prompt=0, response=1, 特殊标记=10），最后 `(loss * mask).sum() / mask_sum` 做加权平均。详见第三节对比表格。

**2.** 特殊标记是推理格式的骨架，数量极少（占序列 1~2%），权重为 1 时模型容易忽略它们的位置。权重 10 迫使优化器优先学会这些标记的正确插入位置，保证推理格式正确。

**3.** `reduction='mean'` 对所有 token loss 求平均；`reduction='none'` 保留每个 token 独立的 loss 值。这里用 'none' 是因为需要在平均前手动乘 mask 修改特殊标记的权重。

### 进阶题

**4.** 权重 2 仍能给予一定强调，但力度弱于 10。结果是格式学习的收敛速度变慢，可能出现更多格式错误（标记位置偏移、缺失闭合标记等），尤其在数据量不足时更明显。

**5.** `--from_weight` 默认 `dpo`，如果改成 `full_sft`：

| 角度 | 从 dpo 开始 | 从 full_sft 开始 |
|------|------------|----------------|
| 基础能力 | 已通过 DPO 对齐偏好，回答质量可靠 | 只有指令跟随，偏好未优化 |
| 学习重点 | 专注学习推理格式（格式=增量） | 需要同时学"好回答"+"推理格式" |
| 学习率风险 | LR=1e-6 适中，DPO 权重稳定 | LR=1e-6 相对偏高，可能破坏 SFT 学到的能力 |
| 预期效果 | 格式学得快，推理内容质量高 | 可能学完格式但推理内容质量下降 |

从 `dpo` 开始能**解耦"学格式"和"学内容"**，让 reasoning 训练只专注于格式。从 `full_sft` 开始相当于让模型同时从头学格式和内容，1e-6 的学习率可能导致灾难性遗忘，效果通常更差。

**6.** 会偏大，但不是异常偏大。设响应 token 总数为 N，特殊 token 占比为 p（通常 < 5%）：
- 未加权时：loss0 = sum(loss) / N
- 加权后：loss1 = sum(loss(普通)) + 10 * sum(loss(特殊)) / N
- 即 loss1 = loss0 * (1 + 9p)
- 当 p=5% 时，loss1 = 1.45 * loss0，约增大 45%

这个增大：① 有限，不会出现 10 倍暴涨；② 对所有样本一致，优化器自动适应；③ 真正重要的是梯度的相对方向而非绝对值。**结论：绝对值确实上升，但幅度有限且一致，不影响收敛。**

### 深入题

**7.** 在 full SFT 中实现单个 token 加权，需要做相同改造：把 CrossEntropyLoss 的 reduction 从 'mean' 改为 'none'，手动构造 mask 张量，对目标 token 设 >1 权重，最后 `(loss * mask).sum() / mask.sum()`。</br></br>为什么要这样做？标准 SFT 用 `label=-100` 只能做**二值区分**（忽略/参与），但很多场景需要对不同 token 差异化加权：</br>- 强调关键内容：答案中的核心结论、数字、术语比语气词更重要，应赋予更高权重</br>- 结构化输出：想让模型学会输出 JSON/代码时，对语法标记（括号、引号）加权可提升格式准确率</br>- 降噪：训练数据中某些部分由弱模型生成、质量较低，可以降权处理</br>- 安全对齐：安全拒绝语句（"我无法帮助"）中每个 token 都至关重要，加权可强化记忆</br></br>下面用一条 10 个 token 的样本数值对比两种分母：</br></br>| 位置 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |</br>|---|---|---|---|---|---|---|---|---|---|---|</br>| loss 值 | 0.5 | 0.3 | **2.0** | 0.4 | 0.6 | **1.5** | 0.7 | 0.2 | 0.8 | 0.9 |</br>| 权重 mask | 1 | 1 | **10** | 1 | 1 | **10** | 1 | 1 | 1 | 1 |</br>| 加权 loss | 0.5 | 0.3 | **20.0** | 0.4 | 0.6 | **15.0** | 0.7 | 0.2 | 0.8 | 0.9 |</br></br>分子（加权 loss 总和）= 0.5+0.3+20+0.4+0.6+15+0.7+0.2+0.8+0.9 = **39.4**</br></br>两种分母对比：</br>- **分母 = mask.sum() = 28**（加权平均）：39.4 / 28 ≈ **1.407**</br>- **分母 = loss_mask_sum（有效 token 数）= 10**（train_reason.py 实际做法）：39.4 / 10 = **3.94**</br>- 参考：不加权时普通平均 = (0.5+0.3+2+0.4+0.6+1.5+0.7+0.2+0.8+0.9)/10 = **0.67**</br></br>结论：两种分母都让特殊 token（位置 3 和 6）对 loss 的贡献被放大。`mask.sum()` 是严格加权平均，loss 绝对值更平稳（1.407 vs 不加权 0.67）；`loss_mask_sum` 得到的 loss 绝对值膨胀更大（3.94），但梯度相对比例一致，优化器自动适应。

**8.** 不会一样。`r1_mix_1024.jsonl` 是 DeepSeek R1 蒸馏数据，推理链完整且经过验证。手动加 `<think>` 的普通 SFT 数据只是形式上加了标记，推理内容本身未经强化/蒸馏，质量不可比。

**9.** 互补关系：加权 loss 是**训练阶段**的软约束（梯度引导模型学会格式），`reasoning_model_reward` 是**强化学习阶段**的硬约束（奖励信号惩罚格式错误）。加权 loss 打基础，GRPO 奖励做精调。
