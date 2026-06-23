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

- `from dataset.lm_dataset import SFTDataset` → 复用 SFT 的数据集（用 SFTDataset 处理推理数据）
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
 ├─ model/model_minimind.py          ← 模型定义
 ├─ dataset/lm_dataset.py            ← SFTDataset（复用，无需新数据集类）
 ├─ trainer/trainer_utils.py         ← 工具函数
 ├─ trainer/train_full_sft.py        ← 对比参考：普通 SFT 的 loss 计算
 ├─ trainer/train_grpo.py            ← 后续阶段：依赖 reasoning 格式做奖励
 ├─ eval_llm.py                      ← enable_thinking 推理时控制
 └─ plan/train_dpo_study_plan.md     ← 前置学习：DPO 偏好对齐
```
