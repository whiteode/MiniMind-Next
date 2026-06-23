# train_dpo.py 学习计划指引

## 一、文件定位

`train_dpo.py` 是 MiniMind 项目**偏好对齐阶段**的训练脚本。在 SFT 的基础上，
用偏好数据（chosen / rejected 对）让模型学会"什么是好的回答"。

```
train_pretrain.py（预训练 ✅）
    ↓ 产出 pretrain.pth
train_full_sft.py（指令微调 ✅）
    ↓ 产出 full_sft.pth
train_lora.py（LoRA 高效微调 ✅）
    ↓ 产出 lora_xxx.pth
train_dpo.py（偏好对齐 ← 你在这里）
    ↓ 产出 dpo.pth
train_reason.py → train_grpo.py（后续阶段）
```

### 为什么需要 DPO？

SFT 只是让模型学会"模仿"好的回答格式，但无法区分"好回答"和"差回答"。
DPO 让模型**偏好** chosen 回答、**远离** rejected 回答，实现对齐。

| 阶段 | 目标 | 数据 |
|------|------|------|
| Pretrain | 学习语言规律 | 纯文本 |
| SFT | 学习对话格式 | 指令+回答 |
| **DPO** | **学习偏好什么回答** | **chosen/rejected 对** |

---

## 二、核心新概念（和 SFT/LoRA 对比）

### 2.1 DPO 的核心思想

DPO（Direct Preference Optimization）直接优化偏好，不需要 RLHF 的 reward model + PPO 复杂流程。

**DPO loss 公式**：

```
L = -log σ( β * (log π(y_chosen|x)/π_ref(y_chosen|x) - log π(y_rejected|x)/π_ref(y_rejected|x)) )
```

直观理解为：
- 让 policy 模型对 chosen 回答的概率尽可能**高于** ref 模型
- 让 policy 模型对 rejected 回答的概率尽可能**低于** ref 模型
- β 控制这个"拉开差距"的力度

### 2.2 双模型架构

DPO 需要**两个模型**（train_dpo.py:174-184）：

```python
model, tokenizer = init_model(...)       # policy 模型 → 可训练，将被优化
ref_model, _ = init_model(...)           # reference 模型 → 冻结，作为基准
ref_model.eval()
ref_model.requires_grad_(False)
```

- **policy 模型**：我们要训练的模型，它的参数会更新
- **ref 模型**：冻结的参考模型，用来计算"更新了多少"
- 两个模型**初始权重完全相同**（都从 `--from_weight` 加载）
- 训练过程中 ref 不变，policy 逐渐偏离 ref

### 2.3 log_probs 的计算方式

与 SFT 不同，DPO 需要逐 token 的 log probability（而非交叉熵 loss）：

```python
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token
```

使用 `log_softmax` + `gather` 取出 labels 对应位置的 log_prob。

### 2.4 数据拼接策略

DPO 的 batch 结构（train_dpo.py:57-66）：

```python
x = torch.cat([x_chosen, x_rejected], dim=0)  # chosen 和 rejected 拼接
y = torch.cat([y_chosen, y_rejected], dim=0)   # 前半 batch = chosen，后半 = rejected
```

这样一次 forward 同时计算 chosen 和 rejected 的 log_probs。

### 2.5 Loss 中的分拆

dpo_loss 内部按 batch 维度分半（train_dpo.py:40-51）：

```python
batch_size = ref_log_probs.shape[0]
chosen_ref_log_probs = ref_log_probs[:batch_size // 2]     # 前半是 chosen
reject_ref_log_probs = ref_log_probs[batch_size // 2:]     # 后半是 rejected
```

### 2.6 极低的学习率

DPO 的学习率非常小（默认 `4e-8`），注释写"建议 <=5e-8 避免遗忘"。
这是因为 DPO 是在**已训练好的 SFT 模型**上微调偏好，LR 过大会破坏 SFT 阶段学到的知识。

---

## 三、与 SFT / LoRA 的关键差异

| 方面 | SFT / LoRA | DPO |
|------|-----------|-----|
| 损失函数 | CrossEntropy | DPO loss |
| 训练数据 | (prompt, answer) 单条 | (chosen, rejected) 偏好对 |
| 模型数量 | 1 个 | 2 个（policy + ref） |
| 参考模型 | 无 | 有，冻结 |
| 学习率 | 1e-6 (full) / 1e-4 (LoRA) | **4e-8**（极小） |
| 参数更新 | 所有参数 / LoRA 参数 | **全部参数** |
| 优化目标 | 模仿回答 | 偏好 chosen，远离 rejected |

---

## 四、学习目标检查清单

- [ ] 理解 DPO 的核心思想（直接偏好优化，无需奖励模型）
- [ ] 理解 DPO loss 的公式和计算过程
- [ ] 理解 policy 模型和 reference 模型的双模型架构
- [ ] 理解 ref 模型为什么需要冻结以及初始权重来源
- [ ] 理解 `logits_to_log_probs` 的实现原理
- [ ] 理解 DPO 数据集中 chosen / rejected 的拼接与分拆
- [ ] 理解为什么 DPO 的 learning rate 极小（4e-8 级别）
- [ ] 对比 DPO 和 SFT 的 loss、数据、训练目标差异
- [ ] 理解 DPODataset 的 `generate_loss_mask` 如何处理 prompt/response 边界
- [ ] 理解 DPO 和 GRPO / PPO 的核心区别（无 reward model、无 critic）

---

## 五、文件逐段精读计划

### 第 1 层：导入与参数定义

- `from dataset.lm_dataset import DPODataset` → DPO 专用数据集
- `--beta`（默认 0.1）：DPO loss 中的温度参数，控制"拉开差距"的力度
- `--learning_rate`（默认 4e-8）：极小学习率，这是 DPO 的关键特征

### 第 2 层：logits_to_log_probs（L24-30）

- `log_softmax`：将 logits 转成 log probability
- `gather`：从 vocab 维度提取目标 token 对应位置的 log_prob
- 输出 shape: `(batch_size, seq_len)`

### 第 3 层：dpo_loss（L33-51）

- 先对 sequence 维度取平均（用 mask 过滤 padding）
- 按 batch 分半：前半 chosen、后半 rejected
- 计算 `pi_logratios` 和 `ref_logratios`
- `logsigmoid(beta * (pi_ratio - ref_ratio))` 作为 loss

### 第 4 层：train_epoch（L54-120）

- 数据拼接：chosen 和 rejected cat 成一个大 batch
- forward：先跑 ref_model（no_grad），再跑 policy model
- 梯度裁剪作用于 `model.parameters()`（全参数训练，不像 LoRA 只裁剪部分）
- 保存的是完整模型权重（和 full SFT 一样）

### 第 5 层：main 函数（L123-219）

- **双模型初始化**（L175-184）：
  ```python
  model, tokenizer = init_model(...)
  ref_model, _ = init_model(...)      # 独立初始化的副本
  ref_model.eval()
  ref_model.requires_grad_(False)
  ```
- 优化器：`AdamW(model.parameters(), lr=4e-8)` → 全参数更新
- **注意**：ref_model 不传给 DDP，也不传给 optimizer，只在前向时用

### 第 6 层：DPODataset（lm_dataset.py:169-239）

- 每条数据含 `chosen` 和 `rejected` 两个对话列表
- `generate_loss_mask`：用 `bos_id` 和 `eos_id` 标记 assistant 回复区域
- 返回 `x_chosen / x_rejected`（input），`y_chosen / y_rejected`（label），`mask_chosen / mask_rejected`

---

## 六、自测题

### 基础题
1. DPO 为什么需要两个模型（policy 和 ref）？ref 模型为什么必须冻结？
2. dpo_loss 中 `batch_size // 2` 的假设是什么？如果 batch_size 是奇数会怎样？
3. DPO 的学习率为什么远比 SFT 小？

### 进阶题
4. 如果把 ref 模型也设为可训练（requires_grad=True），loss 会怎样变化？
5. DPO 的 beta 参数增大或减小分别会有什么影响？
6. 对比 DPO loss 和 RLHF 中 PPO loss 的核心差异。

### 深入题
7. DPODataset 的 `generate_loss_mask` 如何区分 chosen 和 rejected 中的 prompt 和 response 区域？
   和 SFTDataset 的 `generate_labels` 有什么异同？
8. DPO 训练中直接合并两个 batch（chosen+rejected）前向一次和分别前向两次各有什么优劣？
9. 如果用 LoRA 做 DPO，ref 模型是否需要也注入 LoRA？为什么？

---

## 七、关联文件

```
train_dpo.py
 ├─ model/model_minimind.py         ← policy 和 ref 共用的模型定义
 ├─ dataset/lm_dataset.py            ← DPODataset（偏好数据加载 + loss mask）
 ├─ trainer/trainer_utils.py         ← 工具函数
 └─ plan/train_lora_study_plan.md   ← 前置学习：LoRA 微调
```
