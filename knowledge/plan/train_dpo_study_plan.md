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

## 四、学习目标检查清单（含解析）

- [x] 理解 DPO 的核心思想（直接偏好优化，无需奖励模型）
  DPO 跳过 RLHF 的 Reward Model + PPO 两阶段，直接利用偏好对 (chosen, rejected)
  通过对比 policy 和 ref 的 log_probs 差异来计算 loss，无需训练额外的评分模型。

- [x] 理解 DPO loss 的公式和计算过程
  loss = -log σ(β × (优势_chosen - 优势_rejected))
  其中优势 = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
  即：policy 比 ref 额外多偏好 chosen 的程度。

- [x] 理解 policy 模型和 reference 模型的双模型架构
  policy：要训练的模型，参数不断更新。
  ref：从相同初始权重加载，完全冻结，作为衡量"偏离了多少"的基线标尺。

- [x] 理解 ref 模型为什么需要冻结以及初始权重来源
  初始权重 = 与 policy 完全相同的 SFT checkpoint（--from_weight）。
  冻结原因：ref 必须是一个固定标尺，如果它也更新，就不知道"相对于谁"比较了。

- [x] 理解 `logits_to_log_probs` 的实现原理 ← ⭐ 详细讲解
  logits 是模型输出的原始分数（未归一化），shape = (B, L, V)。
  要得到"每个 token 在其真实 label 上的概率的对数"，分两步：
    ① log_softmax(logits, dim=2)：
       在 vocab 维做 log_softmax，公式：log_softmax(x_i) = x_i - log(Σ_j e^{x_j})
       结果 shape = (B, L, V)，每个位置的值表示"第 i 个 token 是 vocab 中第 j 个词
       的 log probability"。
    ② gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)：
       labels shape = (B, L)，内容是每个位置的真实 token ID。
       unsqueeze(2) 变成 (B, L, 1)，然后 gather 在 dim=2 上取 index 对应位置的值。
       通俗说：从 vocab 维度"捞出"真实 label 对应的那个 log_prob。
       结果 shape = (B, L)，每个元素表示"第 i 个 token 在其真实 label 上的 log_prob"。
  与 SFT 的 CrossEntropyLoss 对比：
    SFT 内部做了 log_softmax + nll_loss，你拿不到中间的 log_probs。
    DPO 需要显式拿到 log_probs 才能在 chosen 和 rejected 之间做差值计算。

- [x] 理解 DPO 数据集中 chosen / rejected 的拼接与分拆
  每一条数据包含 chosen（好回答）和 rejected（差回答）各一组序列。
  训练时 torch.cat([chosen, rejected], dim=0) 拼成大 batch：
    前半 batch = chosen，后半 batch = rejected。
  dpo_loss 内部按 batch_size // 2 分拆计算。

- [x] 理解为什么 DPO 的 learning rate 极小（4e-8 级别）← ⭐ 详细讲解
  对比三个阶段的 LR：
    SFT（全参）：1e-6  — 模型从预训练权重出发，学习对话格式
    LoRA：1e-4         — LoRA 矩阵从零初始化，需要大步长
    DPO：4e-8          — 模型已经 SFT 训练好了，只会微调偏好方向
  DPO 是在 SFT 已经训好的模型上做"微调中的微调"。
  模型已经能很好地回答问题了，只需微调偏好方向的权重，
  学习率稍大一点（哪怕 1e-7）就会破坏 SFT 学到的知识（灾难性遗忘）。
  类比：SFT 是学写正楷，LoRA 是练特定字体，DPO 是调整笔锋角度——
  DPO 调整幅度最小，所以步长也应该最小。

- [x] 对比 DPO 和 SFT 的 loss、数据、训练目标差异
  | 维度 | SFT | DPO |
  |------|-----|-----|
  | loss | -log π(y|x) | -log σ(β × 相对优势) |
  | 数据 | (prompt, answer) 单条 | (chosen, rejected) 偏好对 |
  | 目标 | 增大 token 概率 | 拉开 chosen/rejected 的差距 |
  | 模型数 | 1 个 | 2 个（policy + ref）|
  | LR | 1e-6 | 4e-8 |
  | 关注点 | 模仿回答格式 | 区分好坏回答 |

- [x] 理解 DPODataset 的 `generate_loss_mask` 如何处理 prompt/response 边界
  扫描 input_ids 中的 bos_id（"<|bos|>assistant\n" 的 token 序列），
  从 bos_id 之后到下一个 eos_id 之前标记为 1（有效），其余位置为 0。
  与 SFT 的 generate_labels 本质相同（都是 mask 掉 prompt），
  但 SFT 标 -100（被 CrossEntropyLoss 忽略），DPO 标 0/1（手动乘以 mask）。

- [x] 理解 DPO 和 GRPO / PPO 的核心区别（无 reward model、无 critic）
  PPO（Proximal Policy Optimization）：需要 4 个模型（policy, ref, reward, critic），
    流程：生成回答 → Reward Model 打分 → Critic 预估优势值 → PPO clip 更新。
    问题：超参敏感，训练不稳定，显存占用极大。
  GRPO（Group Relative Policy Optimization）详细讲解：
    GRPO = PPO 去掉 Critic 模型 + 用"组内比较"代替 Critic 的预估价值。

    具体怎么做（看 train_grpo.py 的代码）：
      1. 给每个 prompt，让 policy 生成多个回答（num_generations=8）
         比如 prompt="1+1=?"，生成 8 个回答：[2, 2, 3, 2, 1, 2, 4, 2]
      
      2. 用 Reward Model 给每个回答打分
         得到 8 个分数：[1.0, 1.0, 0.3, 1.0, 0.0, 1.0, 0.0, 1.0]
         （回答"2"的得 1.0，回答"3"的得 0.3，回答"1"和"4"的得 0.0）

      3. 按 prompt 分组：grouped_rewards = rewards.view(-1, num_generations)
         得到 [1.0, 1.0, 0.3, 1.0, 0.0, 1.0, 0.0, 1.0]（就是一个 group）
      
      4. 在组内算均值和标准差：
         均值 = (1.0+1.0+0.3+1.0+0.0+1.0+0.0+1.0) / 8 = 0.6625
         标准差 ≈ 0.44
      
      5. 计算组内相对优势（核心创新点）：
         advantage = (reward - 组内均值) / 组内标准差
         回答"2"的 advantage = (1.0 - 0.6625) / 0.44 ≈ +0.76 ✓（比平均好）
         回答"3"的 advantage = (0.3 - 0.6625) / 0.44 ≈ -0.82 ✗（比平均差）
         回答"1"的 advantage = (0.0 - 0.6625) / 0.44 ≈ -1.50 ✗（比平均差很多）

      6. 用这个 advantage 更新 policy：
         让 advantage > 0 的回答概率增大
         让 advantage < 0 的回答概率减小

    GRPO 相比 PPO 的优势：
      不需要 Critic 模型 → 少维护一个模型，省显存
      组内比较天然解决了"奖励尺度不一致"的问题
         （不同 prompt 的得分范围可能不同，但组内归一化后都在同一量纲）
    
    GRPO 相比 DPO 的劣势：
      仍然需要 Reward Model（DPO 完全不需要）
      需要在训练时实时生成多个回答，推理开销大
      如果 num_generations 太小，组内统计不准确

    总结三个方法的递进关系：
      PPO   →   GRPO   →   DPO
      (4模型)   (3模型)    (2模型)
      最复杂    省掉Critic  省掉Reward Model
                    ↓           ↓
              用组内比较代替  用偏好对直接计算
  DPO（Direct Preference Optimization）：三者中最简洁，
    只需要 policy + ref 两个模型，无 Reward Model、无 Critic、无组内比较。
    直接从偏好对 (chosen, rejected) 推导出 loss，一步到位。
  
  用造楼来类比三者的复杂度：
    PPO = 设计院（policy）+ 监理（ref）+ 质检局（reward）+ 造价师（critic）
    GRPO = 设计院 + 监理 + 质检局（去掉造价师，用同行评议代替）
    DPO = 设计院 + 监理（直接告诉工人哪个方案好）

  用一个相同的例子（prompt="1+1=?"）对比三个方法的具体流程：

  ╔══════════════════════════════════════════════════════════════╗
  ║  PPO（4个模型）                                              ║
  ║  权重关系：                                                  ║
  ║    ① Policy ← 待训练的模型，从 SFT 权重初始化，不断更新      ║
  ║    ② Ref    ← Policy 的初始副本，冻结（≠PPO 里叫 ref）       ║
  ║    ③ Reward ← 独立训练的评分模型（额外阶段训练的，不一定      ║
  ║               与 policy 共享任何权重）                        ║
  ║    ④ Critic ← 可与 Policy 共享主干 + 独立价值头，             ║
  ║               价值头随机初始化，从零开始训练预估"未来得分"     ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ① Policy 模型生成回答："2"                                  ║
  ║  ② Reward Model 打分：满分 1.0，回答"2"合理 → 0.8 分        ║
  ║  ③ Ref 模型算 KL 散度：新策略离旧策略多远 → penalty = 0.05   ║
  ║  ④ 实际优势值 = 0.8 - 0.05 = 0.75                           ║
  ║  ⑤ Critic 模型预估优势值：看到"1+1=?"估 0.5                 ║
  ║  ⑥ 差异 = 0.75 - 0.50 = 0.25（更新方向），clip 限制步长      ║
  ║  ⑦ 更新 Policy，同时更新 Critic（让它下次估得更准）           ║
  ║  流程复杂，每一步都可能出错，Critic 还要额外训练。            ║
  ╚══════════════════════════════════════════════════════════════╝

  ╔══════════════════════════════════════════════════════════════╗
  ║  GRPO（3个模型）                                             ║
  ║  权重关系：                                                  ║
  ║    ① Policy ← 待训练的模型，从 SFT 权重初始化，不断更新      ║
  ║    ② Ref    ← Policy 的初始副本，冻结                        ║
  ║    ③ Reward ← 独立训练的评分模型，冻结                        ║
  ║    无 Critic！用"组内比较"代替 Critic 的价值预估             ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ① Policy 模型生成 8 个回答：["2","2","3","2","1","2","4","2"] ║
  ║  ② Reward Model 打分：[1.0, 1.0, 0.3, 1.0, 0.0, 1.0, 0.0, 1.0]║
  ║  ③ 组内均值 = 0.6625，组内标准差 ≈ 0.44                     ║
  ║  ④ 相对优势 = (reward - 均值) / 标准差                        ║
  ║     回答"2"：(1.0-0.6625)/0.44 = +0.76 → 增大概率 ✓         ║
  ║     回答"3"：(0.3-0.6625)/0.44 = -0.82 → 减小概率 ✗         ║
  ║  ⑤ 用相对优势更新 Policy                                     ║
  ║  省掉 Critic，组内比较天然解决"打分量纲不一致"问题。           ║
  ║  仍需 Reward Model，且每步要生成多个回答，推理开销大。         ║
  ╚══════════════════════════════════════════════════════════════╝

  ╔══════════════════════════════════════════════════════════════╗
  ║  DPO（2个模型）                                              ║
  ║  权重关系：                                                  ║
  ║    ① Policy ← 待训练的模型，从 SFT 权重初始化，不断更新      ║
  ║    ② Ref    ← Policy 的初始副本（从相同 --from_weight 加载）， ║
  ║               冻结，作为衡量"偏离了多少"的基线                 ║
  ║    无 Reward Model！无 Critic！直接从偏好对计算 loss          ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ① 取一条偏好数据：chosen="2"（好回答），rejected="3"（差回答） ║
  ║  ② Policy 和 Ref 分别算 log_probs：                          ║
  ║     Policy: 对 chosen="2" 的 log_prob = -0.15               ║
  ║     Policy: 对 rejected="3" 的 log_prob = -0.50             ║
  ║     Ref: 对 chosen="2" 的 log_prob = -0.20                  ║
  ║     Ref: 对 rejected="3" 的 log_prob = -0.45                ║
  ║  ③ 相对优势 = (-0.15+0.50) - (-0.20+0.45) = 0.35-0.25=0.10 ║
  ║  ④ loss = -log σ(0.1 × 0.10) ≈ 0.69                        ║
  ║  ⑤ 反向传播更新 Policy                                      ║
  ║  最简洁。无需生成回答，无需打分，直接用偏好数据计算 loss。     ║
  ║  缺点：依赖现成的偏好数据对，没有探索新策略的能力。            ║
  ╚══════════════════════════════════════════════════════════════╝

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

## 六、自测题（含答案）

### 基础题
1. DPO 为什么需要两个模型（policy 和 ref）？ref 模型为什么必须冻结？

   **答案**：需要对比"训练前"和"训练后"对 chosen/rejected 的偏好差异。
   ref 是 policy 的初始副本，必须冻结——如果 ref 也在变，就不知道"相对于谁"比较了。

2. dpo_loss 中 `batch_size // 2` 的假设是什么？如果 batch_size 是奇数会怎样？

   **答案**：假设 batch 前半是 chosen、后半是 rejected，且两者数量相等。
   如果 batch_size 是奇数，分半后长度不一致，下标会出错或漏掉一条数据。
   所以 batch_size 必须为偶数（实际代码中 chosen+rejected 成对出现，天然偶数）。

3. DPO 的学习率为什么远比 SFT 小？

   **答案**：DPO 是在 SFT 训好的模型上做"微调中的微调"。模型已经能很好地回答问题了，
   只需微调偏好方向。LR 再大一点就会破坏 SFT 学到的知识（灾难性遗忘）。
   对比：SFT=1e-6（学对话格式），LoRA=1e-4（从零初始化），DPO=4e-8（微调偏好）。

### 进阶题
4. 如果把 ref 模型也设为可训练（requires_grad=True），loss 会怎样变化？

   **答案**：ref 变成可训练后，它与 policy 会互相追逐（ref 也在向 chosen 偏移），
   相对优势 = policy_chosen - policy_rejected - (ref_chosen - ref_rejected) 会趋近 0，
   loss 无法有效下降，模型学不到任何偏好。这就是"标尺不能动"的原因。

5. DPO 的 beta 参数增大或减小分别会有什么影响？

   **答案**：β 控制"拉开 chosen/rejected 差距的力度"：
   - β 增大 → σ(β × advantage) 曲线变陡 → 对同样的差距更敏感 → 训练可能更快但容易不稳定
   - β 减小 → 曲线变平缓 → 同等差距的 loss 更小 → 训练更稳定但收敛变慢
   - β=0 → loss = -log σ(0) = -log(0.5) = 0.693（恒定值，学不到任何东西）

6. 对比 DPO loss 和 RLHF 中 PPO loss 的核心差异。

   **答案**：
   PPO loss = -E[min(ratio×A, clip(ratio, 1-ε, 1+ε)×A)] + 额外 Critic MSE loss + KL penalty
     需要 4 个模型，先算优势值（Reward + KL）、再 clip、再更新
   DPO loss = -log σ(β × (优势_policy - 优势_ref))
     只需 2 个模型，直接从偏好对的 log_probs 差异算 loss，一步到位
   PPO 是"走一步看一下"（on-policy），DPO 是"看别人经验直接学"（off-policy）。

### 深入题
7. DPODataset 的 `generate_loss_mask` 如何区分 chosen 和 rejected 中的 prompt 和 response 区域？
   和 SFTDataset 的 `generate_labels` 有什么异同？

   **答案**：扫描 input_ids 中的 bos_id（"<|bos|>assistant\n" 的 token 序列），
   从 bos_id 之后到下一个 eos_id 之前标记为 1（有效），其余位置为 0。
   与 SFT 的 generate_labels 本质相同（都是 mask 掉 prompt 区域），
   区别：SFT 标 -100 被 CrossEntropyLoss 忽略，DPO 标 0/1 手动乘以 mask。

8. DPO 训练中直接合并两个 batch（chosen+rejected）前向一次和分别前向两次各有什么优劣？

   **答案**：
   合并一次 forward：✓ 节省一次前向时间，loss 函数按 batch 分半简洁 ✗ 显存翻倍
   分两次 forward：✓ 显存减半 ✗ 多一次前向时间，需分别收集 log_probs 再手动计算 loss
   当前实现选"一次 forward"是显存换速度。

9. 如果用 LoRA 做 DPO，ref 模型是否需要也注入 LoRA？为什么？

   **答案**：不需要。ref 是冻结的参照物，它的输出作为基线衡量 policy 的偏离程度。
   如果给 ref 也注入 LoRA，ref 本身也会变化，标尺就变了，相对优势计算失去意义。
   ref 保持纯粹的 SFT 权重即可。

---

## 七、关联文件

```
train_dpo.py
 ├─ scripts/Model/model_minimind.py         ← policy 和 ref 共用的模型定义
 ├─ dataset/lm_dataset.py            ← DPODataset（偏好数据加载 + loss mask）
 ├─ scripts/Trainer/trainer_utils.py         ← 工具函数
 └─ plan/train_lora_study_plan.md   ← 前置学习：LoRA 微调
```
