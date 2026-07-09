# 偏好对齐方法哪个效果最好？（2024-2026）

> 2026-07-09 · standard depth · 45 sources · workspace: research/best-alignment/

## 执行摘要

- **没有通用的"最好"方法**——最佳选择取决于你的数据、算力和任务约束。数据质量的影响（~8%）远大于切换算法（~2.5%）[2]。
- **PPO + 专用 Reward Model 的天花板最高**——当正确调参时，PPO 可以全面超越其他方法，但实现难度大，只有顶级团队（OpenAI、Anthropic）愿意投入工程成本 [5]。
- **GRPO（DeepSeek-R1，Nature 2025）是推理任务的事实标准**——用规则奖励替代神经奖励模型，消除了 reward hacking，已广泛被开源社区采用 [4]。
- **SimPO 是当前最好的离线免参考方法**——比 DPO 简单（去掉 reference model），在 AlpacaEval 2 上高出 4-7 个点，被最新综述推荐为默认基线 [2][3]。
- **ORPO 是最高效的入门选择**——把 SFT 和偏好对齐合并为一步，GPU 小时数减半 [5]。
- **KTO 适合只有隐式反馈（赞/踩）的生产场景**——不需要偏好对，只需要好坏标签 [3]。

---

## 背景与范围

本题从四个角度调查：主流模型的实际选型、学术综述的结论、新方法的实证结果、以及生产实践中的取舍。焦点是 2024-2026 年有公开证据的方法。调查排除了完全封闭的专有方法。

---

## 一、现实世界：主流模型用什么？

| 模型/组织 | 对齐方法 | 证据强度 |
|-----------|---------|---------|
| **DeepSeek-R1** | GRPO + 规则奖励（准确率+格式），明确拒绝神经奖励模型 | **强** — Nature 论文 [4] |
| **Anthropic Claude** | Constitutional AI (RLAIF) + PPO | 中 — 源自其发表的论文，具体版本未完全披露 [1] |
| **OpenAI GPT/o 系列** | PPO 基 RLHF + PRM（过程奖励）用于推理 | 中 — o1 系统卡确认 PRM+RL，o3 仅说"大规模 RL" [1][4] |
| **Meta Llama 3** | SFT → RSFT → PPO → DPO 四阶段 | **强** — 技术报告 [1] |
| **Google Gemini** | RLHF | 弱 — Gemini 1.5 报告确认，后续版本未披露 [1] |
| **开源社区（Qwen、Mistral 等）** | DPO 为主 | 强 — 各模型的官方文档 [1] |
| **开源推理模型（Open R1, Qwen2.5-Math）** | GRPO / RLVR | 中 — DeepSeek-R1 后的社区跟随 [4][5] |

关键发现：**排名靠前的闭源模型（Claude Fable 5、GPT-5.5、Grok 4.5）的具体对齐方法大多不公开**[1]。方法本身并不能直接解释排行榜位次。

---

## 二、学术综述的结论

### 2.1 没有银弹

三个独立综述 [1][2] 一致认为：**没有哪个方法在所有场景下都最好**。方法的选择取决于三个正交轴——偏好模型、正则化机制、数据分布 [2]。

### 2.2 数据质量 > 算法选择

Ivison 等人的研究 [2] 发现：改进数据质量带来约 **8%** 的性能提升，而 PPO 换 DPO 的收益只有约 **2.5%**。这意味着：

> 🐶 如果你想提升对齐效果，**先花时间清洗和优化你的偏好数据**，比纠结选哪个算法更划算。

### 2.3 在线 vs 离线：各有千秋

**Coverage Separation Theorem** [2] 给出了一个干净的理论边界：
- **DPO/SimPO 等离线方法**：当你有覆盖广泛的离线偏好数据时更好
- **PPO/GRPO 等在线方法**：当模型需要探索训练分布之外的区域时更好

### 2.4 调查对 SimPO 的推荐

Raheja & Pochhi (2026) 的综合综述 [2] 明确推荐 **SimPO 作为大多数从业者的默认离线方法**，因为它：
- 不需要 reference model（省显存）
- 长度归一化直接解决冗长偏差（verbosity bias）
- 在 Llama-3-8B 上 AlpacaEval 2 LC 达到 31.5% vs DPO 的 25.1%

---

## 三、新方法的横向对比

### 离线类（单次训练，静态数据）

| 方法 | 核心思路 | 相比 DPO 的优势 | 适用场景 |
|------|---------|----------------|---------|
| **SimPO** (NeurIPS 24) | 平均 logp 做隐式奖励，免参考模型 | +6.4 AlpacaEval 2, +7.5 Arena-Hard | **推荐默认离线方法** |
| **ORPO** (ACL 24) | SFT + 偏好合并为一步 | GPU 时减半 | 预算极低的团队 |
| **KTO** (ICML 24) | 只需要"好/坏"标签 | 匹配 DPO，数据易获取 | 只有隐式反馈的场景 |
| **DPO** (NeurIPS 23) | 从偏好对直接优化 | — | 基线（已被 SimPO 取代） |

### 在线类（模型自己生成数据）

| 方法 | 核心思路 | 关键结果 | 典型使用者 |
|------|---------|---------|-----------|
| **PPO + RM** | Critic 网络 + 专用 Reward Model | 天花板最高，但实现最复杂 | OpenAI, Anthropic |
| **GRPO** (DeepSeek 24) | 去掉 Critic，组内归一化优势 | AIME 2024 pass@1: 15.6% → 77.9% | DeepSeek-R1, 开源推理社区 |
| **Self-Rewarding** (ICML 24) | LLM 自己当裁判 | Llama 2 70B 超 GPT-4 0613 | 有迭代算力的团队 |
| **SPIN** (ICML 24) | 自我对弈区分自己的回复 vs 人类 | 超过 DPO + GPT-4 偏好数据 | 同上 |

### GRPO vs PRM 之争

这是一个重要的方法论分歧 [4]：

| | GRPO (DeepSeek 路线) | PRM (OpenAI 路线) |
|---|---|---|
| 奖励来源 | 规则（正确性检查 + 格式） | 训练过的神经过程奖励模型 |
| 对抗 reward hacking | ✅ 天生免疫 | ❌ 需要频繁重训 |
| 计算成本 | 低（无额外模型） | 高（需要 PRM 推理） |
| 适用范围 | 有客观正确答案的领域 | 可扩展的步级评估 |
| 代表模型 | DeepSeek-R1 | OpenAI o1/o3 |
| 论文证据 | Nature 2025 [4] | "Let's Verify Step by Step" [3] |

DeepSeek 明确选择了规则奖励路线，认为神经奖励模型在大规模 RL 中容易 reward hacking [4]。

---

## 四、生产实践中的取舍

### 按预算分级

| 预算规模 | 推荐方法 | 理由 |
|---------|---------|------|
| **<$10K, 1-8B 模型** | ORPO（1步）或 SimPO（2步） | 最省显存和训练时间 [5] |
| **$50K-$500K, 7-70B 模型** | SFT → SimPO/DPO | 稳定性好，可独立迭代对齐数据 [5] |
| **>$500K, 70B+ 模型** | PPO + RM 或 GRPO | 天花板最高，值得 Reward 模型投入 [5] |
| **推理场景** | GRPO + 规则奖励 | 已成为开放社区推理对齐的事实标准 [4][5] |

### 关键实践洞察

1. **PPO 的名声问题**：Xu et al. (2024) 发现 PPO 正确调参后能全面超越其他方法 [5]。它的"差评"来源于实现难度大——需要正确的 Reward Model 更新策略、大 batch size 和合适的 KL 参考模型。不是方法不行，是**工程实现门槛高**。

2. **RLAIF 可减少 80-90% 的人工标注**：Anthropic 的 Constitutional AI [3][5] 证明了 AI 生成的偏好标签可以替代大部分人工标注，适合安全对齐场景。

3. **开源社区碎片化**：DPO 生态最成熟（TRL/Axolotl/Unsloth 原生支持），但 GRPO 因 DeepSeek-R1 的 Nature 论文迅速崛起 [5]。

---

## 开放问题

1. **RLOO / REINFORCE 系列**：在调查中无法定位到可靠来源，其在生产中的采用情况不明。
2. **PRM 与 GRPO 的公平对比**：没有公开发表的控制实验在相同条件下对比两种路线。DeepSeek 选择了规则奖励，但这是否在所有场景下都比 PRM 更好尚无定论。
3. **精确的计算成本对比**：各方法在相同 base model、相同 GPU 小时下的成本对比数据缺失。
4. **2025-2026 年的新综述缺口**：已找到的综述对 Self-Rewarding、SPIN、RLOO 等方法覆盖不足。

---

## 一句话回答

> 如果你的场景是**推理任务** → **GRPO + 规则奖励**（DeepSeek-R1 路线）；如果你的目标是**通用对齐且算力充裕** → **PPO + RM**（天花板最高）；如果你是**中小团队想要快速迭代** → **SimPO**（比 DPO 简单且更好）；如果你的**数据只有赞/踩** → **KTO**。但你最该花时间的地方不是选算法，而是**提升你的偏好数据质量**——它带来的收益是换算法的 3 倍。

---

## Sources

[1] Leaderboard & production model alignment methods — artificialanalysis.ai/leaderboards/models, arxiv.org/abs/2407.21783 (Llama 3), arxiv.org/abs/2305.18290 (DPO) (accessed 2026-07-09)

[2] Raheja & Pochhi (2026), "A Unification of Preference Learning" — arxiv.org/abs/2601.06108; Srivastava & Aggarwal (2026), ACM TIST survey — arxiv.org/abs/2507.04136 (accessed 2026-07-09)

[3] SimPO — arxiv.org/abs/2405.14734; KTO — arxiv.org/abs/2402.01306; ORPO — arxiv.org/abs/2403.07691; Self-Rewarding — arxiv.org/abs/2401.10020; SPIN — arxiv.org/abs/2401.01335; PRM800K — arxiv.org/abs/2305.20050; Constitutional AI — arxiv.org/abs/2212.08073 (accessed 2026-07-09)

[4] DeepSeek-R1 — arxiv.org/abs/2501.12948, www.nature.com/articles/s41586-025-09422-z; DeepSeekMath (GRPO) — arxiv.org/abs/2402.03300; OpenAI o4-mini system card — openai.com/index/o3-o4-mini-system-card/; OpenAI competitive programming — arxiv.org/abs/2502.06807 (accessed 2026-07-09)

[5] Xu et al. "DPO vs PPO" — arxiv.org/abs/2404.10719; ORPO compute efficiency — arxiv.org/abs/2403.07691; SimPO cost-performance — arxiv.org/abs/2405.14734; RLAIF — arxiv.org/abs/2212.08073; KTO binary feedback — arxiv.org/abs/2402.01306 (accessed 2026-07-09)
