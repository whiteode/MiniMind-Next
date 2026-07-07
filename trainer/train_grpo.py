import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')

#全部奖励分数的计算入口。
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    整合所有奖励函数，计算每条生成回复的总奖励。

    GRPO 不使用 critic 网络，而是用这些规则 + reward model 的评分来构造优势函数。
    奖励由两部分组成：
      1) 格式奖励 — 针对推理模型的 <think>/<answer> 标签格式进行打分；
      2) Reward Model 打分 — 用外部 reward model 评估对话质量（可混合 answer-only 评分）。

    Args:
        prompts: list[str], 长度为 B（batch 中的 prompt 数量）
        responses: list[str], 长度为 B * num_generations。
                  GRPO 的核心设计：每个 prompt 生成 num_generations 条回复，构成一个 group。
                  代码中 group 是这样实现的：
                    - 生成阶段：`generate(num_return_sequences=args.num_generations)` 会为每个 prompt
                      采样多条回复；HuggingFace 会把同一个 prompt 在 batch 维复制 num_generations 次，
                      并保证这些复制样本连续排列，因此输出 shape 为 [B * num_generations, P + R]。
                      这里利用了 HuggingFace 的 `num_return_sequences` 机制：
                        输入 [B, P] → 对每条 prompt 独立采样 num_generations 次 → 输出 [B * num_generations, P+R]
                      同一 prompt 的多次采样在输出张量中连续存储，后续 `view(-1, num_generations)` 才能正确分组。
                    - 归一化阶段：`rewards.view(-1, args.num_generations)` 把一维 rewards 重塑为
                      [B, num_generations]，每一行正好对应一个 prompt 的 num_generations 条回复，
                      随后组内做 mean/std 标准化，从而构造出 group-based advantages。
                  后续在 group 内部对奖励做归一化（减均值、除标准差）来构造优势函数，
                  从而替代 PPO 中的 critic 网络（value function）。因此输入维度是 B 个 prompt
                  各对应 num_generations 条回复，总数为 B * num_generations。
        reward_model: 奖励模型（可调用 get_score）
        reward_tokenizer: 奖励模型的 tokenizer
                  两者必须配对使用：reward_tokenizer 负责把对话文本转换成 reward_model 能处理的
                  token id 和 attention mask；reward_model 再基于这些输入输出标量奖励分。
                  HuggingFace 的模型只负责前向计算，不理解原始文本，
                  所以必须由 tokenizer 先完成文本编码。

    Returns:
        rewards: Tensor, shape [B * num_generations], 每条回复的最终奖励
    """
    def reasoning_model_reward(rewards):
        """
        推理格式奖励函数。

        奖励两个层面：
        (a) 整体正则匹配 — 严格匹配 <think>...</think> 和 <answer>...</answer> 的完整结构，
            匹配成功加 0.5, 否则 0。
        (b) 标签计数 — 检查四个标签各自恰好出现一次，每个标签 0.25 分，共 1 分。

        两部分合计最高 1.5 分。
        """
        # 两种换行允许格式：<think>\\n...\\n</think>\\n<answer>\\n...\\n</answer>
        # 以及中间多一个换行的情况
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            """逐一检查四个标签是否各出现一次，每个标签 0.25 分"""
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        # scale: 对 Reward Model 输出分数的裁剪范围 [-scale, scale]。
        # 这与格式奖励（最高 1.5 分）是独立的两个加分项：
        #   最终奖励 = 格式奖励(0~1.5) + RM 分数(裁剪到 [-3, 3])
        # 裁剪只限制 RM 分数，避免极端值主导训练，不影响格式奖励的计算。
        scale = 3.0

        # 遍历每个 prompt 及其 num_generations 条回复
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 解析 prompt 中的多轮对话格式 <|im_start|>role content<|im_end|>
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                # 构造完整对话用于 reward model 评分
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale)

                if args.reasoning == 1:
                    # 推理模式下：额外用 answer 标签内的纯答案部分评分，然后加权融合。
                    # 为什么这样做：
                    #   reward model 本质上是一个"对话质量/正确性"打分器，它不看格式标签，
                    #   只看文本内容。对完整回复打分时，分数里既包含了推理过程质量，
                    #   也包含了最终答案质量；但训练更关心"答案是否正确"。
                    #   因此这里再单独提取 <answer>...</answer> 里的纯答案文本，
                    #   用 reward model 再打一次分，得到一个更聚焦于答案正确性的分数。
                    #   然后用 0.4:0.6 加权融合：
                    #     - 完整回复占 40%，保留对推理过程的激励；
                    #     - 纯答案占 60%，更强地引导模型给出正确答案。
                    #   Reward Model 本质上是一个语言模型 + 一个评分头，它对输入格式不敏感 —
                    #   给它任何对话文本，它都输出一个标量分数衡量"有用性/正确性"。
                    #   因此它既可以给带 <think> 标签的完整推理过程打分，也可以给纯答案打分，
                    #   两者分数含义一致，只是评估的文本范围不同。
                    #   通过 0.4:0.6 的加权，引导模型同时优化推理过程和答案正确性。
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6  # 完整回复占 40%，纯答案占 60%

                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, start_step=0, wandb=None):
    """
    执行一个 epoch 的 GRPO 训练。

    GRPO (Group Relative Policy Optimization) 是 DeepSeek 提出的强化学习算法，
    与 PPO 的主要区别：
      - 不使用 critic 网络（value function）。
        在 PPO 中，critic 网络是一个与 policy 模型共享 backbone 的附加模块（通常
        在最后一层加一个线性头输出标量 value），用于估计每个 state 的期望累积奖励，
        作为优势函数计算时的 baseline。GRPO 完全去掉这个网络，改为用 group 内
        的 reward 均值做 baseline。
      - 对同一个 prompt 采样多个回复构成 group，用 group 内部的 reward 均值/标准差
        来归一化，构造优势函数（无需额外的 value model）。
      - 用 KL 散度惩罚约束策略更新，防止策略偏离 reference 模型太远。
        Reference 模型是训练开始时 policy 模型权重的冻结副本。没有 KL 惩罚时，
        强化学习会导致 policy 模型在 reward 高的区域快速"坍缩"——它会过度优化
        少数高奖励的 token 模式，变得单一化（失去多样性），甚至出现 reward hacking
        （学到利用 reward model 的漏洞而非真正的能力）。KL 惩罚相当于一个"安全绳"：
          ① 每一项更新都受 β × KL(π_θ || π_ref) 约束，迫使 π_θ 不能偏离 π_ref 太远；
          ② 保留了模型的通用语言能力（SFT 阶段学到的知识不会在 RL 阶段被遗忘）；
          ③ 形成一个天然的 exploration 边界——在低奖励区域 π_θ 会被拉回 π_ref，
             不会盲目探索到 OOD 区域。

        KL 惩罚的直觉（用具体数据说明）：
        假设在某个 token 位置上，reference 模型给出的概率分布为：
          token_A: 0.7, token_B: 0.2, token_C: 0.1
        policy 模型经过 RL 更新后变成：
          token_A: 0.95, token_B: 0.04, token_C: 0.01  （过度自信）

        "坍缩"的过程：
          第 1 轮：样本中 token_A 被 reward model 给了高分 → 这条回复获得高 reward → 对应 advantage 为正。
                  GRPO loss 里，advantage 前面的系数是 exp(per_token_logps - per_token_logps.detach())，
                  当 advantage > 0 时，梯度方向会让模型提高这些 token 的 log-probability；
                  反之 advantage < 0 时，会降低。于是出现高分的 token_A 会被反复加强。
          第 2 轮：token_A 概率变高 → 更容易被采样到 → 继续被 reward 强化
          第 n 轮：token_A 概率逼近 1.0，其他 token 概率接近 0，模型丧失多样性

        KL 散度的计算（阻止这种坍缩）：
          KL(π_θ || π_ref) = Σ_{t ∈ vocab} π_θ(t) × log(π_θ(t) / π_ref(t))
                              ^^^^^^^^^   ^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
                                  │          │               │
                                  │          │               └─ "惊讶度"：π_θ 认为 t 的概率
                                  │          │                  相对于 π_ref 大了/小了多少倍
                                  │          │                  正值 = π_θ 比 π_ref 更置信
                                  │          │                  负值 = π_θ 比 π_ref 更不置信
                                  │          │
                                  │          └─ 权重 π_θ(t)：来自 KL 散度的定义式
                                  │             KL(π_θ || π_ref) = E_{x~π_θ}[log(π_θ(x)/π_ref(x))]
                                  │             即"按 π_θ 的分布对每个 token 的 log-ratio 求期望"。
                                  │             π_θ 越倾向于选 t，该位置在期望中的权重越大。
                                  │
                                  └─ 对所有可能的 token 求和，得到分布级别的差距

          具体计算（接上例）：
            token_A: π_θ=0.95, π_ref=0.70
              w = 0.95 × log(0.95/0.70) = 0.95 × 0.305 = 0.290
              ↑ π_θ 给 A 的概率大幅增加了（0.70→0.95），产生了正贡献

            token_B: π_θ=0.04, π_ref=0.20
              w = 0.04 × log(0.04/0.20) = 0.04 × (-1.609) = -0.064
              ↑ π_θ 给 B 的概率减少了（0.20→0.04），"惊讶度"为负

            token_C: π_θ=0.01, π_ref=0.10
              w = 0.01 × log(0.01/0.10) = 0.01 × (-2.303) = -0.023
              ↑ π_θ 给 C 的概率也减少了（0.10→0.01）

            Σ = 0.290 + (-0.064) + (-0.023) = 0.203

          为什么 KL 散度 ≥ 0？用一个比喻：

          假设 π_ref 是一个班的期中考试成绩分布，π_θ 是期末考试成绩分布。
          某个分数段的学生人数发生了变化：

            A 段（优秀）：从 70% 涨到 95%（"赢家"）
            B 段（中等）：从 20% 跌到 4%（"输家"）
            C 段（及格）：从 10% 跌到 1%（"输家"）

          KL 散度相当于全校大会的"喧闹程度"：
            "赢家"的欢呼声 = 0.95 × log(0.95/0.70) ≈ 0.29  — 人多 + 高兴 = 声音大
            "输家"的抱怨声 = 0.04 × log(0.04/0.20) ≈ -0.06 — 人少 + 不爽 = 声音小
            "输家"的抱怨声 = 0.01 × log(0.01/0.10) ≈ -0.02 — 人极少 + 不爽 = 几乎听不见

          总喧闹度 = 0.29 + (-0.06) + (-0.02) = 0.21 > 0

          关键：因为概率总和为 1，"赢家"增加了多少概率，"输家"就必须减少多少。
          但"赢家"现在人多了（π_θ 大），所以其正贡献被放大；
          "输家"人少了（π_θ 小），所以其负贡献被压制。
          结果总是正的。唯一不喧闹的情况是 π_θ = π_ref（成绩没变化，没人欢喜没人愁）。

        这个 0.203 会乘以 β 加到 loss 中。如果 β=0.02，KL 惩罚项 = 0.00406。
        这意味着：要让 token_A 从 0.7→0.95，模型必须付出 0.00406 的额外 loss，
        除非这种偏移带来的奖励增益超过这个代价，否则优化器不会采纳这个更新方向。
        这就起到了"安全绳"的作用——只允许对奖励增益足够大的方向做偏移。

        用一个比喻理解"安全绳"：
          想象你是一个基金经理（policy 模型），你的投资组合（token 概率分布）一开始是
          保守配置（π_ref）：70% 债券 + 20% 股票 + 10% 现金。

          最近 AI 概念大涨（reward model 给 token_A 高分），你想把组合调整成
          95% AI 股票 + 4% 债券 + 1% 现金（π_θ）。

          KL 惩罚就是公司风控部门的规定：
            "每偏离原配置一个单位，需要缴纳 β×KL 的合规成本。
             只有当预期收益大于合规成本时，才批准调整。"

          这里的"预期收益"对应 GRPO loss 中的 advantages（优势函数值）。
            在代码中：per_token_loss = -(exp(Δ) × advantages - β × per_token_kl)
            advantages 是 group 归一化后的奖励：正数表示这条回复比 group 平均好，
            负数表示比 group 平均差。advantages=+0.01 意味着"这条回复比平均好一点点"。

          注意这里的判断条件：
            代码定义 per_token_loss = -(exp(Δ) × advantages - β × per_token_kl)
            这等价于最小化 -(gain - cost)，也就是最大化 gain - cost。
            因此"净赚"的条件是：
              gain - cost > 0
              ⇔ exp(Δ) × advantages > β × per_token_kl
            而不是 -(exp(Δ) × advantages > β × per_token_kl)。
            前面的负号只是把"最大化目标"转成"最小化 loss"，不是条件的一部分。
          只有 exp(Δ) × advantages > β × per_token_kl（即收益超过 KL 代价），
          这次调整才是净赚的，优化器才会采纳。如果 exp(Δ) × advantages < β × per_token_kl，
          风控就会拦下——产生的收益还不够交罚款。

        β 的大小决定了风控的严格程度：
          β=0：没有风控，组合可以随意偏离，几天就全仓 AI 股票（坍缩）
          β=0.01：宽松风控，允许较大偏离
          β=0.05：严格风控，几乎只能微调
          典型值 0.01~0.05 相当于"允许调整，但不能太激进"。

        代码中使用 per-token 无偏 KL 估计 exp(Δ) - Δ - 1（Δ = log π_ref - log π_θ）：
          精确 KL 需要遍历词表：KL = Σ_{t∈V} π_θ(t) × log(π_θ(t)/π_ref(t))  O(V) 计算。
          改为用当前采样 token 的 log-prob 做单点估计（来自 DeepSeekMath）：
            f(Δ) = exp(Δ) - Δ - 1,  其中 Δ = log π_ref - log π_θ

          详细数学含义：
            1) 精确 KL 定义：
                 KL(π_θ || π_ref) = Σ_{t∈V} π_θ(t) × log(π_θ(t) / π_ref(t))
               这表示 policy 分布 π_θ 相对于 reference 分布 π_ref 的总"偏离程度"，
               需要对整个词表 V 求和，计算量 O(V)。

            2) 单点估计的核心思想：
                训练中模型是逐 token 采样的：每个生成 token t 都是按 π_θ 抽出来的。
                因此我们只关注"实际采到的这个 token t"，用它来近似整个分布。
                定义 Δ_t = log π_ref(t) - log π_θ(t)，然后构造：
                  f(Δ_t) = exp(Δ_t) - Δ_t - 1
                这里的 exp 就是指数函数 exp(x) = e^x（e ≈ 2.71828）。
                例如 exp(1) ≈ 2.718，exp(0) = 1，exp(-1) ≈ 0.368。
                这看起来像是一个启发式函数，但它有一个关键性质：
                  E_{t ~ π_θ}[ f(Δ_t) ] = KL(π_θ || π_ref)
                也就是说，如果 t 按 policy 分布 π_θ 采样，那么 f(Δ_t) 的期望
                恰好等于精确 KL 散度。这就是"无偏估计"。

            3) 无偏性的代数验证：
                E[f(Δ)] = Σ_{t∈V} π_θ(t) × [ exp(log π_ref(t) - log π_θ(t))
                                                - (log π_ref(t) - log π_θ(t))
                                                - 1 ]
                        = Σ π_θ(t) × [ π_ref(t)/π_θ(t) - log π_ref(t) + log π_θ(t) - 1 ]
                        = Σ [ π_ref(t) - π_θ(t)×log π_ref(t) + π_θ(t)×log π_θ(t) - π_θ(t) ]
                        这里利用 Σ π_ref(t) = 1 和 Σ π_θ(t) = 1：
                        = 1 - Σ π_θ(t)×log π_ref(t) + Σ π_θ(t)×log π_θ(t) - 1
                        = Σ π_θ(t)×[log π_θ(t) - log π_ref(t)]
                        = Σ π_θ(t)×log(π_θ(t)/π_ref(t))
                        = KL(π_θ || π_ref)   ✓

            4) 为什么选这个函数：
                - O(1) 计算：只需当前 token 的 log-prob，不必遍历词表。
                - 始终 ≥ 0：函数 g(x)=exp(x)-x-1 在 x=0 处取最小值 0，且为凸函数，
                  所以 f(Δ_t) ≥ 0，符合 KL 散度的非负性。
                - 方差更小：当 π_θ ≈ π_ref 时，Δ_t ≈ 0，f(Δ_t) ≈ 0 且变化平缓，
                  不像 log-ratio 那样对微小偏差敏感。

          比喻：精确 KL 像普查 —— 挨家挨户问每个人的收入再算人均收入，精确但贵。
                exp(Δ)-Δ-1 像抽样调查 —— 只随机抽一个人问收入，然后用"这个人在人群
                中被抽中的概率（π_θ）"反过来校正回答。单次可能不准（万一抽到马化腾），
                但反复抽很多次取平均，结果就和普查一样。

          无偏性验证：对 x ~ π_θ 求期望
            "x ~ π_θ" 表示 x 按 policy 分布的 π_θ(·) 采样。
            E_{x~π_θ}[f(x)] = Σ_{t∈V} π_θ(t) × f(t)，即以 π_θ 为权重加权平均。
            E[f(Δ)] = E[π_ref/π_θ - log(π_ref/π_θ) - 1]
                    = Σ π_θ × π_ref/π_θ - E[log(π_ref/π_θ)] - 1
                    = 1 + Σ π_θ × log(π_θ/π_ref) - 1
                    = KL(π_θ || π_ref)   ✓
          优点：(1) O(1) per-token，不必算全词表；(2) 方差更小；(3) 始终 ≥ 0。
          方差更小的原因：训练中 π_θ ≈ π_ref 时 r = π_ref/π_θ ≈ 1。
            令 g(r)=r-log r-1, h(r)=log r。在 r=1 处泰勒展开：
              g(r) ≈ ½(r-1)²          （一阶导为 0，纯二次，r=1 处平坦）
              h(r) ≈ (r-1) - ½(r-1)² （一阶导为 1，有线性项）
            log r 有线性项，r 的小波动会直接产生一阶噪声；而 g(r) 只有二次项，
            在 r=1 附近梯度为 0，同等程度的概率波动带来的数值变化更小，方差更低。

    训练流程（每个 step，举具体数据例子贯穿）：
      设 B=2 个 prompt，num_generations=3。
        prompt[0] = "中国的首都是？"    prompt[1] = "1+1=?"
      1. 对每个 prompt，用 policy 模型生成 num_generations 条回复：
          prompt[0] 的 3 条采样 → "北京"、"北京是中国的首都"、"Beijing"
          prompt[1] 的 3 条采样 → "2"、"二"、"等于2"
          outputs shape = [2×3, P+R] = [6, P+R]。
      2. 用 reward model + 格式规则计算每条回复的奖励：
          rewards = [1.2, 0.8, -0.5, 1.5, 0.3, -1.0]   shape [6]
          前 3 条对应 prompt[0] 的 group，后 3 条对应 prompt[1] 的 group。
      3. 在 group 内归一化得到 advantages：
          group 0: mean=0.5, std=0.85 → advantages = [(1.2-0.5)/0.85, (0.8-0.5)/0.85, (-0.5-0.5)/0.85]
                                             = [0.824, 0.353, -1.176]
          group 1: mean≈0.267, std≈1.25 → advantages ≈ [0.986, 0.026, -1.013]
          拼接后 advantages shape [6]，再全局标准化 → adv ≈ [0.9, 0.4, -1.3, 1.1, 0.0, -1.1]
       4. 计算 policy / reference 模型的 per-token log-probability：
           把完整的 prompt + completion 拼起来做一次前向，而不是只 prefill prompt。
           因为因果注意力 mask 保证第 t 个 token 只能看到前 t 个 token，所以一次前向即可
           并行算出每个位置上对 next-token 的预测 logits，从中取出模型分配给"实际生成的 token"
           的 log-prob——即模型对自身决策的"自信度"。
          具体过程（以 prompt[0]="中国的首都是？" gen[2]="Beijing" 为例）：
            input_ids = [P<0>...P<6>, C<0>B, C<1>e, C<2>i, C<3>j, C<4>i, C<5>n, C<6>g]
                         ↑ prompt 7个token    ↑ completion 7个token
            logits = model(input_ids).logits  shape [1, 14, vocab_size]
                     logits[:, :-1, :] 去掉最后一位（它预测的是第15位，不存在）
                     得到 shape [1, 13, vocab_size]，第 t 个位置预测 token t+1
            对于实际生成的 token "Beijing"（B, e, i, j, i, n, g）：
              logits[0][6] 的分布中 "B" 的 log-prob = -0.15  ← 模型在 prompt 末尾预测"B"的自信度
              logits[0][7] 的分布中 "e" 的 log-prob = -0.02  ← 见到"B"后预测"e"
              logits[0][8] 的分布中 "i" 的 log-prob = -0.01
              logits[0][9] 的分布中 "j" 的 log-prob = -0.03
              logits[0][10] 的分布中 "i" 的 log-prob = -0.01
              logits[0][11] 的分布中 "n" 的 log-prob = -0.05
              logits[0][12] 的分布中 "g" 的 log-prob = -0.01
            per_token_logps[2] = [-0.15, -0.02, -0.01, -0.03, -0.01, -0.05, -0.01]
            值越接近 0 表示模型越"确信"这个 token（如 -0.01 的"i"和"g"
            在英语中几乎确定；-0.15 的"B"稍低，因为"北京"也可以拼成"Peking"）。
             policy 和 ref 各算一遍，得到两组 per_token_logps，shape 都是 [6, R]。
            两个都算是因为 GRPO 的 KL 惩罚需要两者的差值：Δ = log π_ref - log π_θ。
            如果将 policy 比作学生，ref 就是"初始水平"——学生在 RL 中向高 reward 方向
            更新（如学会新句式），ref 始终冻结在原状态。Δ < 0 表示 π_θ > π_ref
            （模型更自信了），Δ > 0 表示 π_θ < π_ref（模型退缩了）。
            对于 policy 模型新学会的模式（如"Beijing"被鼓励），其 log-prob 会
            逐渐升高（更自信），ref 的 log-prob 保持不变，导致 Δ = ref - policy < 0。
      5. 按 GRPO 目标算 loss：
          Δ = ref - policy  
          per_token_kl = exp(Δ) - Δ - 1  shape [6, 3]
           per_token_loss = -(exp(per_token_logps - per_token_logps.detach()) * adv.unsqueeze(1) - β × per_token_kl)
           per_token_logps 来自 policy 模型前向，detach 作用在张量上而非模型上。
           三个部分从右往左看：
           ① -β × per_token_kl：KL 惩罚项，β 越大约束越强
           ② exp(per_token_logps - per_token_logps.detach()) × adv：加权优势。
              exp(Δ) 中的 Δ = log π_θ_old - log π_θ = per_token_logps.detach() - per_token_logps，
              所以 exp(Δ) = π_θ / π_θ_old，是重要性采样权重（importance sampling ratio）。
              它衡量"当前策略 π_θ 相对采样时的旧策略 π_θ_old 变化了多少倍"：
                - exp(Δ) > 1：当前策略比采样时更置信，放大了优势信号；
                - exp(Δ) < 1：当前策略比采样时更保守，缩小了优势信号；
                - exp(Δ) ≈ 1：策略基本没变，几乎不影响优势。
              detach 使 log π_θ_old 不产生梯度——梯度只从分子（新策略 log π_θ）流入，
              实现"用当前概率/旧概率去加权优势，但只更新当前概率"。
              乘以 adv 后，高 adv 的 token 模式被鼓励，低 adv 的被抑制。
           ③ 最外层 -()：因为 optimizer 做梯度下降最小化 loss，而 GRPO 目标是最大化
              E[A·w - β·KL]，所以取负号转为最小化。
          用 completion_mask 去掉 eos 之后的无效 token，对序列内取平均再 batch 平均。
          假设有利的回复（adv>0）loss 小，不利的回复（adv<0）loss 大，梯度推动 policy
          向高 reward 的 token 模式偏移，同时 KL 惩罚防止坍缩。
      6. 梯度累积 + 更新：
          每 accumulation_steps 个 step 调用一次 optimizer.step()。

    Args:
        epoch: 当前 epoch 序号（从 0 开始）
        loader: 当前 epoch 的 DataLoader
        iters: 总 step 数（含跳过的 step），用于进度显示
        ref_model: 冻结的 reference 模型（初始 policy checkpoint）
        reward_model: 奖励模型
        reward_tokenizer: 奖励模型 tokenizer
        start_step: 已跳过的步数（续训时从之前的 step 恢复显示用）
        wandb: wandb/swanlab logger 实例，可选
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        # ==================== 1. 准备 prompt 输入 ====================
        prompts = batch['prompt']  # list[str], length B
        # 左填充（left padding），因为 generate 时最后一个 token 在序列末尾
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        # input_ids: [B, P], attention_mask: [B, P]
        if args.max_seq_len:
            # 如果 prompt 超过 max_seq_len，从右边裁剪（保留最新的 token）
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ==================== 2. 用 policy 模型生成回复 ====================
        with torch.no_grad():
            # DDP 包装后，.module 才是原始模型（generate 方法在原始模型上）
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id)
            # outputs: [B * num_gen, P + R]
            # 每个 prompt 生成 num_gen 条回复，所以 batch 维度翻倍。
            # 关键参数解释：
            #   do_sample=True：开启采样模式。模型对每个位置输出一个概率分布，
            #     不是直接选最大概率的 token（贪心），而是按这个分布随机抽一个 token。
            #     这就是"采样"（sampling）：从模型预测的 token 分布里随机抽样，
            #     因此同一个 prompt 每次运行都可能得到不同回复，从而产生多样性。
            #   temperature=0.8：对概率分布做平滑，让模型不那么确定，
            #     增加输出多样性；temperature 越接近 0 越接近贪心，越接近 1 越随机。
            # 示例：假设 B=2（2 个 prompt），num_gen=3：
            #   prompt = ["中国的首都是？", "1+1=?"]
            #   生成后 outputs 包含 2*3=6 条序列，顺序为：
            #     [0] 中国的首都是？北京                   ← prompt 0 的第 1 条采样
            #     [1] 中国的首都是？北京是中国的政治中心        ← prompt 0 的第 2 条采样
            #     [2] 中国的首都是？Beijing                ← prompt 0 的第 3 条采样
            #     [3] 1+1=?2                              ← prompt 1 的第 1 条采样
            #     [4] 1+1=?二                              ← prompt 1 的第 2 条采样
            #     [5] 1+1=?等于2                            ← prompt 1 的第 3 条采样
            # 这是 HuggingFace num_return_sequences 的内置行为：
            # 同一 prompt 的多条采样在 batch 维连续排列。

        # 分离 completion 部分（去掉 prompt 部分，只保留新生成的 token）
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B * num_gen, R]

        # ==================== 3. 计算 per-token log-probability ====================
        def get_per_token_logps(mdl, input_ids, n_keep):
            """
            计算每个 token 的对数概率。

            思路：对完整序列 input_ids 做一次前向，只保留最后 n_keep+1 个 logits
            （因为序列太长时前面部分的 logits 不参与 loss 计算，被截断以节省显存）。

            Args:
                mdl: 模型
                input_ids: [B*num_gen, P+R]
                n_keep: 需要计算 log-prob 的 token 数量（即 completion 的长度 R）

            Returns:
                per_token_logps: [B*num_gen, R]，每个 completion token 的 log-probability
            """
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            # logits_to_keep=n_keep+1 表示只保留最后 n_keep+1 个位置的 logits
            # 因为我们只需要 completion 部分的 log-prob（等价于取倒数第 n_keep+1 个位置的输出作为起始概率）
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]  # 去掉最后一个位置的 logits（无对应 next token）
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                # gather: 从 vocab 维取出对应 token id 的 log-probability
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        with autocast_ctx:
            # policy 模型的 per-token log-prob
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B * num_gen, R]
            # 如果使用 MoE，额外计算 expert 负载均衡 aux loss
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)

        with torch.no_grad():
            # reference 模型的 per-token log-prob（完全冻结，不计算梯度）
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B * num_gen, R]

        # ==================== 4. 解码 + 计算奖励 ====================
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B * num_gen]

        # ==================== 5. 构造优势函数（Group-Based Normalization） ====================
        # 这是 GRPO 的核心：对同一个 prompt 的 num_generations 条回复，
        # 在 group 内做标准化得到 advantages，抛弃了 PPO 的 critic 网络。
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B * num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B * num_gen]
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 全局标准化：对拼接后的整个 batch 再做一次 (x-mean)/std，使 advantages 严格均值为 0、标准差为 1，消除 group 间尺度差异

        # ==================== 6. 构建 completion mask（只保留到 eos 之前的部分） ====================
        is_eos = completion_ids == tokenizer.eos_token_id  # [B * num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # completion_mask: 从开头到 eos 位置（含）为 1，之后为 0。未产生 eos 的全为 1。
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()

        # ==================== 7. 计算 GRPO Loss ====================
        # KL 散度的近似计算（参考 DeepSeekMath 论文）：
        # KL(π_θ || π_ref) = exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1
        # 这是一个无偏估计且方差较小。
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B * num_gen, R]

        # GRPO 目标函数（最大化）：
        # J(θ) = E[ 1/R Σ ( min( π_θ/π_old * A, clip(π_θ/π_old, 1-ε, 1+ε) * A ) - β * KL ) ]
        # 在这里权重 π_θ/π_old 被简化为 exp(per_token_logps - per_token_logps.detach())
        # 即使用了 stop-gradient 技巧（detach），使 per_token_logps 在优势部分停止梯度回传。
        # β 控制 KL 惩罚的强度（β 越大，π_θ 越保守，越接近 π_ref）。
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)
        # policy_loss: 先对每个序列内的 token 取平均，再对 batch 取平均
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # aux_loss from MoE 的负载均衡 loss，通过 accumulation_steps 缩放以适应梯度累积
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        # ==================== 8. 梯度累积 + 更新 ====================
        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ==================== 9. 日志记录 ====================
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps  # 恢复未缩放的 loss 值
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        # ==================== 10. 保存 checkpoint ====================
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)  # torch.compile 会包裹 _orig_mod
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # ==================== 11. 显存清理 ====================
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=8, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化分布式环境和随机种子 ==========
    # init_distributed_mode 会自动解析环境变量（RANK, WORLD_SIZE, MASTER_ADDR 等）
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 不同进程使用不同种子避免数据同步一致
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 创建目录、初始化模型配置、检查续训状态 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 注意：max_seq_len 在模型配置中取 prompt 长度 + 生成长度之和
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    # from_resume==1 时尝试从 ../checkpoints 加载之前的 checkpoint
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度上下文 ==========
    # CPU 不使用 autocast（不支持）；GPU 上用 bfloat16 或 float16 自动混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置 wandb / swanlab 可视化 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None  # 续训时强制恢复同一个 run
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型、tokenizer、数据集、优化器 ==========
    # base_weight: 推理模型用 "reason"，普通模型用 "full_sft" 作为基座
    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    # --- Policy 模型（被训练的 actor）---
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # --- Reference 模型（参数冻结的初始策略，用于 KL 散度计算）---
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # --- Reward 模型（外部加载，冻结，用于生成奖励信号）---
    # Reward 模型与待训练的 MiniMind policy 模型是完全独立的两个模型体系：
    #   - Policy / Reference 模型: MiniMind 架构（model/model_minimind.py），权重从 base_weight
    #     ("reason" 或 "full_sft") 初始化，GRPO 训练会更新其参数。
    #   - Reward 模型: 通过 --reward_model_path 指定的独立模型（默认 internlm2-1_8b-reward），
    #     使用 HuggingFace AutoModel.from_pretrained 加载，是一个经过偏好对齐训练的外部奖励模型，
    #     与 MiniMind 没有任何共享参数或架构关系。
    #  Reward 模型在整个 GRPO 训练中完全冻结（requires_grad_(False)），仅用作评分器：
    #  接收 policy 模型生成的回复，输出一个标量分数作为奖励信号，指导 policy 模型的参数更新方向。
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    # --- 数据集和 DataLoader ---
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)                         # 每个 epoch 的 step 数
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs  # 实际参数更新次数
    # CosineAnnealingLR：学习率从 lr 到 lr/10 按照余弦曲线衰减
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 6. 从 checkpoint 恢复状态（续训支持） ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 用 DistributedDataParallel 包装模型 ==========
    if dist.is_initialized():
        # RoPE 频率矩阵是 buffer（非参数），且在 forward 内部计算时不依赖 DDP 同步，因此忽略
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 训练主循环 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)  # 分布式采样器按 epoch 打乱
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()  # 用于非分布式时取随机顺序
        # 续训时跳过前 start_step 个 batch
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, ref_model, reward_model, reward_tokenizer, start_step, wandb)
        else:
            grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, 0, wandb)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
