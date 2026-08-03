# 已完成学习清单

## eval_llm.py

- [✓] 文件定位与前置知识理解
- [✓] import & 全局（warnings.filterwarnings 作用）
- [✓] init_model() 两个分支（原生 .pth vs HF 格式）
- [✓] argparse 参数全表理解
- [✓] 对话循环：预设 prompts / 交互模式选择
- [✓] Chat Template 与 Tokenization（pretrain vs SFT 差异）
- [✓] reason 模型的 enable_thinking 参数
- [✓] model.generate() 各参数作用
- [✓] 响应处理与速度统计（切片 + 解码）
- [✓] 自测问题 1~17 全部回答并注释存档

### 动手练习完成情况

#### 基础
- [✓] 练习 1：pretrain vs full_sft 对比测试
- [✓] 练习 2：temperature 0.1 / 0.85 / 1.5 对比测试
- [✓] 练习 3：top_p 0.5 / 0.9 / 1.0 对比测试

#### 进阶
- [✓] 练习 4：修改 prompts 列表，支持按 --weight 追加领域 prompt
- [✓] 练习 5：LoRA 加载逻辑修改，支持多 LoRA 合并（逗号分隔）
- [✓] 练习 6：新增 --repetition_penalty 参数并测试 1.0 / 1.2 / 2.0

#### 深入
- [✓] 练习 7：理解 streamer 调用链路与 CustomStreamer 设计原理
- [✓] 练习 8：对比 serve_openai_api.py 与 eval_llm.py 的 init_model 差异
- [✓] 练习 9：量化推理插入位置分析（权重 vs 激活量化）

### 学习文档
- [✓] _rules/answer_format.md — 回答格式行为准则阅读理解
- [✓] plan/eval_llm_study_plan.md — 学习计划与测试记录（已更新）
- [✓] knowledge/run_troubleshooting.md — 运行问题记录
- [✓] knowledge/model_download.md — 模型下载指南
- [✓] knowledge/basic_tests.md — 基础测试结果存档
- [✓] eval_llm.py — 源码注释已全部整理（含自测答案）

## train_pretrain.py（预训练脚本 ✅ 已完成）

学习计划见 `plan/train_pretrain_study_plan.md`

- [✓] 导入与全局配置理解
- [✓] `train_epoch()` 核心循环（前向/反向/梯度累积/学习率调度）
- [✓] `main()` 函数与参数解析
- [✓] checkpoint 保存与恢复机制
- [✓] 训练结果观察与参数调优
- [✓] 量化方法系统学习（GPTQ / AWQ / SmoothQuant / NF4 / GGUF / QuaRot / SpinQuant / AffineQuant / TurboQuant / LLM.int8()）
- [✓] 自测练习题 1~12 全部回答并注释存档

## train_full_sft.py（指令微调 ✅ 已完成）

学习计划见 `plan/train_full_sft_study_plan.md`

- [✓] 理解 SFTDataset 的数据处理流程（apply_chat_template）
- [✓] 理解 generate_labels 的 loss masking 策略
- [✓] 理解 init_model 从预训练权重加载的机制
- [✓] 对比 pretrain 和 SFT 的 loss 计算差异
- [✓] 掌握全模型微调（full SFT）和 LoRA 微调的区别

---

## 下一个：LoRA 参数高效微调 ✅ 已完成

学习计划见 `plan/train_lora_study_plan.md`

- [✓] 理解 LoRA 的核心思想（低秩分解，冻结原权重）
- [✓] 理解 LoRA 前向传播流程（旁路加法）
- [✓] 理解 apply_lora 的模块注入机制
- [✓] 理解参数冻结与仅训练 LoRA 权重的梯度控制
- [✓] 理解梯度裁剪作用在 lora_params 而非 model.parameters()
- [✓] 理解 LoRA 权重的保存/加载机制（save_lora / load_lora）
- [✓] 对比 full SFT 和 LoRA 的异同（参数量、显存、效果）
- [✓] 理解多 LoRA 合并与权重融合（apply_lora_multi / load_lora_multi）

---

## 下一个：偏好对齐训练（DPO）✅ 已完成

学习计划见 `plan/train_dpo_study_plan.md`

- [✓] 理解 DPO 的核心思想（直接偏好优化 vs RLHF）
- [✓] 理解偏好数据集的格式与构造（chosen / rejected）
- [✓] 理解 DPO loss 的计算公式
- [✓] 理解 GRPO 与 DPO 的区别
- [✓] 理解 reference model 的作用与冻结机制
- [✓] 掌握完整训练流程：SFT → DPO → Reason → GRPO

---

## 下一个：推理能力训练（Reason）✅ 已完成

学习计划见 `plan/train_reason_study_plan.md`

- [✓] 理解 reasoning 训练的目标（让模型学会"先思考再回答"）
- [✓] 理解 <think>/<answer> 特殊 token 的作用
- [✓] 理解 reasoning 数据的构造格式
- [✓] 理解 enable_thinking 在推理时的处理逻辑
- [✓] 对比 reasoning SFT 和普通 SFT 的差异
- [✓] 理解加权 loss 的核心原理（特殊 token 权重 10 倍的原因）
- [✓] 理解 reduction='none' + 手动 mask 相比 label=-100 的灵活之处
- [✓] 理解推理数据集的来源（r1_mix_1024.jsonl 是蒸馏数据）
- [✓] 理解 train_reason.py 的模型架构（单模型，基于 DPO 权重）
- [✓] 理解 loss_mask_sum vs mask.sum() 两种归一化的数值差异
- [✓] 自测题 1~9 全部回答并更新至文档

---

## 下一个：组相对策略优化训练（GRPO）✅ 已完成

学习计划见 `plan/train_grpo_study_plan.md`

- [✓] 理解 GRPO 与 DPO 的核心区别（在线采样 vs 静态数据）
- [✓] 理解 GRPO 的奖励模型和奖励信号设计
- [✓] 理解 GRPO 依赖 reasoning 格式的原因
- [✓] 理解 GRPO 的组采样和相对优势计算
- [✓] 理解 GRPO 训练在 MiniMind 完整流程中的位置
- [✓] 理解 3 模型架构（policy + ref + reward）
- [✓] 理解格式奖励 + Reward Model 评分的双层奖励设计
- [✓] 理解 KL 惩罚 exp(kl) - kl - 1 的作用与性质
- [✓] 理解 `detach()` 在 GRPO loss 中的真实作用（防止梯度自抵消）
- [✓] 理解 DPO β 乘法 vs GRPO β 加法的本质区别（附比喻集）
- [✓] 自测题 1~9 全部回答并更新至文档

---

## 下一个：近端策略优化训练（PPO）✅ 已完成

学习计划见 `plan/train_ppo_study_plan.md`

- [✓] 理解 PPO 的 5 模型架构及其各自的作用
- [✓] 理解 CriticModel 的设计（共享主体 + value_head）
- [✓] 理解 Critic 随机初始化的含义
- [✓] 理解 `A = R - V(s)` 的优势计算方式
- [✓] 理解 PPO 裁剪机制 `min(surr1, surr2)` 的作用
- [✓] 理解旧策略同步机制（每 K 步同步一次）
- [✓] 理解 PPO 的 3 成分 Loss（策略 + 价值 + KL）
- [✓] 对比 PPO 与 GRPO 的优势计算方式差异
- [✓] 对比 PPO 与 GRPO 的模型数量差异
- [✓] 理解 Actor 与 Critic 独立优化器、独立调度器的设计
- [✓] 理解 PPO 在完整管线中的位置

---

## 下一个：自博弈优化训练（SPO）✅ 已完成

学习计划见 `plan/train_spo_study_plan.md`

- [✓] 理解 SPO 与 DPO / GRPO / PPO 的本质区别（基线来源不同）
- [✓] 理解 AutoAdaptiveValueTracker 的 Beta 分布设计思想
- [✓] 理解 get_baselines() 和 update() 的滑动更新机制
- [✓] 理解 compute_rho() 的自适应衰减和 D_half 的含义
- [✓] 理解 SPO 的 3 模型架构与 Loss 结构
- [✓] 理解 get_per_token_logps 的逐行 gather 处理
- [✓] 理解 baseline 反标准化和 advantages.clamp 的作用
- [✓] 对比 SPO / GRPO / PPO / DPO 四种对齐方式的优缺点

---

## 下一个：知识蒸馏训练（train_distillation.py）✅ 已完成

学习计划见 `plan/train_distillation_study_plan.md`

- [✓] 理解蒸馏的核心思想（让小模型模仿大模型）
- [✓] 理解温度 T 对 softmax 分布的影响
- [✓] 理解 Soft Target vs Hard Target 的信息量差异
- [✓] 理解 KL 散度在蒸馏中的作用（衡量学生与老师的分布差距）
- [✓] 理解 `F.kl_div` 的参数设计（log 概率 vs 概率）与数值稳定性
- [✓] 理解 `log_softmax` 减去 max 的原理与数学等价性证明
- [✓] 理解为什么老师用概率、学生用 log 概率（不对称性设计）
- [✓] 理解蒸馏 loss 乘 `temperature^2` 的原因（梯度补偿）
- [✓] 理解 `alpha * CE + (1-alpha) * Distill` 的加权策略
- [✓] 理解 CE Loss 的具体计算过程（shift_labels + loss_mask）
- [✓] 理解双模型架构（学生可训练 + 老师冻结）
- [✓] 理解老师 logits 截断到学生词表大小的原因
- [✓] 理解词表对齐问题（同 tokenizer 安全，不同 tokenizer 需映射）
- [✓] 理解蒸馏与之前所有方法的对比（模型数量、Loss、适用场景）
- [✓] 理解蒸馏的知识迁移 vs RL 的能力创造
- [✓] 自测题 1~14 全部回答并更新至文档

---

## 下一个：分词器训练（train_tokenizer.py）✅ 已完成

学习计划见 `plan/train_tokenizer_study_plan.md`（已创建）

- [✓] 理解 BPE（Byte Pair Encoding）分词算法的原理
- [✓] 理解 `tokenizers` 库的核心组件（Tokenizer / Trainer / PreTokenizer / Decoder）
- [✓] 理解特殊 token 的作用（bos / eos / Sep）
- [✓] 理解 vocab_size 对模型能力的影响
- [✓] 理解为什么 MiniMind 不建议重新训练 tokenizer
- [✓] 动手练习：用不同 vocab_size 训练 tokenizer 并对比效果

### 学习文档
- [✓] plan/train_tokenizer_study_plan.md — 学习计划与测试记录（已更新）

---

## model_minimind.py 模型架构 ✅ 已完成

学习计划见 `plan/model_minimind_study_plan.md`

- [✓] 理解 RMSNorm 的原理及与 LayerNorm 的区别
- [✓] 理解 RoPE 的两步流程（预计算 + 注入）
- [✓] 理解 Attention 的两条计算路径（Flash Attention vs 手动计算）
- [✓] 理解 GQA 的分组机制和 repeat_kv 的实现
- [✓] 理解 KV Cache 的作用与内存布局
- [✓] 理解 SwiGLU 的三矩阵结构和 8/3 倍的由来
- [✓] 理解 MoE 的路由机制和辅助损失
- [✓] 理解 MiniMindBlock 的 Pre-Norm + 残差结构
- [✓] 理解 Weight Tying（权重绑定）的原理
- [✓] 能逐模块计算 26M / 104M 参数的来源
- [✓] 深入理解：先重塑再 RoPE 的原因、Pre-Norm vs Post-Norm 梯度流、Flash Attention 因果计算优化、因果掩码注入时机、MoE 辅助损失 Pᵢ/fᵢ 计算

### 学习文档
- [✓] plan/model_minimind_study_plan.md — 学习计划与 Q&A 记录（已更新，含大量图解）

---

## model_lora.py LoRA 底层实现 ✅ 已完成

学习计划见 `plan/model_lora_study_plan.md`

- [✓] 理解 LoRA 的数学原理（低秩分解 A×B）
- [✓] 理解 LoRA 的注入机制（如何替换 nn.Linear）
- [✓] 理解 apply_lora / save_lora / load_lora 的实现细节
- [✓] 理解多 LoRA 合并（apply_lora_multi / load_lora_multi）
- [✓] 理解 LoRA 权重融合（merge）的原理
- [✓] 对比 model_lora.py 与 train_lora.py 的分工

### 学习文档
- [✓] plan/model_lora_study_plan.md — 学习计划与 Q&A 记录（已更新）

---

## 训练工具函数（trainer_utils.py）✅ 已完成

学习计划见 `plan/trainer_utils_study_plan.md`

- [✓] 理解 is_main_process / Logger 的分布式控制逻辑
- [✓] 理解 get_lr 余弦退火学习率调度
- [✓] 理解 init_distributed_mode DDP 初始化
- [✓] 理解 lm_checkpoint 的断点续传机制（存档 vs 读档）
- [✓] 理解 init_model 的权重加载流程
- [✓] 理解 SkipBatchSampler 的跳批次机制
- [✓] 理解 lm_checkpoint 原子写入与 world_size 步数换算
- [✓] 理解 SkipBatchSampler + indices 的一致性保证

---

## 数据集加载（dataset/lm_dataset.py）✅ 已完成

学习计划见 `plan/lm_dataset_study_plan.md`（已更新）

- [✓] 理解 PretrainDataset 的数据加载与 tokenization
- [✓] 理解 SFTDataset 的 Chat Template 格式与 loss masking
- [✓] 理解 DPODataset 的 chosen/rejected 对比数据构造
- [✓] 理解 RLAIFDataset 的 RLHF 数据格式
- [✓] 理解 Dataset 的 __len__ / __getitem__ 设计
- [✓] 理解数据集与训练脚本的对接方式
- [✓] 对比 PretrainDataset 手动加 BOS/EOS 与 SFTDataset Chat Template 自带特殊 token 的差异
- [✓] 对比 SFTDataset.generate_labels()（-100）与 DPODataset.generate_loss_mask()（0/1）的设计选择
- [✓] 理解 DPODataset __getitem__ 的 shift 处理与 6 个返回值的含义
- [✓] 自测问题 Q1~Q5 全部回答并更新至文档

---

## API 服务部署（scripts/serve_openai_api.py）✅ 已完成

学习计划见 `plan/serve_openai_api_study_plan.md`

- [✓] 理解 init_model 的两条加载路径（原生 .pth vs HuggingFace 格式）
- [✓] 理解 ChatRequest Pydantic 模型的作用
- [✓] 理解 CustomStreamer + Queue + Thread 的流式生成机制
- [✓] 理解 generate_stream_response 的 SSE 格式编码
- [✓] 理解 chat_completions 流式/非流式双分支
- [✓] 理解 __main__ 中 uvicorn 的启动方式
- [✓] 对比 serve_openai_api.py 与 eval_llm.py 的差异
- [✓] 理解 `tools` 字段目前只是兼容占位，完整 function calling 还需要工具执行与多轮回注逻辑
- [✓] 理解字符级 prompt 截断与 token 级截断之间的差异
- [✓] 完成服务启动、非流式请求和流式 SSE 请求练习

---

## 下一个：OpenAI API 客户端（scripts/chat_openai_api.py）✅ 已完成

学习计划见 `plan/chat_openai_api_study_plan.md`

- [✓] 理解 OpenAI SDK 的 `base_url` 与兼容接口配置
- [✓] 理解 `chat.completions.create()` 的请求参数映射
- [✓] 理解非流式响应的 `message.content` 读取方式
- [✓] 理解流式响应的 `chunk.choices[0].delta.content` 读取方式
- [✓] 理解 `conversation_history` 的上下文维护与历史轮数截断
- [✓] 对比 `stream=True` 与 `stream=False` 的客户端处理流程
- [✓] 动手修改客户端：切换流式模式、调整历史轮数和生成参数

---

## 下一个：模型格式转换（scripts/convert_model.py）

学习计划见 `plan/convert_model_study_plan.md`

- [ ] 理解三种转换函数的定位与使用场景
- [ ] 理解 MiniMind HF 格式 vs Llama HF 格式的区别
- [ ] 理解 `register_for_auto_class()` 的作用
- [ ] 理解 `strict=False` 在权重加载中的含义
- [ ] 理解 SwiGLU 的 `intermediate_size` 计算公式
- [ ] 理解 `tokenizer_config.json` 修补的原因
- [ ] 理解 `safe_serialization` 与 `.bin` / `.safetensors` 的区别
- [ ] 动手练习：运行转换并验证加载
- [ ] 动手练习：用 argparse 改造 `__main__`


