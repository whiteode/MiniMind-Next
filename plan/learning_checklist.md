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

---

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

## 下一个：偏好对齐训练（DPO / GRPO）

学习计划见 `plan/train_dpo_study_plan.md`

- [ ] 理解 DPO 的核心思想（直接偏好优化 vs RLHF）
- [ ] 理解偏好数据集的格式与构造（chosen / rejected）
- [ ] 理解 DPO loss 的计算公式
- [ ] 理解 GRPO 与 DPO 的区别
- [ ] 理解 reference model 的作用与冻结机制
- [ ] 掌握完整训练流程：SFT → DPO → Reason → GRPO
