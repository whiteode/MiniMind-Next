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

## 下一个：train_pretrain.py（预训练脚本）

学习计划见 `plan/train_pretrain_study_plan.md`

- [ ] 导入与全局配置理解
- [ ] `train_epoch()` 核心循环（前向/反向/梯度累积/学习率调度）
- [ ] `main()` 函数与参数解析
- [ ] checkpoint 保存与恢复机制
- [ ] 训练结果观察与参数调优
