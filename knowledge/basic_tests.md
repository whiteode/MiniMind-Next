# 基础动手练习测试记录

## 练习 1：pretrain vs full_sft 对比

```bash
python eval_llm.py --weight pretrain
```

**结果**：pretrain 模型输出质量极差，生成大量无意义重复内容（"祝！谢谢。如果您..."），不具备对话能力。这符合预期——pretrain 只在纯文本上做过自回归语言建模，没学过对话格式。

对比 `--weight full_sft` 输出明显有结构、有信息量。SFT 阶段让模型学会了对话格式和指令遵循能力。

---

## 练习 2：temperature 对比

### temperature=0.1

```bash
python eval_llm.py --weight full_sft --temperature 0.1
```

**特点**：输出极其确定性和重复性，容易出现局部循环（如大量重复"进行数据分析"），多样性几乎为零。极端贪心解码的效果。

**示例**（"你有什么特长？"的部分输出）：
```
我被设计用来回答各种问题、提供信息、执行任务或执行任务。
我被设计用来帮助用户解决问题、提供信息、进行对话、
进行娱乐、进行教育、进行科学研究、进行数据分析、
进行数据分析、进行数据分析、进行数据分析...
```

### temperature=0.85（默认值）

```bash
python eval_llm.py --weight full_sft --temperature 0.85
```

**特点**：最佳平衡。回答内容连贯、信息量充足、结构清晰。8 个预设 prompt 的回答质量都较好。

**示例**（"用Python写斐波那契数列函数"）：
```python
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

**示例**（"解释什么是机器学习"）：
```
机器学习是一种人工智能的分支，它利用算法和统计模型来分析和学习数据，
从而使计算机能够从数据中自动学习并改进性能。机器学习可以分为三类：
监督学习、无监督学习和强化学习。
```

### temperature=1.5

```bash
python eval_llm.py --weight full_sft --temperature 1.5
```

**特点**：多样性最高，但质量下降明显。部分回答逻辑松散、偏离主题，甚至出现胡言乱语。数学/代码类任务受影响最严重。

**示例**（斐波那契数列函数回答混乱，未能给出正确实现）：
```
斐波那契数列是一个数学序列，其中每当一个数字超过它的某个值时，
它会增加1和1开始计算次数之比。
```

### 对比总结

| Temperature | 多样性 | 连贯性 | 信息准确度 | 适用场景 |
|-------------|--------|--------|-----------|---------|
| 0.1 | 极低 | 高 | 中（易重复死循环） | 需要完全确定性的场景 |
| 0.85 | 适中 | 高 | 高 | 通用对话（推荐默认值） |
| 1.5 | 高 | 低 | 低 | 创意生成、头脑风暴 |

---

## 练习 3：top_p 对比（待测试）

```bash
# 待执行
python eval_llm.py --weight full_sft --top_p 0.5
python eval_llm.py --weight full_sft --top_p 0.9
python eval_llm.py --weight full_sft --top_p 1.0
```
