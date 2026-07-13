# serve_openai_api.py 学习计划

> **文件位置**: `scripts/serve_openai_api.py`（197 行）
> **角色**: 把训练好的 MiniMind 模型封装成 OpenAI 兼容的 API 服务，支持流式和非流式推理
> **前置知识**: 已学完 eval_llm.py（了解模型加载和 generate 参数）、model_minimind.py（模型架构）、model_lora.py（LoRA 加载）

---

## 文件全景图

```
serve_openai_api.py
│
├── init_model(args)               ← 模型加载：原生 .pth 或 HuggingFace 格式
│
├── ChatRequest(BaseModel)         ← Pydantic 请求体模型（参数校验）
├── CustomStreamer(TextStreamer)   ← 自定义流式输出器（线程安全）
│
├── generate_stream_response()     ← 流式生成 + SSE 格式编码
│
├── chat_completions()             ← FastAPI 端点 /v1/chat/completions
│   ├── stream=True  → StreamingResponse + SSE
│   └── stream=False → 完整生成 + JSON 返回
│
└── __main__                        ← argparse + uvicorn 启动
```

---

## 第一章：init_model —— 两条加载路径

```python
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:                          # 路径含"model" → 原生 .pth
        ckp = f'../{args.save_dir}/{args.weight}_...pth'
        model = MiniMindForCausalLM(MiniMindConfig(...))    # 从零构造模型
        model.load_state_dict(torch.load(ckp), strict=True) # 加载权重
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'...pth')
    else:                                                   # 其他路径 → HuggingFace 格式
        model = AutoModelForCausalLM.from_pretrained(...)
    return model.eval().to(device), tokenizer
```

### 1.1 判断条件 `'model' in args.load_from`

为什么用字符串 contains 而不是显式参数？

- `--load_from` 的默认值是 `'../model'`（MiniMind 的 tokenizer 目录）
- 当用原生 `.pth` 权重时，传 `--load_from ../model` → 路径含 `'model'` → 走原生分支
- 当用 HuggingFace 格式时，传 `--load_from /path/to/hf` → 不含 `'model'` → 走 transformers 分支

这是一种依赖路径命名的隐式判断。缺点是如果某个 HuggingFace 路径恰好包含 `model` 子串，会走错分支。

### 1.2 原生分支 vs eval_llm.py 的 init_model

对比 `eval_llm.py` 的 `init_model`：

| 维度 | serve_openai_api.py | eval_llm.py |
|------|--------------------|-------------|
| 模型构建 | 每次都 `MiniMindForCausalLM(config)` + `load_state_dict` | 同左 |
| LoRA 加载 | 支持（`--lora_weight`） | 支持（`--lora_weight`） |
| 路径判断 | `'model' in args.load_from` | 同左 |
| tokenizer | `AutoTokenizer.from_pretrained(args.load_from)` | 同左 |
| 设备 | `args.device` 参数 | `args.device` 参数 |
| huggingface 分支 | `AutoModelForCausalLM.from_pretrained` | `AutoModelForCausalLM.from_pretrained` |

**基本一致**，只是服务端版本少了 `--flash_attn` / `--torch_compile` 等推理优化参数（生产环境通常靠服务框架做优化）。

### 1.3 HuggingFace 格式分支

```python
model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
```

这一行意味着如果你的模型已经转换为 HuggingFace 格式（`convert_model.py` 干的事），就可以直接用 `from_pretrained` 加载，**不需要硬编码模型结构参数（hidden_size、num_layers 等）**。config.json 里自带这些信息。

---

## 第二章：ChatRequest —— Pydantic 请求模型

```python
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = False
    tools: list = []
```

Pydantic 的 `BaseModel` 自动做：
1. **类型校验**：`temperature` 必须是 float，`max_tokens` 必须是 int
2. **默认值填充**：客户端没传的字段用默认值
3. **自动序列化/反序列化**：FastAPI 自动把 JSON body 解析成 `ChatRequest` 对象

`tools: list = []` 的存在是为了兼容 OpenAI API 格式，但 MiniMind 不支持 function calling（只是占位）。

**和 OpenAI API 的对应关系**：

| 字段 | 对应 OpenAI API 参数 |
|------|---------------------|
| `model` | `model`（这里固定返回"minimind"） |
| `messages` | `messages` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `max_tokens` | `max_tokens` |
| `stream` | `stream` |
| `tools` | `tools`（未实现，仅占位） |

---

## 第三章：CustomStreamer —— 线程安全的流式输出器

### 3.1 TextStreamer 回顾

HuggingFace Transformers 的 `TextStreamer` 是一个回调类，`model.generate()` 在每生成一个 token 后会自动调用 `on_finalized_text(text, stream_end)`。

默认的 `TextStreamer` 直接 `print` 到终端。`CustomStreamer` 把它改成了 **往 Queue 里放**。

### 3.2 自定义实现

```python
class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)      # 把生成的文本 chunk 放进队列
        if stream_end:
            self.queue.put(None)  # 放哨兵值，表示生成结束
```

**设计要点**：

- `skip_prompt=True`：只输出新生成的 token，不输出输入的 prompt
- `skip_special_tokens=True`：跳过 `<|im_end|>` 等特殊 token
- `Queue` 是线程安全的：生产者（`model.generate` 的线程）往里放，消费者（API 响应生成器）往外取
- `None` 作为结束信号：消费者读到 `None` 就知道生成完了

### 3.3 为什么需要 Thread？

```python
def _generate():
    model.generate(...)

Thread(target=_generate).start()

while True:
    text = queue.get()
    ...
```

`model.generate()` 是同步阻塞调用。如果不放线程里跑，`generate()` 执行期间无法同时做 `queue.get()`（死锁）。用线程把"生产者"和"消费者"分开：

```
主线程（API响应）                     子线程（生成）
     │                                   │
     │  Thread(target=_generate).start() │
     │ ──────────────────────────────────→│
     │                                   │ model.generate()
     │  while True:                      │   │
     │    queue.get() ← 阻塞等待 ← ← ← ← │   │ on_finalized_text("你好")
     │    yield "你好"                    │   │ on_finalized_text("！")
     │    queue.get() ← 阻塞等待 ← ← ← ← │   │ on_finalized_text(None)
     │    yield ... → 流式返回给客户端     │   │
     │    queue.get() → None → break      │   │
```

---

## 第四章：generate_stream_response —— 流式生成与 SSE 格式

### 4.1 SSE（Server-Sent Events）协议

OpenAI API 的流式响应格式是 SSE，每个事件以 `data: ` 开头、`\n\n` 结尾：

```
data: {"choices": [{"delta": {"content": "你好"}}]}

data: {"choices": [{"delta": {"content": "！"}}]}

data: {"choices": [{"delta": {}}]}
```

`generate_stream_response` 是一个 Python **生成器**（有 `yield`），每次 yield 一个 JSON 字符串。这个生成器被传给 `StreamingResponse`，后者自动包装成 SSE 格式。

### 4.2 关键操作：prompt 截断

```python
new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[-max_tokens:]
```

注意这里 `[-max_tokens:]` —— 取了格式化后字符串的**末尾 max_tokens 个字符**。这不是 token 级别的截断，是字符级别的。目的是防止对话历史太长导致输入超出模型能处理的范围。

但这里有个微妙的问题：字符数 ≠ token 数。如果对话很长，截断后可能还是超出模型的 `max_seq_len`。后面的 `tokenizer(..., truncation=True)` 会做最终的 token 级别截断，所以这行只是一个粗略的预截断。

### 4.3 非流式分支的响应格式

非流式响应的 JSON 结构完全兼容 OpenAI API：

```json
{
    "id": "chatcmpl-1741852800",
    "object": "chat.completion",
    "created": 1741852800,
    "model": "minimind",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "生成的回复"},
        "finish_reason": "stop"
    }]
}
```

---

## 第五章：chat_completions —— FastAPI 端点

### 5.1 路由装饰器

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
```

- `@app.post("/v1/chat/completions")`：和 OpenAI API 路径完全一致，方便客户端直接切换
- `async def`：FastAPI 异步处理，不阻塞事件循环
- `request: ChatRequest`：自动从请求 JSON body 解析并校验

### 5.2 流式 vs 非流式分支

```
chat_completions(request)
├── request.stream == True
│   └── StreamingResponse(generate_stream_response(...), media_type="text/event-stream")
│       └── 生成器逐步 yield → FastAPI 逐一发 SSE 事件
│
└── request.stream == False
    ├── apply_chat_template → tokenize → model.generate
    └── 返回完整 JSON
```

### 5.3 错误处理

```python
try:
    ...
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

所有异常统一转成 HTTP 500 错误。流式分支的异常在 `generate_stream_response` 里捕获并以 `{"error": str(e)}` 返回（不会引发 HTTPException，因为流已经开始响应了）。

---

## 第六章：__main__ —— 命令行入口

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(...)
    # 7 个模型加载参数
    args = parser.parse_args()
    device = args.device
    model, tokenizer = init_model(args)
    uvicorn.run(app, host="0.0.0.0", port=8998)
```

### 6.1 uvicorn

`uvicorn` 是 Python 的 ASGI 服务器，用于运行 FastAPI 应用：

```python
uvicorn.run(app, host="0.0.0.0", port=8998)
```

- `app`：FastAPI 实例（模块级全局对象）
- `host="0.0.0.0"`：监听所有网络接口（包括局域网）
- `port=8998`：端口号

### 6.2 模块级全局变量

```python
app = FastAPI()           # 模块级——在 import 时就被创建
model, tokenizer = ...    # 在 __main__ 中赋值

# 其他函数直接使用这些全局变量
def generate_stream_response(messages, temperature, top_p, max_tokens):
    new_prompt = tokenizer.apply_chat_template(...)   # 用全局 tokenizer
    inputs = tokenizer(...).to(device)                 # 用全局 device
    ...
```

`generate_stream_response` 和 `chat_completions` 都依赖模块级全局变量 `model`、`tokenizer`、`device`。这在单进程服务中是 OK 的，但在多 worker 场景下需要小心（每个 worker 都会加载一份模型）。

---

## 第七章：启动方式

```bash
# 1. 原生 .pth 权重（默认路径 ../model/ + ../out/）
python serve_openai_api.py --weight full_sft --hidden_size 512

# 2. HuggingFace 格式权重
python serve_openai_api.py --load_from ../out/hf_model

# 3. 带 LoRA
python serve_openai_api.py --weight full_sft --lora_weight lora_medical

# 4. MoE 模型
python serve_openai_api.py --weight dpo --hidden_size 640 --use_moe 1

# 启动后访问：http://localhost:8998/v1/chat/completions
```

---

## 第八章：serve_openai_api.py vs 其他推理方式的对比

### 8.1 各入口的定位

| 入口 | 场景 | 流式 | API | Web 界面 |
|------|------|------|-----|---------|
| `eval_llm.py` | 命令行调试 | ✅ | ❌ | ❌ |
| `serve_openai_api.py` | 服务部署 | ✅ | ✅（OpenAI 兼容） | ❌ |
| `chat_openai_api.py` | API 客户端测试 | ✅ | 调用端 | ❌ |
| `web_demo.py` | 可视化交互 | ✅ | ❌ | ✅（Streamlit） |

### 8.2 与 eval_llm.py 的区别

已在 eval_llm 练习 8 中详细对比过 `init_model` 部分。额外补充：

- `serve_openai_api.py` 的生成参数更少（没有 `top_k`、`repetition_penalty`），更适合生产环境的默认配置
- `serve_openai_api.py` 没有 `enable_thinking` 参数（reason 模型的 thinking 控制）
- `serve_openai_api.py` 通过 `apply_chat_template` 处理消息格式，`eval_llm.py` 也是

---

## 第九章：学习路线

建议学习顺序：

1. **阅读本章学习计划** — 理解整体架构（你现在在这里 ✅）
2. **逐段阅读源码**，对照本章理解
3. **动手练习**（见下方）

---

## 动手练习

### 基础

1. **启动服务**：用 `--weight full_sft --hidden_size 512` 启动 API 服务
2. **测试非流式**：用 curl 发一条请求
   ```bash
   curl http://localhost:8998/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"minimind","messages":[{"role":"user","content":"你好"}],"stream":false}'
   ```
3. **测试流式**：改 `"stream":true`，观察 SSE 事件格式

### 进阶

4. **阅读 `chat_openai_api.py`**：理解客户端如何调用该 API
5. **对比 `eval_llm.py` 和 `serve_openai_api.py` 的 init_model**：列出所有差异
6. **理解 CustomStreamer 的 Queue 机制**：如果不用 Queue，用 `yield from` 可以吗？思考线程安全

### 深入

7. **分析 prompt 截断问题**：`[-max_tokens:]` 是字符截断，如果对话历史全是中文（一个中文字符≈1.5 token），截断后可能仍然超长。如何改进？
8. **添加新参数**：给 `ChatRequest` 和 `model.generate()` 加上 `top_k` 和 `repetition_penalty` 参数
9. **并发思考**：当前实现是单线程的，如果两个请求同时到达会怎样？如何用锁或请求队列解决？
