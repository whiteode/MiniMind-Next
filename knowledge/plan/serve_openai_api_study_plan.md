# serve_openai_api.py 学习计划

> **文件位置**: `scripts/Deploy/serve_openai_api.py`（197 行）
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

**Q: 要支持 function calling 取决于什么？**

支持 function calling 依赖以下几个层面，缺一不可：

**1. 训练数据（最核心）**

模型必须**在训练阶段见过** function calling 的样本，学会：
- **判断何时调用工具**：用户说"今天北京天气怎么样？" → 模型输出 `{"name": "get_weather", "arguments": {"city": "北京"}}`，而不是编造一个答案
- **正确格式化调用**：输出结构化的 JSON，参数名和类型正确
- **把工具返回结果融合到回答中**：工具返回 `{"temp": 25, "condition": "晴"}` → 模型说"北京今天25°C，晴天"

没有这类训练数据，模型根本不知道有工具这回事——它只会照常生成闲聊文本。

**2. Chat Template（MiniMind 已经部分支持）**

`model/tokenizer_config.json` 里的 chat template 已有工具格式化逻辑：

```jinja
{%- if tools %}
{{- '<|im_start|>system\n' }}
    # Tools
    
    You may call one or more functions to assist with the user query.
    
    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {{ tool | tojson }}
    </tools>
    ...
{%- endif %}
```

当传入 `tools` 参数时，template 会自动把工具定义以 XML 格式嵌入 system prompt。这部分**基础设施是有的**。

**3. API 层（serve_openai_api.py 没做）**

当前代码虽然接收了 `tools: list = []`，但**扔掉没用**：

```python
new_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    # 没传 tools=request.tools  ← 工具定义被丢弃
)
```

要让它工作，需要改成：
```python
new_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=request.tools  # ← 把工具定义传进去
)
```

**4. 推理循环（需要额外逻辑）**

True function calling 还需要**多轮交互**：
```
用户: "北京天气怎么样？"
模型: <tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>
  → 服务端解析模型输出，执行 get_weather("北京") → {"temp": 25, "condition": "晴"}
  → 把结果作为 tool 角色塞回 messages
模型: "北京今天25°C，天气晴朗。"
```

这需要服务端**解析模型输出**、**执行工具函数**、**把结果注入下一轮对话**——serve_openai_api.py 完全没做这层。

**5. 模型参数量**

MiniMind Small 只有 26M 参数。function calling 涉及意图识别 → 参数提取 → 结构化输出 → 结果融合的复杂推理，通常需要 7B+ 级别的模型才能稳定做到。26M 模型即使喂了训练数据，效果也可能很差。

**一句话总结**：

| 依赖 | 状态 |
|------|------|
| 训练数据 → 模型学会"有工具时该干什么" | ❌ MiniMind 没训过 |
| Chat Template → 把工具定义格式化给模型看 | ✅ 已支持 |
| API 层 → 把 tools 传给 template | ❌ serve_openai_api.py 没传 |
| 推理循环 → 解析工具调用 + 执行 + 结果回注 | ❌ 完全没实现 |
| 模型容量 → 26M 太小可能不稳定 | ⚠️ 即使其他全做了也可能不可靠 |

所以 `tools: list = []` 现在确实只是个占位符。

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

**Q: `on_finalized_text(text, stream_end)` 这个函数干嘛的？**

这是 HuggingFace `TextStreamer` 的一个回调方法。

**触发机制**：

`model.generate()` 内部每次生成一个 token 后，会按顺序调用两个回调：

| 回调 | 调用时机 | 用途 |
|------|---------|------|
| `on_token_ended(token_id, ...)` | **每个 token** 生成后 | token 级别通知（原始 token ID） |
| `on_finalized_text(text, stream_end)` | **每个完整片段**可输出后 | 文本级别通知（解码后的字符串） |

`on_finalized_text` 不一定每个 token 调用一次。TextStreamer 内部有一个**缓存机制**：

```
输入: "今天我们一起去"
生成: "公" "园" "散" "步" "吧"

实际的 on_finalized_text 回调:
回调1: text="公园散"    ← 缓存了"公园"，遇到"散"时 flush
回调2: text="步吧"      ← 缓存了"步"，遇到"吧"时 flush
回调3: text="" stream_end=True  ← EOS token，通知结束
```

**参数含义**：
- `text`：已解码的文本片段（可能包含多个 token 合并的结果）
- `stream_end`：`True` 表示模型生成了 EOS token，生成结束

**为什么不能直接在回调里 yield？**

因为 `model.generate()` 是同步阻塞调用，而 `on_finalized_text` 是回调不是生成器。假如直接在里面 yield：

```python
# ❌ 错误尝试
def on_finalized_text(self, text, stream_end):
    yield text  # 生成器嵌套，外层捕获不到这个 yield
```

所以需要 **Queue + Thread** 的桥接模式：回调往队列里放，主线程从队列里取并 yield。

**调用链路全貌**：

```
model.generate()
  └─> TextStreamer.on_token_ended(token_id)     ← 每个 token 生成后
       └─> decode + 缓存 → on_finalized_text()  ← 解码出完整文本片段
            └─> CustomStreamer.on_finalized_text(text)
                 └─> queue.put(text)            ← 放到线程安全队列
                      └─> 生成器 yield text      ← 编码成 SSE 事件
                           └─> FastAPI StreamingResponse → 客户端收到
```

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

### 深入理解：同步阻塞 + 为什么不能直接在回调里 yield

**一、什么是同步阻塞调用？**

`model.generate()` 是一个同步阻塞函数。拆开理解：

- **同步**：调用者发起调用后，必须等待函数返回才能继续执行下一行代码
- **阻塞**：函数执行期间，当前线程被"卡住"，不能做任何其他事情

把线程想象成一个流水线工人：
```
同步阻塞（实际行为）:
  工人: 启动机器 → 站在旁边盯着 → 等到机器停了 → 拿结果 → 去干下一件事
  ↑ generate() 期间线程什么事都做不了，只能等
```

如果要类比异步非阻塞：
```
异步非阻塞（Triton/TensorRT 推理引擎）:
  工人: 启动机器 → 去干别的活 → 机器出结果时被通知 → 回来拿结果
  ↑ PyTorch 原生不支持，需要专门的推理引擎
```

**二、为什么不能在回调里 yield？**

Python 的 `yield` 只在生成器函数内部有效，不会"穿透"函数调用边界：

```python
def callback():
    yield "hello"       # 这是 callback 自己的生成器，跟外层无关

def my_generator():
    callback()           # callback() 的 yield 被丢弃了！
    yield "world"        # 这是 my_generator 唯一能 yield 的

gen = my_generator()
print(next(gen))  # "world" — callback 的 yield 根本没被捕获
```

套到 TextStreamer 场景：

```python
class CustomStreamer(TextStreamer):
    def on_finalized_text(self, text, stream_end):
        yield text  # ❌ 双重问题
```

**问题一：同步阻塞**。`model.generate()` 在运行中，整个执行栈是：`generate_stream_response → generate → on_finalized_text`。`on_finalized_text` 必须**返回**后，`generate` 才能继续算下一个 token。如果在里面 `yield`，这个 `yield` 属于 `on_finalized_text` 自己变成的生成器，跟外层的 `generate_stream_response` 毫无关系。

**问题二：yield 不穿透**。即使忽略同步阻塞，Python 的 yield 也不会跨函数调用传递：
```
generate_stream_response → generate → on_finalized_text → yield
                                                             ↑
                                                   这个 yield 属于最内层函数
                                                   外层完全收不到
```

**三、把 generate 放线程的真正原因**

```python
# ❌ 单线程：死锁
def generate_stream_response(...):
    model.generate(streamer=streamer)  # 同步阻塞，卡在这里不走了
    while True:                        # 永远不会执行到这里
        text = queue.get()
        yield text

# ✅ 双线程：生产者-消费者
def generate_stream_response(...):
    # 子线程 → 生产（跑 model.generate，每出一个 token 就 put 到 queue）
    Thread(target=lambda: model.generate(streamer=streamer)).start()
    # 主线程 → 消费（从 queue 取 token，yield 给 FastAPI）
    while True:
        text = queue.get()   # 阻塞等，但不会死锁——子线程正在跑 generate
        if text is None:
            break
        yield text           # yield 给 FastAPI 的 StreamingResponse
```

两个线程各自阻塞在不同的地方，互不干扰：
```
时间轴 →

主线程:  setup → T.start() → queue.get() ←阻塞→ 拿到"你好" → yield → queue.get() ←阻塞→ ...
                                ↑等待                          ↑等待
子线程:              generate() → 算token1 → put("你好") → 算token2 → put("！") → ...
```

**四、本质视角：回调世界 ↔ 生成器世界**

| | 回调（Callback） | 生成器（Generator） |
|---|---|---|
| 数据流向 | 被调用方推给调用方 | 调用方逐次拉取 |
| 控制权 | 被调用方主动推送 | 调用方控制节奏 |
| 典型模式 | 观察者模式 | 迭代器模式 |

`on_finalized_text` 是典型的**回调**——`model.generate` 内部主动推送数据。但 FastAPI 的 `StreamingResponse` 要的是**生成器**——逐条拉取数据发给客户端。

**Queue + Thread 就是这两个世界之间的桥梁**：

```
回调世界:  model.generate → on_finalized_text → queue.put(text)
                                                    ↓ Queue（线程安全缓冲区）
生成器世界:  queue.get() → yield text → FastAPI → 客户端
```

**Q: yield 能不能跨线程用？**

不能。更准确地说是 **yield 不能跨函数边界用** —— 跟线程无关。

核心规则：`yield` 只属于它直接所在的函数，不穿透到调用方。

```python
# 纯语法规则，同一个线程内：
def inner():
    yield "inner的值"     # ← 这个 yield 属于 inner

def outer():
    inner()                # inner() 返回一个生成器对象，被丢弃了
    yield "outer的值"      # ← outer 只能 yield 自己的

list(outer())  # → ["outer的值"]，inner 的 yield 消失了
```

加上线程也改变不了这个规则：
```python
def on_finalized_text(text, stream_end):
    yield text              # ← 属于 on_finalized_text，不属于外层

model.generate(streamer=...)  # 内部回调 on_finalized_text
```

要把回调里的值传给外层生成器，需要 **Queue 做桥梁**：回调 `put`，外层 `get` 再 `yield`。Queue 本身是线程安全的，解决了"跨线程共享数据"的问题。

**那如果不用 model.generate()，自己写推理循环呢？**

可以纯 yield 实现流式，不需要线程：

```python
def generate_stream(messages, ...):
    inputs = tokenizer(...)
    generated = inputs.input_ids
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(generated).logits[:, -1, :]  # 只看最后一位
        next_token = sample(logits, temperature, top_p)
        yield tokenizer.decode(next_token[0])           # ✅ 纯生成器
        generated = torch.cat([generated, next_token], dim=-1)
```

但代价是**每步都重新计算所有历史 token 的 KV**，比 `model.generate()` 慢几十倍（后者内部有 KV Cache 优化）。所以代码选了线程 + `model.generate()` —— 牺牲写法简洁性，换取性能。

```
                         ┌──────────────────────────────────────────┐
                         │              子线程（生成）               │
                         │                                          │
  ┌──────────────────┐   │  Thread(target=_generate).start()        │
  │ 主线程（API响应）  │   │                                          │
  │                  │   │  ┌──────────────────┐                    │
  │  while True:     │   │  │ model.generate() │                    │
  │                  │   │  │                    │                   │
  │    queue.get()◄───────────┐                │                   │
  │    │             │   │  │ │ on_finalized_   │                   │
  │    └─ yield ◄───────  │  │  text("你好") ─────put("你好")──┐    │
  │    "你好"       │   │  │                    │             │    │
  │                  │   │  │ on_finalized_   │             │    │
  │    queue.get()◄───────────┐ text("！") ──────put("！")───┤    │
  │    │             │   │  │ │                    │             │    │
  │    └─ yield ◄───────  │  │ on_finalized_   │             │    │
  │    "！"         │   │  │  text("")          │             │    │
  │                  │   │  │  stream_end=True ───put(None)────┘    │
  │    queue.get()   │   │  └──────────────────┘                    │
  │    │ → None      │   └──────────────────────────────────────────┘
  │    └─ break      │
  └──────────────────┘
         │
         ▼
  StreamingResponse (SSE)
         │
         ▼
    客户端收到流式结果
```

**关键理解**：
- 子线程负责**生产**（跑 `model.generate`，每出一个 token 就往队列里放）
- 主线程负责**消费**（不断 `queue.get()` 阻塞等待，拿到一个就 `yield` 一个）
- `Queue` 是线程安全的，子线程 `put` 和主线程 `get` 不会冲突
- `None` 是结束哨兵：子线程放 `None` → 主线程读到 `None` → `break` 退出循环

**Q: put 后没被 get 拿走，子线程会一直卡住吗？**

**不会，正好反过来。** 这取决于 Queue 的 `maxsize` 参数。

当前代码 `queue = Queue()` 没有传参，`maxsize=0` 表示**无上限**，`put()` 永远不阻塞：

```python
# 子线程（生成者）：
queue.put("你好")    # 队列空 → 直接放入，立即返回 ✅
queue.put("！")      # 不管队列多长 → 直接放入，立即返回 ✅

# 主线程（消费者）：
queue.get()          # 队列空 → 阻塞等待，直到有数据 🔄
queue.get()          # 队列有数据 → 立即取走 ✅
```

**真正卡住的是 `get()` 不是 `put()`**。

时间轴：
```
子线程:  gen() → put("你好") → gen() → put("！") → gen() → put("吗") → put(None) → 结束
         不阻塞 ✓   继续算    不阻塞 ✓  继续算    不阻塞 ✓  不阻塞 ✓
         
主线程:  get() ──阻塞──→ 拿到"你好" → yield → get() ──阻塞──→ 拿到"！" → yield → get() → None → break
         ↑ 队列空，等着               ↑ 队列又空了，等着
```

如果设了 `maxsize=1`（队列最多存 1 个元素），情况才反过来：
```python
queue = Queue(maxsize=1)
queue.put("你好")   # 队列空 → 放入 ✓，队列满了
queue.put("！")     # ❌ 队列已满 → 阻塞！等主线程 get() 取走才能继续放
```

当前代码用无上限队列是因为：**要保证 `model.generate()` 不被阻塞**。`on_finalized_text` 回调必须快速返回，否则 GPU 计算的下一轮循环会被拖慢。`put` 不阻塞 → 回调秒回 → GPU 全速推理。

**一句话**：当前代码是 `put()` 永远不阻塞，`get()` 在队列空时阻塞。子线程一直在跑，主线程等着消费子线程的产出。

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

**流式和非流式的格式区别**：

**非流式（stream=False）**—— 一次 HTTP 响应返回完整 JSON：

```json
{
    "id": "chatcmpl-1741852800",
    "object": "chat.completion",
    "created": 1741852800,
    "model": "minimind",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
        "finish_reason": "stop"
    }]
}
```

**流式（stream=True）**—— 多行 SSE 事件，每行以 `data: ` 开头、`\n\n` 结尾：

```
data: {"choices": [{"delta": {"content": "你好"}}]}

data: {"choices": [{"delta": {"content": "！"}]}}

data: {"choices": [{"delta": {"content": "有"}]}}

data: {"choices": [{"delta": {"content": "什"}]}}

...

data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}}

data: [DONE]
```

**核心区别对比**：

| 维度 | 非流式 | 流式 |
|------|--------|------|
| HTTP 响应次数 | 1 次拿到全部 | N 次（每个 token 一个事件） |
| 内容到达 | 一次性 | 逐步到达 |
| 字段名 | `message`（含 role+content） | `delta`（只有 content） |
| 停止信号 | `finish_reason` 在 message 里 | `finish_reason` 在 delta 里 |
| 客户端解析 | `response.choices[0].message.content` | `chunk.choices[0].delta.content` |

客户端代码差异（来自 `chat_openai_api.py`）：

```python
# 非流式
assistant_res = response.choices[0].message.content

# 流式
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
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
# 1. 原生 .pth 权重（默认路径 ../scripts/Model/ + ../out/）
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
