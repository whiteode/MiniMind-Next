"""基于 TextStreamer 的流式输出：把解码文本逐段送入 queue，供 FastAPI StreamingResponse 消费。

注意：skip_prompt=True 时 TextStreamer 会把第一次 put() 当作 prompt 跳过。
因此在自写解码循环（kv_generate.generate_kv）前，需先 put(prompt) 解锁，否则第一个生成 token 会被吞掉。
"""
from queue import Queue
from transformers import TextStreamer


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)
