"""MiniMind 的 OpenAI 兼容 API 服务：支持跨请求 prefix cache（--enable_kv）。"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import time
import torch
import warnings
import uvicorn

from dataclasses import dataclass
from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from scripts.Deploy.model_loader import add_model_args, config_from_args, init_model
from scripts.Deploy.kv_generate import SamplingParams, generate_kv
from scripts.Deploy.prefix_cache import MAX_PREFIX_CACHE, find_prefix, store_prefix
from scripts.Deploy.chat_streamer import CustomStreamer

warnings.filterwarnings('ignore')


@dataclass
class ServerState:
    """服务运行期状态：模型、分词器与是否启用 prefix cache。"""
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    enable_kv: bool = False


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = False
    tools: list = []


def generate_response(state: ServerState, new_prompt, max_new_tokens, temperature, top_p, streamer=None):
    """生成回复（仅返回新增 token 的 ids）。
    enable_kv 时走前缀缓存 + generate_kv，否则走 model.generate。"""
    tokenizer = state.tokenizer
    inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(state.model.device)

    if state.enable_kv:
        if streamer is not None:
            streamer.put(inputs.input_ids)   # 模拟 HF generate：先 put prompt（skip_prompt 会跳过），避免吞掉第一个生成 token
        full_ids = inputs.input_ids[0].tolist()
        prefix_len, past_key_values = find_prefix(full_ids)
        print(f'[PrefixCache] 前缀命中 {prefix_len}/{len(full_ids)} token，prefill 增量 {len(full_ids) - prefix_len}')
        # 只把增量喂给模型；前缀未命中（prefix_len=0）时 past_key_values=None → 全量 prefill
        chunk_ids = torch.tensor(full_ids[prefix_len:], dtype=torch.long, device=state.model.device).unsqueeze(0)
        all_ids = torch.tensor(full_ids, dtype=torch.long, device=state.model.device).unsqueeze(0)
        generated_ids, past_key_values, all_ids_new = generate_kv(
            state.model, chunk_ids, all_ids, past_key_values,
            eos_token_id=tokenizer.eos_token_id,
            params=SamplingParams(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p),
            streamer=streamer,
        )
        store_prefix(all_ids_new[0].tolist(), past_key_values)
        return generated_ids

    with torch.no_grad():
        generated_ids = state.model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + max_new_tokens,
            do_sample=True,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=top_p,
            temperature=temperature,
            streamer=streamer,
        )
    return generated_ids[:, inputs.input_ids.shape[1]:]


def generate_stream_response(state: ServerState, messages, temperature, top_p, max_tokens):
    try:
        new_prompt = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        queue = Queue()
        streamer = CustomStreamer(state.tokenizer, queue)

        def _generate():
            generate_response(state, new_prompt, max_tokens, temperature, top_p, streamer=streamer)

        Thread(target=_generate).start()

        while True:
            text = queue.get()
            if text is None:
                yield json.dumps({
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }, ensure_ascii=False)
                break

            yield json.dumps({
                "choices": [{"delta": {"content": text}}]
            }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        try:
            if request.stream:
                return StreamingResponse(
                    (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                        state=state,
                        messages=request.messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens,
                    )),
                    media_type="text/event-stream"
                )

            new_prompt = state.tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            generated_ids = generate_response(
                state, new_prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            answer = state.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"
                    }
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    add_model_args(parser)
    parser.add_argument('--enable_kv', default=False, action='store_true', help="启用跨请求 prefix cache：多轮对话复用已算 K/V，只 prefill 新增部分")
    args = parser.parse_args()

    model, tokenizer = init_model(config_from_args(args))
    state = ServerState(model=model, tokenizer=tokenizer, enable_kv=args.enable_kv)
    if args.enable_kv:
        print(f'[Prefix Cache] 已启用，最多缓存 {MAX_PREFIX_CACHE} 个会话的 K/V')
    uvicorn.run(create_app(state), host="0.0.0.0", port=8998)


if __name__ == "__main__":
    main()
