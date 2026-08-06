"""chat_llm 方案B（滑动窗口 KV）的窗口构造与裁剪逻辑单元测试。"""
import os
import torch
from transformers import AutoTokenizer

from scripts.Deploy.chat_llm import trim_window, window_chunk_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tokenizer():
    return AutoTokenizer.from_pretrained(os.path.join(ROOT, 'scripts/Model'))


def test_window_chunk_first_turn_has_system():
    text = window_chunk_text(_tokenizer(), '你好', is_first=True, last_is_eos=False)
    assert text.startswith('<|im_start|>system')
    assert text.endswith('<|im_start|>assistant\n')


def test_window_chunk_after_eos_adds_newline():
    text = window_chunk_text(_tokenizer(), '你叫什么', is_first=False, last_is_eos=True)
    assert text.startswith('\n<|im_start|>user\n你叫什么')
    assert text.endswith('<|im_start|>assistant\n')


def test_window_chunk_after_truncation_fills_eos():
    text = window_chunk_text(_tokenizer(), '你叫什么', is_first=False, last_is_eos=False)
    assert text.startswith('<|im_end|>\n<|im_start|>user\n')


def test_trim_window_drops_front_and_offsets():
    # 缓存 K/V 布局 [batch, seq, kv_heads, head_dim]，seq 在 dim 1
    all_ids = torch.arange(10).unsqueeze(0)
    cache = [(torch.arange(10).view(1, 10, 1, 1), torch.arange(10).view(1, 10, 1, 1))]
    ids, c, off = trim_window(all_ids, cache, max_tokens=4, pos_offset=100)
    assert off == 106
    assert ids.shape[1] == 4
    assert c[0][0].shape == (1, 4, 1, 1)   # seq 维被裁到 4
    assert ids[0].tolist() == [6, 7, 8, 9]


def test_trim_window_noop_when_within_limit():
    all_ids = torch.arange(4).unsqueeze(0)
    ids, c, off = trim_window(all_ids, None, max_tokens=8, pos_offset=5)
    assert off == 5
    assert ids.shape[1] == 4
    assert c is None
