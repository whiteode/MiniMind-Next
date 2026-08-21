"""kv_generate 的采样逻辑（apply_sampling / sample_token）单元测试。"""
import torch
from scripts.Deploy.kv_generate import SamplingParams, apply_sampling, sample_token


def test_top_p_masks_low_prob():
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0, -5.0]])
    filtered = apply_sampling(logits, temperature=1.0, top_p=0.9, top_k=0)
    assert filtered[0, -1] == float('-inf')   # 尾部低概率被屏蔽


def test_top_k_keeps_only_top_k():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
    filtered = apply_sampling(logits, temperature=1.0, top_p=1.0, top_k=2)
    assert (filtered[0] > float('-inf')).sum().item() == 2


def test_repetition_penalty_penalizes_seen():
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    filtered = apply_sampling(logits, temperature=1.0, top_p=1.0, top_k=0,
                              repetition_penalty=2.0, seen=torch.tensor([0]))
    # token0 的正 logit 被除以 2 → 1.5，低于未惩罚的 token1 的 2.0
    assert filtered[0, 0] == 1.5


def test_sampling_deterministic_with_seed():
    torch.manual_seed(0)
    logits = torch.randn(1, 100)
    a = sample_token(logits.clone(), temperature=1.0)
    torch.manual_seed(0)
    b = sample_token(logits.clone(), temperature=1.0)
    assert torch.equal(a, b)


def test_sampling_params_defaults():
    p = SamplingParams()
    assert p.max_new_tokens == 8192
    assert p.temperature == 0.85
    assert p.top_p == 0.85
