"""web_demo 的纯工具函数（不依赖 streamlit，便于复用与测试）。"""
import os
import random
import re

import numpy as np
import torch


MODEL_PATHS = {
    "MiniMind2-R1 (0.1B)": ["MiniMind2-R1", "MiniMind2-R1"],
    "MiniMind2-Small-R1 (0.02B)": ["MiniMind2-Small-R1", "MiniMind2-Small-R1"],
    "MiniMind2 (0.1B)": ["MiniMind2", "MiniMind2"],
    "MiniMind2-MoE (0.15B)": ["MiniMind2-MoE", "MiniMind2-MoE"],
    "MiniMind2-Small (0.02B)": ["MiniMind2-Small", "MiniMind2-Small"]
}


# 原生 .pth 权重可选的训练阶段（resource/MiniMind2-PyTorch/<weight>_<hidden_size>[__moe].pth）
NATIVE_WEIGHTS = ["pretrain", "full_sft", "dpo", "reason", "ppo_actor", "grpo", "spo"]


def resolve_model_path(name):
    """优先使用本地 resource/<name>（离线可用），否则退回 HF repo id（需联网）。"""
    local = os.path.join('resource', name)
    return local if os.path.isdir(local) else name


def seed_generation():
    """为一次生成设置随机种子。"""
    setup_seed(random.randint(0, 2 ** 32 - 1))


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_assistant_content(content, show_thinking=False):
    """把 <think>...</think> 渲染成可折叠 HTML（仅思维链模型需要）。"""
    if not show_thinking:
        return content

    if '<think>' in content and '</think>' in content:
        content = re.sub(r'(<think>)(.*?)(</think>)',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\2</details>',
                         content, flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        content = re.sub(r'<think>(.*?)$',
                         r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理中...</summary>\1</details>',
                         content, flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        content = re.sub(r'(.*?)</think>',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\1</details>',
                         content, flags=re.DOTALL)

    return content
