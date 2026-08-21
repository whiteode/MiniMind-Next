"""训练脚本公共设施：统一 CLI 参数、环境初始化、续训/保存、epoch 循环与 RL 奖励。

9 个 train_*.py 过去各自复制了约 60 行样板（argparse / 分布式 / 种子 / wandb /
autocast / 模型加载 / 保存 / epoch 循环 / calculate_rewards），这里收敛为可复用
函数，各脚本只保留算法本身。
"""
import os
import re
import sys

sys.path.insert(0, os.getcwd())

from dataclasses import dataclass

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from scripts.Model.model_minimind import MiniMindConfig
from scripts.Trainer.trainer_utils import (
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
)

DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


@dataclass
class TrainCtx:
    """单次训练运行上下文：run() 组装，train_epoch 经 ctx 取用模型/优化器/参数（无需模块全局变量）。"""
    args: object
    lm_config: object
    model: object
    tokenizer: object
    optimizer: object
    scaler: object
    autocast_ctx: object
    wandb: object = None
    lora_params: object = None      # lora 阶段
    ref_model: object = None        # dpo 阶段
    teacher_model: object = None    # distillation 阶段


# ---------------------------------------------------------------------------
# CLI 参数
# ---------------------------------------------------------------------------

def add_train_args(parser, **defaults):
    """注册 9 个训练脚本共用的 CLI 参数；各脚本差异默认值通过 **defaults 覆盖。

    传 from_weight=None 可跳过该参数（RL / 蒸馏脚本用自己的 from_* 参数）。
    """
    D = dict(
        save_dir='models',
        save_weight='model',
        epochs=1,
        batch_size=32,
        learning_rate=1e-4,
        device=DEFAULT_DEVICE,
        dtype='bfloat16',
        num_workers=8,
        accumulation_steps=1,
        grad_clip=1.0,
        log_interval=100,
        save_interval=1000,
        hidden_size=512,
        num_hidden_layers=8,
        max_seq_len=340,
        use_moe=0,
        data_path='resource/minimind_dataset/data.jsonl',
        from_weight='none',
        from_resume=0,
        use_wandb=False,
        wandb_project='MiniMind',
        use_compile=0,
    )
    D.update(defaults)

    parser.add_argument('--save_dir', type=str, default=D['save_dir'], help='模型保存目录')
    parser.add_argument('--save_weight', type=str, default=D['save_weight'], help='保存权重的前缀名')
    parser.add_argument('--epochs', type=int, default=D['epochs'], help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=D['batch_size'], help='batch size')
    parser.add_argument('--learning_rate', type=float, default=D['learning_rate'], help='初始学习率')
    parser.add_argument('--device', type=str, default=D['device'], help='训练设备')
    parser.add_argument('--dtype', type=str, default=D['dtype'], help='混合精度类型（bfloat16 / float16）')
    parser.add_argument('--num_workers', type=int, default=D['num_workers'], help='DataLoader 线程数')
    parser.add_argument('--accumulation_steps', type=int, default=D['accumulation_steps'], help='梯度累积步数')
    parser.add_argument('--grad_clip', type=float, default=D['grad_clip'], help='梯度裁剪阈值')
    parser.add_argument('--log_interval', type=int, default=D['log_interval'], help='日志打印间隔')
    parser.add_argument('--save_interval', type=int, default=D['save_interval'], help='模型保存间隔')
    parser.add_argument('--hidden_size', type=int, default=D['hidden_size'], help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', type=int, default=D['num_hidden_layers'], help='隐藏层数量')
    parser.add_argument('--max_seq_len', type=int, default=D['max_seq_len'], help='训练的最大截断长度')
    parser.add_argument('--use_moe', type=int, default=D['use_moe'], choices=[0, 1], help='是否使用MoE架构（0=否，1=是）')
    parser.add_argument('--data_path', type=str, default=D['data_path'], help='训练数据路径')
    if D['from_weight'] is not None:
        parser.add_argument('--from_weight', type=str, default=D['from_weight'], help='基于哪个权重训练（none=从头开始）')
    parser.add_argument('--from_resume', type=int, default=D['from_resume'], choices=[0, 1], help='是否自动检测&续训（0=否，1=是）')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    parser.add_argument('--wandb_project', type=str, default=D['wandb_project'], help='wandb项目名')
    parser.add_argument('--use_compile', type=int, default=D['use_compile'], choices=[0, 1], help='是否使用torch.compile加速（0=否，1=是）')
    return parser


# ---------------------------------------------------------------------------
# 环境 / 配置 / 训练设施
# ---------------------------------------------------------------------------

def init_train_env(args):
    """分布式初始化 + 固定随机种子 + 建目录。返回 local_rank。"""
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f'cuda:{local_rank}'
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    return local_rank


def make_lm_config(args, **overrides):
    """构造 MiniMindConfig；overrides 可覆盖（如 RL 脚本 max_seq_len=prompt+gen）。"""
    return MiniMindConfig(
        hidden_size=overrides.pop('hidden_size', args.hidden_size),
        num_hidden_layers=overrides.pop('num_hidden_layers', args.num_hidden_layers),
        use_moe=overrides.pop('use_moe', bool(args.use_moe)),
        **overrides,
    )


def make_autocast_ctx(args):
    """按 device/dtype 构造 autocast 上下文（CPU 用 nullcontext）。"""
    if 'cuda' not in args.device:
        return nullcontext()
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    return torch.amp.autocast('cuda', dtype=dtype)


def init_wandb(args, run_name, ckp_data=None):
    """按需初始化 swanlab（仅主进程）。返回 wandb 句柄或 None。"""
    if not (args.use_wandb and is_main_process()):
        return None
    import swanlab as wandb
    wandb_id = ckp_data.get('wandb_id') if ckp_data else None
    wandb.init(project=args.wandb_project, name=run_name,
               id=wandb_id, resume='must' if wandb_id else None)
    return wandb


def get_ckp_data(args, lm_config, save_weight=None):
    """读取续训存档（from_resume=1 时）。"""
    if args.from_resume != 1:
        return None
    return lm_checkpoint(lm_config, weight=save_weight or args.save_weight, save_dir='checkpoints')


def load_resume_state(ckp_data, model, optimizer, *, scaler=None, scheduler=None,
                      extra=(), strict=True):
    """从续训档恢复状态，返回 (start_epoch, start_step)。extra 为 [(属性名, 对象), ...]。"""
    if not ckp_data:
        return 0, 0
    # 兼容 torch.compile 包装（OptimizedModule 的 _orig_mod）或 DDP 包装（module）
    target_model = getattr(model, '_orig_mod', model)
    target_model = getattr(target_model, 'module', target_model)
    target_model.load_state_dict(ckp_data['model'], strict=strict)
    optimizer.load_state_dict(ckp_data['optimizer'])
    if scaler is not None:
        scaler.load_state_dict(ckp_data['scaler'])
    if scheduler is not None:
        scheduler.load_state_dict(ckp_data['scheduler'])
    for name, obj in extra:
        if obj is not None and name in ckp_data:
            obj.load_state_dict(ckp_data[name])
    return ckp_data['epoch'], ckp_data.get('step', 0)


def wrap_ddp(model, local_rank):
    """按需包 DistributedDataParallel（并忽略 RoPE 常量 buffer）。"""
    if not dist.is_initialized():
        return model
    model._ddp_params_and_buffers_to_ignore = {'freqs_cos', 'freqs_sin'}
    return DistributedDataParallel(model, device_ids=[local_rank])


def for_each_epoch(train_ds, args, start_epoch, start_step, *, train_sampler=None, saved_indices=None):
    """逐 epoch 产出 (epoch, loader, iters, start_step_effective)。

    负责 set_epoch / 每轮种子 / 打乱索引 / 断点续训跳过 / SkipBatchSampler / 跳过日志。
    saved_indices：预训练续训时恢复上一轮的打乱顺序。
    """
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        if saved_indices is not None and epoch == start_epoch:
            indices = saved_indices
        else:
            indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        iters = len(loader) + skip
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
        yield epoch, loader, iters, (start_step if skip > 0 else 0), indices


def save_checkpoint(lm_config, model, optimizer, args, epoch, step, *,
                    iters=None, save_weight=None, scaler=None, scheduler=None, wandb=None,
                    extra_state=None, weight_saver=None):
    """保存 checkpoints 续训档；仅在所有 Epoch 完全训练结束（最终步）时才保存到 models/ 目录。

    weight_saver(model, path)：自定义权重写入（如 LoRA 的 save_lora）。
    """
    if not is_main_process():
        return
    save_weight = save_weight or args.save_weight

    # 1. 始终保存 checkpoints/ 续训快照（用于中断后 --from_resume 1 恢复）
    lm_checkpoint(lm_config, weight=save_weight, model=model, optimizer=optimizer,
                  epoch=epoch, step=step, wandb=wandb, save_dir='checkpoints',
                  scaler=scaler, scheduler=scheduler, **(extra_state or {}))

    # 2. 仅在全部训练完成（最后一个 Epoch 的最后一个 step）时，才导出最终权重到 models/
    is_final_step = (epoch == args.epochs - 1) and (iters is not None and step == iters - 1)
    if is_final_step:
        model.eval()
        if weight_saver is None:
            raw = getattr(model, 'module', model)
            raw = getattr(raw, '_orig_mod', raw)
            moe_suffix = '_moe' if lm_config.use_moe else ''
            path = f'{args.save_dir}/{save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            state = {k: v.half().cpu() for k, v in raw.state_dict().items()}
            torch.save(state, path)
            del state
        else:
            path = f'{args.save_dir}/{save_weight}_{lm_config.hidden_size}.pth'
            weight_saver(model, path)
        Logger(f'🎉 训练全部完成！最终模型权重已保存至: {path}')
        model.train()


# ---------------------------------------------------------------------------
# RL 奖励（GRPO / PPO / SPO 共用）
# ---------------------------------------------------------------------------

def reasoning_format_rewards(responses, device):
    """推理模型格式奖励：<think>/<answer> 标签齐全 +0.5，四个标签各出现一次各 +0.25。"""
    patterns = (
        r'^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$',
        r'^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$',
    )
    rewards = []
    for response in responses:
        r = 0.5 if any(re.match(p, response, re.S) for p in patterns) else 0.0
        for tag in ('<think>', '</think>', '<answer>', '</answer>'):
            if response.count(tag) == 1:
                r += 0.25
        rewards.append(r)
    return torch.tensor(rewards, device=device)


def score_response(prompt, response, reward_model, reward_tokenizer, *, reasoning=False, scale=3.0):
    """对单条 (prompt, response) 打分；推理模型叠加 answer 段 0.4/0.6 加权。"""
    pattern = r'<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>'
    matches = re.findall(pattern, prompt, re.DOTALL)
    messages = [{'role': role, 'content': content.strip()} for role, content in matches]

    chat = messages + [{'role': 'assistant', 'content': response}]
    score = max(min(reward_model.get_score(reward_tokenizer, chat), scale), -scale)

    if reasoning:
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            chat = messages + [{'role': 'assistant', 'content': answer_match.group(1).strip()}]
            answer_score = max(min(reward_model.get_score(reward_tokenizer, chat), scale), -scale)
            score = score * 0.4 + answer_score * 0.6
    return score


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer, *,
                      reasoning, device, num_generations=None):
    """统一奖励：格式奖励（可选）+ reward 模型打分。

    num_generations 给定（GRPO：每个 prompt 生成 N 条）时按 N 展开；否则一一对应（PPO/SPO）。
    """
    rewards = reasoning_format_rewards(responses, device) if reasoning \
        else torch.zeros(len(responses), device=device)
    if num_generations:
        scores = [
            score_response(prompts[i], responses[i * num_generations + j],
                           reward_model, reward_tokenizer, reasoning=reasoning)
            for i in range(len(prompts)) for j in range(num_generations)
        ]
    else:
        scores = [score_response(p, r, reward_model, reward_tokenizer, reasoning=reasoning)
                  for p, r in zip(prompts, responses)]
    return rewards + torch.tensor(scores, device=device)
