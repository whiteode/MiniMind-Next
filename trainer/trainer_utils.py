"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """
    判断当前进程是否为主进程（Master Process）。
    
    用于在分布式训练中控制“仅由主进程执行”的操作，例如：
    - 打印日志 (print/logging)
    - 保存模型权重 (save checkpoint)
    - 记录可视化数据 (TensorBoard/Wandb)

    返回:
    -------
    bool
        如果当前是单机环境、未启动分布式训练，或者在分布式环境中属于主进程(Rank 0)，
        则返回 True；否则返回 False。
    """
    # 核心逻辑拆解：
    # 1. not dist.is_initialized():
    #    判断分布式进程组是否【未】初始化。
    #    如果返回 True，说明是普通的单 GPU 或单 CPU 训练（没有搞分布式），那当前进程自然就是“主进程”。
    #
    # 2. dist.get_rank() == 0:
    #    如果分布式已经初始化了，则获取当前进程的全局编号（Rank）。
    #    Rank 0 在分布式系统中被约定为主进程。
    #
    # 两者用 `or` 连接：未初始化分布式 OR 当前是 Rank 0，都算作主进程。
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    计算余弦退火（Cosine Annealing）学习率。
    
    学习率变化曲线呈余弦波形，前期下降慢，中期下降快，后期减速收敛。
    最终学习率会平滑衰减至初始学习率的 10%。

    参数:
    ----------
    current_step : int
        当前训练步数 (或当前 epoch)
    total_steps : int
        总训练步数 (或总 epoch)
    lr : float
        初始最大学习率

    返回:
    -------
    float
        当前步数对应调整后的学习率
    """
    # 核心公式拆解说明：
    # 1. math.pi * current_step / total_steps: 
    #    将训练进度映射到 [0, π] 的弧度区间。
    #
    # 2. math.cos(...): 
    #    余弦值随进度从 1.0 (训练开始) 逐渐变化到 0.0 (训练过半)，最终降至 -1.0 (训练结束)。
    #
    # 3. 1 + math.cos(...): 
    #    将余弦值区间缩放并平移到 [2.0, 0.0]。
    #
    # 4. 0.1 + 0.45 * (...):
    #    - 训练开始 (cos=1) : 0.1 + 0.45 * 2 = 1.0 (保持原学习率)
    #    - 训练过半 (cos=0) : 0.1 + 0.45 * 1 = 0.55 (降至 55%)
    #    - 训练结束 (cos=-1): 0.1 + 0.45 * 0 = 0.1  (保留 10% 底线，防止学习率归零导致模型停止学习)
    
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    核心配置与标识
    lm_config:

    含义：模型配置对象（MiniMindConfig）。

    作用：函数会读取它的 hidden_size 和 use_moe 属性，用来拼接保存的文件名（例如：pretrain_512_moe.pth）。这样你一眼就能通过文件名知道这个模型的大小和架构。

    weight='full_sft':

    含义：保存权重的文件前缀名。

    作用：区分不同的训练阶段。比如预训练时传入 pretrain，微调时传入 sft，这样不同阶段的进度不会互相覆盖。

    save_dir='../checkpoints':

    含义：存档文件夹的路径。

    作用：所有的 .pth 权重文件都会被扔到这个文件夹里。代码开头有 os.makedirs，如果文件夹不存在会自动创建。

    2. 训练状态（用于断点续传）
    model=None:

    含义：当前的神经网络模型实例。

    作用：

    如果是 None：函数进入“读档模式”，尝试从磁盘读取已有的进度。

    如果不是 None：函数进入“存档模式”，把模型的参数存起来。

    optimizer=None:

    含义：优化器实例（如 AdamW）。

    作用：极其重要！ 优化器里存着每个参数的“动量”信息。如果不存优化器，断点续训时就像跑步跑一半突然停下再起步，失去了之前的惯性，Loss 会瞬间跳变（Spike）。

    epoch / step:

    含义：当前训练到了第几轮、第几步。

    作用：记录进度。下次读档时，代码知道应该从哪里继续，以及如何恢复学习率曲线。

    3. 可视化与扩展
    wandb=None:

    含义：W&B（或 SwanLab）的日志对象。

    作用：它会保存当前实验的 run_id。如果你训练一半断网或者崩溃了，下次续训时，代码能自动连回到同一个网页实验页面，让曲线接上，而不是新开一个页面。

    **kwargs (关键字参数):

    含义：这是一个“万能口袋”。

    作用：如果你还有其他想存的东西（比如 scaler 混合精度缩放器、自定义的计数器等），可以直接传进来。函数内部会自动遍历它们：

    如果是带 state_dict 的对象（比如 scaler），就存它的状态。

    如果是普通变量，就直接原样保存。
    
    
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'
    """
    为什么要专门加一个 _resume 后缀？
    这个文件和普通的 .pth 权重文件有本质区别，它的“含金量”更高：

    普通权重文件 (ckp_path)：

    内容：只存 model.state_dict()（即神经元的权重数字）。

    特点：经过了 .half() 处理，文件小，适合部署到手机或服务器上跑推理。

    续训存档文件 (resume_path)：

    内容：是一个“大杂烩”字典，包含：

    模型权重。

    优化器状态 (optimizer)：包含 AdamW 的动量信息（这是最占空间但续训必须有的）。

    训练进度：当前的 epoch 和 step。

    环境信息：当时用了几张显卡 (world_size)。

    实验 ID：wandb_id，保证网页上的曲线能接上。

    特点：文件非常大（通常是普通权重的 3 倍左右），仅用于防止训练中断。
    """
    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        # 情况 A：模型被 torch.compile 编译过getattr 发现了 raw_model 里面有 _orig_mod 这个属性。于是它把这个隐藏的原始模型取出来，赋值给 raw_model。
        # 情况 B：模型是普通模型，没被编译过getattr 找不到 _orig_mod 属性。触发第三个参数（默认值），直接返回 raw_model 本身。
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        """
        这是为了防止： 如果在保存过程中电脑突然断电或程序崩溃，直接保存会导致旧的存档被覆盖，新的存档还没写完（文件损坏）。先写临时文件可以确保：要么保存成功，要么保留上一次完整的存档。
        """
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        """
        这个 wandb_id 拿来干什么？
        你可以把 wandb_id 想象成这局游戏的存档位编号。

        没有 ID 时：每次运行脚本，WandB 都会认为这是一个全新的实验（Run 1, Run 2...），网页上的 Loss 曲线会从 0 步重新开始画。

        有了 ID 并在续训时传入：


        # 续训时的逻辑
        wandb.init(project=..., id=wandb_id, resume='must')
        WandB 发现 ID 匹配，就会把新的数据追加在之前的曲线后面。这样你看到的 Loss 曲线就是连续的，能清晰地对比中断前后的模型表现
        """


        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        """
        World（世界）：在 PyTorch 的分布式通信包（torch.distributed）中，整个训练任务被看作一个“世界”。

        Size（大小）：这个世界里有多少个成员。

        代码逻辑：

        如果你是单机单卡训练，world_size = 1。

        如果你是单机 8 卡训练，world_size = 8。

        如果你是两台机器，每台 8 卡（多机多卡），world_size = 16。
        """


            
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)