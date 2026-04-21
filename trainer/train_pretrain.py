import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    单轮训练的核心函数。
    和 CNN 一样，遍历 DataLoader，计算 loss，反向传播。

    epoch
    当前正在执行的训练轮次编号（从 0 开始）。函数内部用它来记录日志、计算学习率、显示进度等。

    loader
    一个 DataLoader 实例，负责迭代当前轮的数据。它产生一批批的 (input_ids, labels) 供模型训练。

    iters
    本轮总的迭代步数（也就是 len(loader) + 可能跳过的步数）。用于计算学习率调度、打印 ETA 以及进度显示。

    start_step=0
    可选参数，默认 0。若从某个中断点（checkpoint）恢复训练，可以传入上次结束时的步数，这样循环会从 start_step+1 开始计数，日志和保存逻辑也会正确。

    wandb=None
    用于可视化的 W&B（或替代的 swanlab）对象。传入时函数会在每次日志输出时调用 wandb.log；未传或为 None 则不记录。
    """
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # input_ids: 输入的文本 Token ID 序列，形状通常是 (Batch_Size, Max_Seq_Len)
        # labels: 目标 Token ID 序列。在预训练中，其实就是把 input_ids 往左移了一位（预测下一个词）
        """
        LLM 的核心任务是“文字接龙”。
        假设我们有一句话：“我 爱 人工 智能”，被切分成了 4 个 Token。

        输入 (input_ids)： [我, 爱, 人工]

        目标 (labels)： [爱, 人工, 智能]
        这就叫“往左移了一位”。模型在第 1 个时间步看到“我”，它要预测“爱”；在第 2 个时间步看到“我 爱”，要预测“人工”，以此类推。在代码实现中，往往是输入一整句 [我, 爱, 人工, 智能]，然后在内部通过掩码（Mask）机制防止它看到后面的词，直接计算它对下一个词的预测误差。
        """
        # 1. 动态学习率：LLM 训练极其依赖学习率调度（通常是 Warmup + 余弦退火 Cosine Decay）
        """
        这段代码实现的是一个经典的“余弦退火”学习率曲线。公式是：$$lr_{current} = lr_{initial} \times \left( 0.1 + 0.45 \times \left(1 + \cos\left(\pi \times \frac{current\_step}{total\_steps}\right)\right) \right)$$
        为什么这么算？ 随着 current_step 从 0 增加到 total_steps，$\cos$ 的输入从 $0$ 变到 $\pi$，$\cos$ 的值从 $1$ 降到 $-1$。结果： 整个括号里的乘数会从 $0.1 + 0.45 \times (1+1) = 1.0$，平滑地下降到 $0.1 + 0.45 \times (1-1) = 0.1$。意义： 训练刚开始时，学习率最大（100%），模型快速学习；到了训练末期，学习率降到初始值的 10%，让模型进行细微的调整（收敛）。这比固定的学习率效果好得多。
        
        """
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        """
    
        在 PyTorch 中，optimizer 维护着一组参数（params），而每组参数都会对应一个参数组（param_group）。
        optimizer.param_groups 就是一个列表，里面的每个元素都是一个字典，描述了一组参数以及它们对应的超参数设置。

        常见字段
        字段	说明
        params	这一组要优化的参数列表（Iterable[Tensor]）。
        lr	当前学习率（可在训练中被调度器或手动修改）。
        weight_decay	L2 正则化系数（如果用了的话）。
        betas	Adam/AdamW 等优化器的动量系数 (β₁, β₂)。
        eps	防止除零的微小常数。
        amsgrad	AdamW 中是否使用 AMSGrad 变体。
        其他特定优化器的超参	比如 momentum（SGD）、alpha、centered 等等。

        """

        
        # 2. 混合精度前向传播 (AMP: Automatic Mixed Precision)
        # LLM 参数太多，用 float32 显存会爆炸，所以用 bfloat16 或 float16 计算，节省一半显存并加速
        """
        autocast_ctx 就是一个上下文管理器（context manager），用于在前向计算部分自动控制数值精度：

        在 GPU 上，它会启用 PyTorch 的自动混合精度 (AMP)：

        通过 torch.cuda.amp.autocast(dtype=…) 包裹的代码块内部，张量运算会尽可能使用 float16/bfloat16 进行计算。
        对于那些数值敏感（或不支持低精度）的操作，AMP 会自动回退到 float32。
        整体效果是 显存占用减半、计算加速，同时保持模型精度。
        在 CPU 上（或者当 dtype 不需要混合精度时），autocast_ctx 是个 nullcontext()，相当于什么都不做，保证代码仍然可运行。


        
        """
        with autocast_ctx:
            # 前向传播：模型输出。
            res = model(input_ids, labels=labels)
            # LLM 的主损失是交叉熵（Next-token prediction）；如果用了 MoE（混合专家模型），还会有负载均衡辅助损失 (aux_loss)
            """
            交叉熵损失 (Cross-Entropy)： 和你在 CNN 里做图像分类完全一样。CNN 是 10 分类或 1000 分类，而 LLM 是把词表大小（比如 50000 个词）看作 50000 个类别。模型输出一个概率分布，交叉熵衡量模型预测的那个词与正确答案词汇的差异。

            辅助损失 (aux_loss)： 这专属于 MoE（混合专家）架构。MoE 模型内部可能有 8 个“专家网络”，每次只挑 2 个工作。为了防止模型“偷懒”（只让最强的那 1 个专家干活，其他 7 个闲置），我们引入 aux_loss。它的作用是强制让所有专家雨露均沾地接收任务。   
            """
            loss = res.loss + res.aux_loss
            # 梯度累积：如果显存只够跑 Batch Size = 4，但你想达到 Batch Size = 32 的效果，
            # 可以算 8 次前向传播（损失除以 8），累积梯度后再更新一次权重。
            """
            这是梯度累积 (Gradient Accumulation) 的核心数学操作。
            假设你的显存只够塞下 batch_size=4，但你想达到 batch_size=32 的平滑梯度效果。你可以跑 8 次 batch_size=4 的前向和反向传播，把梯度累加起来，最后更新一次权重。
            因为损失函数通常是对 batch 内的样本求平均值。把 8 个 mini-batch 拼成一个大 batch，大 batch 的平均 loss 在数学上等于这 8 个 mini-batch loss 的平均。所以我们要把每次算出来的 loss 除以 8，这样累加出来的梯度大小，才严格等价于你直接跑一次 batch_size=32 的梯度。
            """
            loss = loss / args.accumulation_steps
        # 3. 反向传播
        # 由于使用了混合精度（float16 容易下溢出导致梯度变为 0），所以用 scaler 放大 loss 再反传
        """
        下溢出： float16 能表示的最小正数大概是 $6.1 \times 10^{-5}$。在 LLM 训练后期，梯度往往非常小（比如 $10^{-6}$）。如果直接用 float16 存，它会被直接截断成 $0.0$。梯度全变 0，模型就没法学习了。Scaler (自动混合精度缩放器) 的作用： scaler.scale(loss) 会在反向传播前，把 loss 乘上一个很大的数（比如 $65536$ 或 $2^{16}$）。因为 loss 变大了，求导算出来的梯度也跟着等比例变大，比如原本是 $10^{-6}$ 的梯度变成了 $0.065$，安全地落在了 float16 的表示范围内。等反向传播算完后，代码会在scaler.unscale_(optimizer)把它除回去，还原真实的梯度大小。
        """
        scaler.scale(loss).backward()
        # 4. 只有达到了累积步数，才真正更新一次模型参数
        if (step + 1) % args.accumulation_steps == 0:# 累积步数到了，更新一次参数
            scaler.unscale_(optimizer)# 更新前先把梯度缩放回来
            # 梯度裁剪：LLM 训练很容易出现梯度爆炸（某个 step 梯度突然无穷大导致模型崩盘），这是关键的保护措施
            """
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) 这个函数的第一个参数传入 model.parameters()，是因为在 PyTorch 中，每个参数张量（tensor）都有一个附属属性叫 .grad（存着它的梯度）。这个函数会去遍历 model.parameters() 中所有的权重，读取它们的 .grad，把所有梯度拼在一起算一个总范数（L2 Norm）。如果这个总长度超过了 args.grad_clip（比如 1.0），它就把所有梯度等比例缩小。这能防止某个极其异常的 batch 产生巨大梯度炸毁模型。
            """
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 优化器步进更新参数
            """
            它更新的是整个神经网络里所有可训练的权重矩阵。
            在 LLM 中，包括：词嵌入矩阵（Embedding）、Transformer 每一层里 Q, K, V 的线性映射权重、多层感知机（MLP）的权重、LayerNorm 的权重等。优化器会根据刚刚算出来的梯度，按照学习率把这些权重调整一点点。
            """
            scaler.step(optimizer)
            """
            scaler.step(optimizer) 的内部逻辑如下：

            检查： 它先扫描所有的梯度（刚才 unscale_ 还原回来的值）。

            判断： 检查梯度中是否存在 Inf 或 NaN。

            决策： * 如果梯度正常：它才会在内部默默调用 optimizer.step()，真正更新权重。

            如果梯度损坏：它会跳过这一次更新（Skip update），防止模型跑飞。
            
            """
            scaler.update()# 更新 scaler 的缩放因子

            """
            
            scaler.update() 又是干嘛的？scaler 本质上是一个动态缩放器。它会根据训练情况自动调整那个缩放因子 $S$：如果最近几次更新都很顺利：它会尝试把 $S$ 调大一点（比如翻倍），以便更好地捕捉微小的梯度，防止下溢出。如果刚才发生了梯度溢出（即 step 被跳过了）：它会立刻把 $S$ 调小（比如减半），尝试在下一个 Batch 躲开溢出区。
            """
            # 清空梯度，set_to_none=True 比传统 zero_grad() 更省一点点显存
            """
            在 PyTorch 的设计中，如果你调用 .backward()，新算出来的梯度会**累加（+=）**到现有的 .grad 里面，而不是覆盖。这对于 RNN 或刚刚提到的“梯度累积”非常方便。
            但是，如果你已经用这些梯度更新完权重了（即开启了下一个全新的 batch），你必须把旧梯度清空，否则下一次算出来的梯度会加上上一次的废梯度。
            为什么用 set_to_none=True？ 传统做法是把梯度矩阵全填为 0，这仍然占用着显存带宽。set_to_none=True 会直接把 .grad 的内存指针删掉释放掉，等下次反向传播再重新分配，这样稍微快一点点且省一点显存。
            
            """
            optimizer.zero_grad(set_to_none=True)
        # ================= 下面都是日志打印和模型保存逻辑 =================
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 还原真实的 loss（乘回累积步数），仅用于日志显示
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            
            """
    
            在 PyTorch 中，optimizer 维护着一组参数（params），而每组参数都会对应一个参数组（param_group）。
            optimizer.param_groups 就是一个列表，里面的每个元素都是一个字典，描述了一组参数以及它们对应的超参数设置。

            常见字段
            字段	说明
            params	这一组要优化的参数列表（Iterable[Tensor]）。
            lr	当前学习率（可在训练中被调度器或手动修改）。
            weight_decay	L2 正则化系数（如果用了的话）。
            betas	Adam/AdamW 等优化器的动量系数 (β₁, β₂)。
            eps	防止除零的微小常数。
            amsgrad	AdamW 中是否使用 AMSGrad 变体。
            其他特定优化器的超参	比如 momentum（SGD）、alpha、centered 等等。

            """

            """
            
            简单来说，param_groups 是在训练开始前（初始化优化器时）手动定义的，而不是在训练过程中动态产生的。以下是详细的拆解：
            1. 为什么叫“组”？（划分依据）在深度学习中，我们通常不会对模型里的所有参数一视同仁。最常见的划分依据是 “是否需要权重衰减 (Weight Decay)”：第一组 (Weight Decay)： 所有的矩阵权重（如 Transformer 中的 Linear 层、Embedding 层）。我们希望它们受到 L2 正则化约束，防止过拟合。第二组 (No Weight Decay)： 所有的偏置项（Bias）和归一化层（LayerNorm）的缩放参数。经验表明，对这些参数进行衰减反而会损害性能。代码示例（如何手动划分）：Python# 这是一个经典的 LLM 优化器初始化逻辑
            def get_optimizer(model, lr):
                # 过滤出需要 decay 和不需要 decay 的参数
                decay_params = [p for n, p in model.named_parameters() if p.requires_grad and ("bias" not in n and "LayerNorm" not in n)]
                no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and ("bias" in n or "LayerNorm" in n)]
                
                # 构建 param_groups 列表
                optim_groups = [
                    {"params": decay_params, "weight_decay": 0.01}, # 第一组
                    {"params": no_decay_params, "weight_decay": 0.0} # 第二组
                ]
                return torch.optim.AdamW(optim_groups, lr=lr)
            2. “我可以拿到所有组的吗？”是的，绝对可以。optimizer.param_groups 就是一个普通的 Python 列表。
            如果你想看第一组的学习率：optimizer.param_groups[0]['lr']
            如果你想看共有多少组：len(optimizer.param_groups)
            为什么你代码里写的是 [-1]？因为在 train_epoch 函数的前几行，代码做了一个循环：Pythonfor param_group in optimizer.param_groups:
                param_group['lr'] = lr
            这行代码已经把所有组的学习率都同步修改成了最新的 lr（由余弦退火算法算出）。既然所有组的学习率都一样，那么取 [0] 还是 [-1]（最后一组）拿到的数值都是相同的，开发者习惯用 [-1] 来代表“当前生效的配置”。3. 它和 Step（步数）的关系Step 不决定分组，但 Step 决定了组内 lr 的值。分组（Static）： 在 main 函数里定义优化器时就定死了。整个训练过程中，参数属于哪一组通常不会变。数值（Dynamic）： 在每个 step 结束时，你会根据当前是第几个 step，计算出一个新的 lr，然后去覆盖更新 param_groups 字典里的那个 'lr' 值。总结概念解释param_groups 列表像一个“部门清单”，每个部门（组）管理不同的模型参数。划分依据通常是参数类型（Linear vs LayerNorm），有时也用于给不同层设置不同的学习率（例如：微调时让 Header 层学得比 Backbone 快）。Step 的角色Step 是自变量，用来计算 lr。算出 lr 后，你再去挨个修改这些“部门”里的配置。
            如果你在 MiniMind 的代码里只看到 optimizer = optim.AdamW(model.parameters(), lr=...)，而没有像我上面那样写复杂的列表，那么默认情况下 len(param_groups) 只有 1。此时 [-1]、[0] 指向的都是同一个唯一的参数组。


            """

            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60#预估剩余时间，单位是分钟
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: 
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
            
            """
            
            这行代码的作用是将当前步（Step）的训练状态数据实时发送并记录到可视化面板中。

            在 LLM 训练中，由于训练周期长（可能几天甚至几周），仅仅看终端里滚动的数据非常不直观。wandb（Weights & Biases）或代码里实际使用的 swanlab 就像是给你的训练过程装了一个“实时监控摄像头”。

            1. 拆解字典中的各个字段
            这些字段代表了训练中最关键的几个指标：

            字段名	物理意义	监控它的目的
            loss	总损失	核心指标。看模型整体是否在收敛，曲线应该平滑下降。
            logits_loss	预测损失	衡量模型“文字接龙”预测得准不准。如果它不降，说明模型没学到语言规律。
            aux_loss	辅助损失	MoE 模型专用。监控 8 个专家是否被平均使用了，防止“一家干活，七家闲逛”。
            learning_rate	当前学习率	检查余弦退火策略是否正常。你应该能看到一个先上升（Warmup）后下降的弯曲弧线。
            epoch_time	预计剩余时间	监控训练速度。如果这个值突然变大，说明显卡可能过热降频或服务器负载变高。
            2. 为什么要用 if wandb: 包裹？
            这是一种防御性编程风格：

            如果你在启动脚本时没有添加 --use_wandb 参数，那么 wandb 对象就是 None。

            如果直接调用 wandb.log()，程序会因为尝试操作空对象而崩溃（AttributeError）。

            加上这个判断，确保了代码即使在离线状态（没有网络或不想记录日志）下也能正常运行。

            3. 在网页后端发生了什么？
            当你执行 wandb.log({...}) 时：

            序列化：Python 把这些数字打包成一个小的 JSON 数据包。

            异步传输：wandb 会在后台开启一个单独的进程把数据发往云端服务器，不会卡住你的 GPU 训练。

            实时绘图：你在浏览器里打开网页，就能看到这些数字自动变成了动态更新的折线图。
            """

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            """
            这一坨代码主要是为了“扒掉模型的外衣”，安全地存到硬盘上：

            model.eval(): 关闭 Dropout 等训练专用的随机操作。

            raw_model = model.module if isinstance(model, DistributedDataParallel) else model: 如果你是多卡训练，模型会被包在一个叫 DistributedDataParallel 的壳子里，权重其实在 model.module 里面。这里是把它扒出来。

            getattr(raw_model, '_orig_mod', raw_model): 如果你前面用了 torch.compile（PyTorch 2.0 的编译加速），它又会包一层。这里是把这层壳也扒掉，拿到最原始的、原汁原味的 Transformer 模型。

            .half().cpu(): 把精度从 float32 或 bfloat16 强行转成 float16 (half)，并且从显存移动到系统内存（CPU）。这样保存的文件体积少一半，且不占宝贵的显卡内存。
            """
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 获取真实的底层模型：因为模型可能被 DDP（分布式）或者 torch.compile（编译加速）包裹了一层
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)# 如果用了 torch.compile，还要再扒一层,拿到最原始的模型_orig_mod
            state_dict = raw_model.state_dict()
            # 以半精度 (half) 将权重保存到 CPU 上，节省硬盘空间和显存
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存额外的训练状态（如优化器状态、当前 step），以便断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict # 及时释放内存

        del input_ids, labels, res, loss # 显式释放变量，帮助 Python 垃圾回收，防显存泄漏


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()
    # 值得注意的参数：
    # --accumulation_steps: 梯度累积，变相增加 Batch Size。
    # --dtype bfloat16: bfloat16 相比 float16 动态范围更大，不容易溢出，是现代 LLM 训练的标配（需要 Ampere 架构及以上显卡，如 RTX 30/40 系）。
    # --max_seq_len: 模型能一次性“看”多长的句子。类似 CNN 的输入图片分辨率，越大越吃显存，且呈平方级增长（由于 Transformer 的 Attention 机制）。

    # ========== 1. 初始化环境和随机种子 ==========
    # DDP 分布式训练初始化：多张显卡协同训练时，每张卡是一个单独的进程。local_rank 就是当前进程所在显卡的 ID。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # MiniMindConfig：类似 CNN 里的 ResNet Config，里面定义了隐藏层维度、层数、注意力头数等架构信息。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 检查是否有之前保存的检查点，用于断点续训（防崩溃神技）
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    """
    bfloat16 (Brain Float 16) 是 Google 专为 AI 发明的格式。

    float16 精度高，但表示范围小（最大只能到 65504），很容易上下溢出。

    bfloat16 砍掉了精度（小数位数变少了），但保留了和 float32 一模一样的指数范围！这意味着它极难发生溢出。
    目前 LLM 训练几乎强制要求 bfloat16（因为 LLM 激活值的数值方差非常大）。如果你的显卡太老（比如 RTX 20 系或 T4），不支持 bfloat16 硬件加速，代码就只能 fallback 降级去用 float16（同时配合前面说的 scaler 保命）。
    """
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # Wandb (或这里的 SwanLab) 是一个非常好用的可视化看板，类似 TensorBoard，但适合看 LLM 训练的各种曲线。
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 初始化模型和分词器（Tokenizer：负责把文字切分成 token id）
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        # PyTorch 2.x 的黑科技，自动融合算子，能白嫖 10%-20% 的训练速度提升
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 定义数据集和采样器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DistributedSampler 确保在多卡训练时，每张卡分到的数据是不重复的
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # scaler 配合混合精度前向传播使用
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # LLM 预训练最常用的优化器：AdamW，通常比 SGD 收敛好得多
    """
    SGD（随机梯度下降）太死板了，它给模型里所有的参数都用一模一样的学习率。
    LLM 的损失空间就像极其复杂的高山流水。有些参数（比如词表里罕见词的权重）很久才更新一次，需要迈大步；有些参数（底层通用特征）每次都在更，需要迈小步。

    Adam 会跟踪每个参数梯度的历史均值（一阶矩，类似惯性动量）和历史方差（二阶矩，看震荡幅度），自动为模型里数亿个参数每个人分配定制化的学习率。

    W (Weight Decay) 代表它使用了修复版的 L2 正则化。
    所以 AdamW 面对 LLM 这种深不见底的网络时，不仅收敛快，还不易陷入死胡同。
    """
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    # 如果是续训，把模型权重、优化器历史动量、缩放器全塞回去
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        """
        DDP (Distributed Data Parallel)： 多卡训练时，PyTorch 会在每个显卡上复制一个一模一样的模型。每张卡吃不同的数据，算不同的梯度，然后在反向传播结束时，多张卡会在底层进行网络通信，把大家的梯度加起来取平均，确保所有卡更新后的权重永远保持一致。

        位置编码矩阵： Transformer 依靠 freqs_cos 和 freqs_sin 这两个三角函数矩阵来感知“词的前后顺序”（就像钟表的指针）。但这两个矩阵是写死的数学公式，不是可训练的权重（就像 CNN 里某些固定的高斯模糊核）。

        为什么要 ignore？ 如果不告诉 DDP，DDP 就会傻傻地在每次反向传播时，通过显卡间的网线去同步这些根本没有变过的固定的矩阵。加上这一句，可以节约大量没必要的 GPU 通信带宽，提升训练速度。
        """
        # 这里非常关键！freqs_cos 和 freqs_sin 是“旋转位置编码”（RoPE）的缓存矩阵。
        # 它们是固定的不可训练参数（类似于 CNN 里固定的滤波器），不需要被 DDP 在卡与卡之间同步梯度。
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 用 DDP 包装模型，使得每次 forward/backward 时自动同步多张卡的梯度
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)# 让每个 epoch 数据打乱顺序不同
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()

        # 针对断点续训的优化：如果上次训练在 step 500 断了，这次可以直接跳过前 500 个 batch，
        # 而不是从头遍历这个 epoch，极其节省时间。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()