import torch
from torch import optim, nn

# ==========================================
# 1. 定义 LoRA 网络结构
# ==========================================
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），通常远小于输入输出维度，控制低秩矩阵的大小
        
        # 降维矩阵 A: 将输入维度压缩到 rank 维度
        self.A = nn.Linear(in_features, rank, bias=False)  
        # 升维矩阵 B: 将 rank 维度恢复到输出维度
        self.B = nn.Linear(rank, out_features, bias=False)  
        
        # 【关键初始化策略】
        # 矩阵 A 使用高斯分布初始化（正态分布），打破对称性
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵 B 全 0 初始化
        # 这样做是为了保证初始状态下 B(A(x)) == 0，
        # 从而确保在刚加上 LoRA 模块时，模型的输出与原模型完全一致，不会破坏预训练权重。
        self.B.weight.data.zero_()

    def forward(self, x):
        """
        LoRA 插入到 LLM 计算时的前向传播流程。
        
        参数说明:
            x: 输入特征 (维度: [batch_size, seq_len, in_features])
            x_base_output: 原模型矩阵 W_0 计算出的输出结果 (维度: [batch_size, seq_len, out_features])
                           注意：此时原模型 W_0 的参数在训练中是被 【冻结(Freeze)】 的
                           
        计算流程图示:
                     输入特征 (x)
                          │
            ┌─────────────┴─────────────┐
            │ (复制一份)                  │
            ▼                           ▼
       [原模型 W_0] (已冻结)       [LoRA 旁路]
            │                           │
            │                     1. 经 A 降维: self.A(x)  -> 得到 [batch_size, seq_len, rank]
            │                           │
            │                     2. 经 B 升维: self.B(...) -> 得到 [batch_size, seq_len, out_features]
            │                           │
     (x_base_output)             (lora_output)
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
                    3. 【 ⊕ 加法器 】 -> 两路结果尺寸一致，直接相加合并
                          │
                          ▼
                       最终输出
        """
        # 前向传播：先降维，再升维
        return self.B(self.A(x))


# ==========================================
# 2. 将 LoRA 注入到目标模型中
# ==========================================
def apply_lora(model, rank=8):
    # 遍历模型中的所有子模块
    for name, module in model.named_modules():
        # 这里设定了一个条件：只对 nn.Linear 层且权重为方阵（输入维度==输出维度）的层注入 LoRA。
        # 注意：实际应用中，LoRA 也可以应用于非方阵，这里是代码本身的特定限制。
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 实例化 LoRA 模块，并将其移至与当前层相同的设备 (CPU/GPU)
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(module.weight.device)
            
            # 将 LoRA 模块作为属性绑定到原 layer 上，方便后续调用和保存
            setattr(module, "lora", lora)
            
            # 保存原本的前向传播函数
            original_forward = module.forward

            # 显式绑定：重写前向传播
            # 【重要技巧】使用默认参数 (layer1=original_forward, layer2=lora) 
            # 是为了解决 Python 循环中闭包的“延迟绑定（Late Binding）”问题，
            # 确保每个 layer 绑定的都是自己对应的函数和 lora 实例。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 原路输出 + LoRA分支输出 (即: Wx + \Delta Wx)
                return layer1(x) + layer2(x)

            # 替换原模块的 forward 方法
            module.forward = forward_with_lora


# ==========================================
# 3. 加载 LoRA 权重
# ==========================================
def load_lora(model, path):
    # 读取保存的字典状态
    state_dict = torch.load(path, map_location=model.device if hasattr(model, 'device') else 'cpu')
    
    # 兼容处理：如果模型使用了 DataParallel (DDP)，权重键名会多出 'module.' 前缀，需要去掉
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    # 遍历当前模型模块，寻找挂载了 'lora' 属性的层
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 筛选出属于当前特定 lora 模块的权重，并去掉前缀使其与 LoRA 类的内部变量名匹配
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 加载权重
            module.lora.load_state_dict(lora_state)


# ==========================================
# 4. 提取并保存 LoRA 权重
# ==========================================
def save_lora(model, path):
    # 兼容处理：获取可能被 DDP 或 torch.compile 包装过的原始模型
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    
    # 遍历模块，只提取 LoRA 的权重（抛弃原模型庞大的基础权重，实现轻量化保存）
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            # 拼接正确的 key 格式，例如: "layer1.lora.A.weight"
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
            
    # 保存为一个极小的 .pt / .pth 文件
    torch.save(state_dict, path)