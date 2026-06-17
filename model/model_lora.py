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
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(module.weight.device)
            setattr(module, "lora", lora)
            setattr(module, "lora_list", [lora])

            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, lora_modules=module.lora_list):
                lora_out = sum(lm(x) for lm in lora_modules)
                return layer1(x) + lora_out

            module.forward = forward_with_lora


def apply_lora_multi(model, ranks=None):
    """为每层注入多个不同 rank 的 LoRA 模块，用于多 LoRA 合并推理"""
    if ranks is None:
        ranks = [8]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora_modules = nn.ModuleList()
            for r in ranks:
                lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=r).to(module.weight.device)
                lora_modules.append(lora)
            setattr(module, "lora_list", lora_modules)

            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, lms=lora_modules):
                lora_out = sum(lm(x) for lm in lms)
                return layer1(x) + lora_out

            module.forward = forward_with_lora


# ==========================================
# 3. 加载 LoRA 权重
# ==========================================
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device if hasattr(model, 'device') else 'cpu')
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora_list') and len(module.lora_list) == 1:
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            if lora_state:
                module.lora_list[0].load_state_dict(lora_state)


def load_lora_multi(model, paths, merge_weights=None):
    """加载多个 LoRA 权重文件到同一个模型的 lora_list 中"""
    if merge_weights is None:
        merge_weights = [1.0] * len(paths)
    for idx, path in enumerate(paths):
        state_dict = torch.load(path, map_location=model.device if hasattr(model, 'device') else 'cpu')
        state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
        for name, module in model.named_modules():
            if hasattr(module, 'lora_list') and idx < len(module.lora_list):
                lora_key_prefix = f'{name}.lora_list.{idx}.'
                lora_state = {k.replace(lora_key_prefix, ''): v for k, v in state_dict.items() if lora_key_prefix in k}
                if lora_state:
                    module.lora_list[idx].load_state_dict(lora_state)
                    if merge_weights[idx] != 1.0:
                        for p in module.lora_list[idx].parameters():
                            p.data.mul_(merge_weights[idx])


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