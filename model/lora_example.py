import os
import torch
from torch import nn
from model_lora import LoRA, apply_lora, save_lora, load_lora
# ==========================================
# 步骤 1: 准备目标模型和数据
# ==========================================
class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个方阵线性层 (会被注入 LoRA，因为 128 == 128)
        self.layer1 = nn.Linear(128, 128)
        # 定义一个非方阵线性层 (不会被注入，因为 128 != 64)
        self.layer2 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

# 实例化模型
model = SimpleNetwork()

# 创建模拟输入数据 (Batch Size = 2, Feature = 128)
dummy_input = torch.randn(2, 128)

# 记录注入前模型的输出
with torch.no_grad():
    output_before = model(dummy_input)

# ==========================================
# 步骤 2: 注入 LoRA
# ==========================================
print("注入 LoRA 前，layer1 是否有 lora 属性:", hasattr(model.layer1, "lora"))

apply_lora(model, rank=4)

print("注入 LoRA 后，layer1 是否有 lora 属性:", hasattr(model.layer1, "lora"))
print("注入 LoRA 后，layer2 是否有 lora 属性:", hasattr(model.layer2, "lora")) # 应该是 False

# ==========================================
# 步骤 3: 验证初始化特性 (输出应该与原模型完全一致)
# ==========================================
with torch.no_grad():
    output_after_init = model(dummy_input)

# 由于 LoRA 的 B 矩阵全0初始化，此时输出差异应为 0
difference = (output_before - output_after_init).abs().sum().item()
print(f"\n刚注入 LoRA 时，模型输出差异为: {difference:.6f}")

# ==========================================
# 步骤 4: 模拟训练与权重变化
# ==========================================
# 随机改变一下 LoRA 的权重，模拟经过了一段时间的微调训练
model.layer1.lora.B.weight.data.normal_(0, 0.1)

with torch.no_grad():
    output_after_training = model(dummy_input)

train_diff = (output_before - output_after_training).abs().sum().item()
print(f"模拟训练后（权重改变），模型输出差异为: {train_diff:.6f}")

# ==========================================
# 步骤 5: 测试保存与加载
# ==========================================
save_path = "lora_weights.pt"

# 只保存 LoRA 权重
save_lora(model, save_path)
print(f"\nLoRA 权重已保存至: {save_path}")
print(f"保存的文件大小为: {os.path.getsize(save_path)} 字节 (非常轻量！)")

# 重置模型，重新注入 LoRA（模拟在新环境加载）
# 注意：为了正确测试 LoRA 加载，需要使用相同的基础模型权重
torch.manual_seed(42)  # 设置相同的随机种子
new_model = SimpleNetwork()
new_model.load_state_dict(model.state_dict(), strict=False)  # 复制基础权重（忽略 lora 相关的额外属性）
apply_lora(new_model, rank=4)

# 此时新模型的 B 矩阵又是全 0
with torch.no_grad():
    output_new_model_init = new_model(dummy_input)

# 加载训练好的 LoRA 权重
load_lora(new_model, save_path)

with torch.no_grad():
    output_new_model_loaded = new_model(dummy_input)

# 验证加载后的输出是否与上面“模拟训练后”的输出完全一致
load_diff = (output_after_training - output_new_model_loaded).abs().sum().item()
print(f"加载 LoRA 权重后，与目标输出的差异为: {load_diff:.6f} (应为0)")

# 清理测试文件
if os.path.exists(save_path):
    os.remove(save_path)