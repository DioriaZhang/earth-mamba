import torch
import torch.nn as nn

class ARMG(nn.Module):
    """
    Noise-robust 表征门控（Adaptive Refinement Mamba Gate, ARMG）
    对应「抗干扰 / 噪声鲁棒」设计：软阈值抑制椒盐与强度相关光噪，无噪时趋近恒等映射。
    
    公式:
        mu = GlobalAvgPool(|x|)
        tau = w * mu + b
        x_clean = tau * tanh(x / tau)
        
    功能:
        1. 椒盐噪声抑制: 利用 tanh 的饱和特性截断大幅值的离群点。
        2. 光噪(Poisson)适应: 利用 mu 感知信号强度，自适应调整阈值 tau。
        3. 无损保留: 当 tau 很大时，tanh(x/tau) ≈ x/tau，输出趋近于 x (恒等映射)。
    """
    def __init__(self, dim, init_value=5.0, eps=1e-6):
        """
        Args:
            dim (int): 输入特征通道数 (C)
            init_value (float): b 的初始化值。设大一点可以保证初始状态接近 Identity。
            eps (float): 防止除零的极小值。
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # 可学习参数 w 和 b
        # 形状为 (dim,) 以便对每个通道独立学习阈值
        self.w = nn.Parameter(torch.zeros(dim)) # 初始权重设为0，让 tau 主要由 b 决定
        self.b = nn.Parameter(torch.ones(dim) * init_value) # 初始偏置设大

    def forward(self, x):
        """
        Args:
            x: 输入张量，支持 (B, H, W, C) 或 (B, L, C)
        Returns:
            x_clean: 去噪后的特征，形状同输入
        """
        # 1. 记录原始形状并标准化为 (B, L, C) 以便计算统计量
        input_shape = x.shape
        is_spatial = x.dim() == 4 # (B, H, W, C)
        
        if is_spatial:
            B, H, W, C = input_shape
            # Flatten spatial dims: (B, H, W, C) -> (B, H*W, C)
            x_flat = x.view(B, -1, C)
        else:
            B, L, C = input_shape
            x_flat = x

        # 2. 计算统计感知量 mu
        # mu = GlobalAvgPool(|x|) -> shape (B, C)
        # 对空间维度 L 求平均
        mu = x_flat.abs().mean(dim=1) 

        # 3. 计算自适应阈值 tau
        # tau = w * mu + b
        # self.w, self.b: (C,) -> broadcast to (B, C)
        tau = mu * self.w + self.b
        
        # 强制 tau 为正数且不为 0 (取绝对值 + eps)
        # 物理意义：噪声门限必须是一个正幅值
        tau = tau.abs() + self.eps

        # 4. 广播 tau 到输入维度
        # (B, C) -> (B, 1, 1, C) or (B, 1, C)
        if is_spatial:
            tau_broadcast = tau.view(B, 1, 1, C)
        else:
            tau_broadcast = tau.view(B, 1, C)

        # 5. 应用软阈值公式
        # x_clean = tau * tanh(x / tau)
        # 这里的 x 使用原始输入的 shape
        x_clean = tau_broadcast * torch.tanh(x / tau_broadcast)

        return x_clean

# 单元测试 (Unit Test)
if __name__ == "__main__":
    print("Testing ARMG Module...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 定义模块
    dim = 96
    model = ARMG(dim=dim).to(device)
    
    # 2. 创建假数据 (B, H, W, C)
    x = torch.randn(2, 64, 64, dim).to(device)
    
    # 3. 模拟椒盐噪声 (手动添加离群点)
    # 将某个点的值设为极大 (比如 100.0，正常分布通常在 -3~3 之间)
    x[0, 32, 32, 0] = 100.0 
    
    print(f"Input Max (with noise): {x.max().item():.4f}")
    
    # 4. 前向传播
    out = model(x)
    
    print(f"Output Shape: {out.shape}")
    print(f"Output Max (suppressed): {out.max().item():.4f}")
    
    # 验证是否被抑制
    if out.max().item() < 90.0:
        print("SUCCESS: 椒盐噪声已被抑制 (Soft Thresholding 生效)")
    else:
        print("FAIL: 噪声未被抑制，请检查参数初始化")
        
    # 5. 验证反向传播
    loss = out.mean()
    loss.backward()
    if model.w.grad is not None:
        print("SUCCESS: 梯度回传正常")
