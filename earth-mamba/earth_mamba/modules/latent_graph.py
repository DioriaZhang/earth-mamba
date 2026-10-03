import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LatentGraph(nn.Module):
    """
    Path B — 语义潜在图（与多向扫描互补的 scan-rotation / 全局结构分支）

    设计逻辑：
    A 路（稀疏 SSM + 四向 scan）侧重序列化局部细节；B 路将特征聚合到少量原型节点，
    在相似度图上推理后再投影回原空间。这是**语义级**全局建模，文档中已说明并非严格几何旋转不变。
    
    流程：
    1. Aggregation: 像素 -> 潜在节点 (M个)
    2. Graph Reasoning: 动态构图 + GCN 交互
    3. Re-projection: 潜在节点 -> 像素
    
    参考文献：
    - Graph-Based Global Reasoning Networks (GloRe)
    - MEMMAMBA (Wang et al., 2025)
    """
    def __init__(self, dim, num_nodes=64, scale_factor=None):
        """
        Args:
            dim (int): 输入特征维度 D
            num_nodes (int): 潜在节点数量 M (即 Prototypes 数量，默认 64)
            scale_factor (float): 缩放因子，默认为 1/sqrt(D)
        """
        super().__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.scale = scale_factor if scale_factor is not None else dim ** -0.5
        
        # ============================================================
        # Step 1: 语义聚合 (Aggregation) 
        # ============================================================
        # 定义原型矩阵 P 的投影变换: x * W_p
        # W_p 形状为 (dim, num_nodes)
        self.assign_proj = nn.Linear(dim, num_nodes, bias=False)
        
        # ============================================================
        # Step 2: 构图推理 (Graph Reasoning)
        # ============================================================
        # GCN 的权重矩阵 W_g: (D, D)
        self.gcn_proj = nn.Linear(dim, dim, bias=True)
        
        # 可学习的缩放因子 beta，控制邻接矩阵分布的尖锐程度
        # 初始化为 1.0 (或者 log(1/0.07) 等温度参数)
        self.beta = nn.Parameter(torch.ones(1))
        
        # 激活函数
        self.act = nn.ReLU()
        
        # ============================================================
        # Step 3: 几何重投影 (Re-projection)
        # ============================================================
        # 重投影复用 Step 1 计算出的分配矩阵 G，无需额外参数

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) 或 (B, L, C)
        Returns:
            Y_graph: (B, H, W, C) 或 (B, L, C)
        """
        # 0. 形状标准化 -> (B, N, D)
        input_shape = x.shape
        if x.dim() == 4:
            B, H, W, C = x.shape
            N = H * W
            x_flat = x.view(B, N, C)
        else:
            B, N, C = x.shape
            x_flat = x

        # ============================================================
        # Step 1: 语义聚合 (Aggregation)
        # ============================================================
        # 计算分配矩阵 G (Assignment Matrix)
        # logits: (B, N, M)
        assign_logits = self.assign_proj(x_flat) * self.scale
        
        # G = Softmax(x W_p / sqrt(D)) -> (B, N, M)
        # dim=2 表示在 M 个原型节点间归一化 (每个像素属于哪个节点的概率)
        # 或者是 dim=1 表示每个节点由哪些像素组成。
        # 根据 GloRe 和一般 Non-local 逻辑，通常是对 M 归一化。
        g_map = torch.softmax(assign_logits, dim=-1)
        
        # 聚合节点特征 V_nodes
        # V = G^T * x
        # (B, N, M)^T -> (B, M, N)
        # (B, M, N) @ (B, N, D) -> (B, M, D)
        v_nodes = torch.bmm(g_map.transpose(1, 2), x_flat)

        # ============================================================
        # Step 2: 动态构图推理 (Graph Reasoning)
        # ============================================================
        # 构建邻接矩阵 S (Cosine Similarity)
        # S_ij = (V_i * V_j^T) / (|V_i| * |V_j|)
        
        # L2 Normalize along feature dim (D)
        v_norm = F.normalize(v_nodes, p=2, dim=-1) # (B, M, D)
        
        # S = V_norm @ V_norm^T -> (B, M, M)
        adj_raw = torch.bmm(v_norm, v_norm.transpose(1, 2))
        
        # 生成最终邻接矩阵 G_adj
        # G_ij = Softmax(beta * S_ij)
        adj = torch.softmax(self.beta * adj_raw, dim=-1) # Row-normalize
        
        # GCN 传播
        # Formula: ReLU(G_adj * V_nodes * W_g)
        # 1. Linear Transform: V * W_g -> (B, M, D)
        v_transform = self.gcn_proj(v_nodes)
        
        # 2. Graph Convolution: Adj * V_transform
        # (B, M, M) @ (B, M, D) -> (B, M, D)
        v_gcn = torch.bmm(adj, v_transform)
        
        # 3. Activation & Residual
        # \hat{V} = V + ReLU(...)
        v_new = v_nodes + self.act(v_gcn)

        # ============================================================
        # Step 3: 几何重投影 (Re-projection)
        # ============================================================
        # Y_graph = G * \hat{V}
        # (B, N, M) @ (B, M, D) -> (B, N, D)
        y_graph = torch.bmm(g_map, v_new)
        
        # 恢复原始形状
        if len(input_shape) == 4:
            y_graph = y_graph.view(B, H, W, C)
            
        return y_graph

# 单元测试
if __name__ == "__main__":
    print("Testing LatentGraph Module...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dim = 96
    num_nodes = 32
    model = LatentGraph(dim=dim, num_nodes=num_nodes).to(device)
    
    # 输入 (B, H, W, C)
    x = torch.randn(2, 64, 64, dim).to(device)
    
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    
    if x.shape == out.shape:
        print("SUCCESS: 维度保持一致")
    else:
        print("FAIL: 维度不匹配")
        
    # 检查反向传播
    loss = out.mean()
    loss.backward()
    if model.beta.grad is not None:
        print(f"SUCCESS: 梯度回传正常 (beta grad: {model.beta.grad.item():.6f})")
