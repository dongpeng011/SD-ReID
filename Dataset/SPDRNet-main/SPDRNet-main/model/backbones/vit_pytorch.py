import math
from functools import partial
from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc


# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------
def _ntuple(n):
    def parse(x):
        if isinstance(x, abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# --------------------------------------------------------
# Standard ViT Components (Mlp, Attention, Block, PatchEmbed)
# --------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        self.num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


# --------------------------------------------------------
# 1. SPCP: Semantic Part Expert Construction & Purification
# --------------------------------------------------------
class SPCP_Module(nn.Module):
    """ Section 3.2: 语义部位专家提纯模块 """

    def __init__(self, num_y, num_x, K=6, dim=768, delta=0.01):
        super().__init__()
        self.K = K
        self.dim = dim
        self.delta = delta

        # 预计算人体拓扑的高斯分布掩码 M_{i,k}
        # y 坐标范围 [0, num_y-1]
        y_coords = torch.arange(num_y).view(num_y, 1).repeat(1, num_x).view(-1).float()  # [N]
        centers = torch.linspace(0, num_y - 1, steps=K)  # [K]

        # 计算高斯响应 \phi(y_i, k)
        sigma = num_y / K  # 容差范围
        dist = (y_coords.unsqueeze(1) - centers.unsqueeze(0)) ** 2  # [N, K]
        phi = torch.exp(-dist / (2 * sigma ** 2))  # [N, K]

        # 归一化 Soft-assignment
        M_ik = phi / phi.sum(dim=1, keepdim=True)  # [N, K]
        # 过滤背景噪声 (M_ik > \delta)
        M_ik = torch.where(M_ik > delta, M_ik, torch.zeros_like(M_ik))

        self.register_buffer('M_ik', M_ik)  # [N, K] 存入缓冲区

        # 为每个局部专家分配独立的 Self-Attention (Q, K, V 不共享)
        self.expert_attentions = nn.ModuleList([
            Attention(dim=dim, num_heads=8, qkv_bias=True) for _ in range(K)
        ])

    def forward(self, x):
        # x shape: [B, N, D]
        B, N, D = x.shape
        expert_features = []

        for k in range(self.K):
            # 1. Soft aggregation (潜在语义聚合)
            mask_k = self.M_ik[:, k].unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
            E_k_raw = x * mask_k  # [B, N, D]

            # 2. Expert Self-Attention Purification (注意力提纯去噪)
            E_k_attn = self.expert_attentions[k](E_k_raw) + E_k_raw  # [B, N, D]

            # 3. Spatial Average Pooling (沿序列维度压缩)
            E_k = E_k_attn.mean(dim=1, keepdim=True)  # [B, 1, D]
            expert_features.append(E_k)

        # 级联 K 个专家的特征 -> [B, K, D]
        return torch.cat(expert_features, dim=1)


# --------------------------------------------------------
# 2. PGDR: Prompt-Guided Dynamic Routing
# --------------------------------------------------------
class PGDR_Module(nn.Module):
    """ Section 3.3: 场景提示引导的动态门控路由 """

    def __init__(self, dim=768, d_g=128, K=6, tau=1.0):
        super().__init__()
        self.tau = tau

        # 可学习的场景提示先验 S
        self.S = nn.Parameter(torch.zeros(1, d_g))
        nn.init.normal_(self.S, std=0.02)

        # 联合潜空间对齐降维矩阵 W_s
        self.Ws = nn.Linear(dim, d_g, bias=False)

        # 决策映射矩阵 W_g 和偏置 b_g
        self.Wg = nn.Linear(d_g, K, bias=True)

    def forward(self, E_patch, local_experts):
        """
        E_patch: 原始 Patch 序列 [B, N, D]
        local_experts: 局部专家特征 [B, K, D]
        """
        # 1. 提取全局视觉上下文 v_ctx -> [B, D]
        v_ctx = E_patch.mean(dim=1)

        # 2. 潜空间联合对齐 U_joint -> [B, d_g]
        U_joint = self.Ws(v_ctx) + self.S

        # 3. 稀疏截断激活 (ReLU) -> [B, d_g]
        h_gating = F.relu(U_joint)

        # 4. 对数几率映射 L_route -> [B, K]
        L_route = self.Wg(h_gating)

        # 5. 温度缩放 Softmax 权重分配 -> [B, K]
        omega = F.softmax(L_route / self.tau, dim=-1)

        # 6. 特征动态限流调制 (逐元素广播) E_route -> [B, K, D]
        E_route = local_experts * omega.unsqueeze(-1)

        return E_route, omega


# --------------------------------------------------------
# 3. GL-Fusion: Global-Local Collaborative Fusion
# --------------------------------------------------------
class GL_Fusion_Module(nn.Module):
    """ Section 3.4: 全局-局部特征协同融合 """

    def __init__(self, K=6, dim=768):
        super().__init__()
        # 将展平的 [B, K*D] 降维映射回 C 维 (此处 C=dim=768 以支持残差相加)
        self.Wf = nn.Linear(K * dim, dim, bias=True)

    def forward(self, E_global, E_route):
        """
        E_global: 未被切割的深层全局序列 [B, N, D]
        E_route: 动态路由调制后的局部序列 [B, K, D]
        """
        # 1. 提取全局身份锚点 F_global -> [B, D]
        F_global = E_global.mean(dim=1)

        # 2. 局部序列展平与对齐 F_local -> [B, D]
        B, K, D = E_route.shape
        F_local = self.Wf(E_route.view(B, K * D))

        # 3. 残差协同相加 -> [B, D]
        F_final = F_global + F_local
        return F_final


# --------------------------------------------------------
# Main Model: SPDR-Net Framework
# --------------------------------------------------------
class SPDR_ViT(nn.Module):
    def __init__(self, img_size=(384, 128), patch_size=16, stride_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 K=6, d_g=128, tau=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride_size=stride_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 2. Transformer Blocks (ViT Backbone)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # -----------------------------------------------------
        # SPDR-Net Core Modules (Replacing old ISGA modules)
        # -----------------------------------------------------
        # 模块1: 语义部位专家提纯
        self.spcp = SPCP_Module(num_y=self.patch_embed.num_y, num_x=self.patch_embed.num_x,
                                K=K, dim=embed_dim)

        # 模块2: 场景提示门控路由
        self.pgdr = PGDR_Module(dim=embed_dim, d_g=d_g, K=K, tau=tau)

        # 模块3: 全局-局部融合
        self.gl_fusion = GL_Fusion_Module(K=K, dim=embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Weight Init
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]

        # Patch Embed & ViT Forward
        E_patch = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        Z_input = torch.cat((cls_tokens, E_patch), dim=1)
        Z_input = Z_input + self.pos_embed
        Z_input = self.pos_drop(Z_input)

        for blk in self.blocks:
            Z_input = blk(Z_input)
        Z_input = self.norm(Z_input)

        # 剥离 CLS Token，提取所有 Patch Token -> [B, N, D]
        E_global_patches = Z_input[:, 1:, :]

        # [步骤1] SPCP: 提纯生成候选专家 -> [B, K, D]
        local_experts = self.spcp(E_global_patches)

        # [步骤2] PGDR: 计算动态权重与限流调制 -> E_route: [B, K, D], omega: [B, K]
        E_route, omega = self.pgdr(E_global_patches, local_experts)

        # [步骤3] GL-Fusion: 协同残差融合 -> [B, D]
        F_final = self.gl_fusion(E_global_patches, E_route)

        return F_final, omega

    def forward(self, x):
        # 返回 F_final 用于分类/度量学习，返回 omega 用于计算稀疏路由熵损失
        F_final, omega = self.forward_features(x)
        logits = self.fc(F_final)

        if self.training:
            return logits, F_final, omega
        else:
            return F_final  # 测试阶段仅返回特征向量


# --------------------------------------------------------
# Sparsity Routing Entropy Loss
# --------------------------------------------------------
class SparsityRoutingEntropyLoss(nn.Module):
    """ Section 3.5: 迫使门控网络做出极化且确定的路由决策 """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, omega):
        # omega shape: [B, K]
        # 公式: L_{ent} = (1/B) * \sum_{i} \sum_{k} - \omega_{i,k} \log(\omega_{i,k} + \epsilon)
        entropy = - torch.sum(omega * torch.log(omega + self.eps), dim=1)
        return entropy.mean()


# --------------------------------------------------------
# Model Builder
# --------------------------------------------------------
def vit_base_patch16_384_SPDR(img_size=(384, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0,
                              drop_path_rate=0.1, num_classes=1000, K=6, **kwargs):
    model = SPDR_ViT(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12,
                     mlp_ratio=4, qkv_bias=True, drop_path_rate=drop_path_rate, drop_rate=drop_rate,
                     attn_drop_rate=attn_drop_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                     num_classes=num_classes, K=K, **kwargs)
    return model


# --------------------------------------------------------
# test
# --------------------------------------------------------
if __name__ == '__main__':
    print("Initialize SPDR-Net...")
    model = vit_base_patch16_384_SPDR(num_classes=751, K=6)  # K=6 是你论文敏感性分析得出的最优解

    # 模拟输入张量 [Batch_Size, Channels, Height, Width]
    # 论文设定高度为 384, 宽度为 128
    dummy_input = torch.randn(4, 3, 384, 128)

    print("Forward Pass (Training Mode)...")
    model.train()
    logits, f_final, omega = model(dummy_input)

    print(f"Logits shape (用于 Cross Entropy): {logits.shape}")  # [4, 751]
    print(f"F_final shape (用于 Triplet Loss): {f_final.shape}")  # [4, 768]
    print(f"Omega shape (用于 Entropy Loss): {omega.shape}")  # [4, 6]

    # 验证路由权重之和是否为 1 (Softmax 特性)
    print(f"Sum of routing weights per sample: {omega.sum(dim=1).detach().numpy()}")

    # 测试自定义的稀疏路由熵损失
    entropy_criterion = SparsityRoutingEntropyLoss()
    loss_ent = entropy_criterion(omega)
    print(f"Sparsity Routing Entropy Loss: {loss_ent.item():.4f}")

    print("\nForward Pass (Inference Mode)...")
    model.eval()
    with torch.no_grad():
        test_feature = model(dummy_input)
        print(f"Test Feature shape: {test_feature.shape}")  # [4, 768]