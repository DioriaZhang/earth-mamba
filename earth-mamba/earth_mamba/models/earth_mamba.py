import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_

# 引用本地组件
from ..utils.layer_utils import LayerNorm, Permute, Linear, PatchMerge
from .earth_mamba_block import EarthMambaBlock
from ..modules.spatial_sparse import SparseSS2D


def _resolve_ssm_class(ssm_version: str, ssm_cls_override=None):
    """根据 ssm_version 字符串返回 SSM 模块类。优先使用显式传入的 ssm_cls_override。"""
    if ssm_cls_override is not None:
        return ssm_cls_override
    if ssm_version == "mamba3":
        from ..modules.ss2d_mamba3 import SS2D_Mamba3
        return SS2D_Mamba3
    return SparseSS2D  # 默认 Mamba-1 稀疏 SSM

class EarthMamba(nn.Module):
    """Earth-Mamba 分类骨干：patch 嵌入 + 多阶段 EarthMambaBlock 堆叠。

    ssm_version 控制 SSM 核心实现：
      "mamba1"（默认）：SparseSS2D（Mamba-1 ZOH + Sparsemax 稀疏门控）
      "mamba3"         ：SS2D_Mamba3（Mamba-3 指数梯形离散化 + RoPE + Triton SISO）
    """
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        # =========================
        # [Mamba-3 专用参数]
        ssm_version="mamba1",  # "mamba1" | "mamba3"
        ssm_headdim=64,        # Mamba-3: 每头维度，需满足 d_inner % headdim == 0
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="LN",
        downsample_version="v2",
        patchembed_version="v1",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
        # =========================
        # [创新模块配置 - 全局控制]
        _SS2D=None,           # None：根据 ssm_version 自动选择；显式传入可覆盖
        use_armg=True,        # 全局开启/关闭 ARMG
        use_graph=True,       # 全局开启/关闭 Graph
        use_sparse_ssm=True,  # 全局开启/关闭 Path A（SSM 支路残差）
        graph_num_nodes=64,   # Graph 节点数
        ssm_backend=None,     # None：走环境变量 EARTH_MAMBA_SSM_BACKEND 或默认 CUDA
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        # 根据 ssm_version 或显式传入的 _SS2D 确定 SSM 模块类
        _SS2D_resolved = _resolve_ssm_class(ssm_version, _SS2D)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None
        self.patch_embed = self._make_patch_embed(in_chans, dims[0], patch_size, patch_norm, channel_first=self.channel_first, version=patchembed_version)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = self._make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                channel_first=self.channel_first,
                version=downsample_version,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                downsample=downsample,
                channel_first=self.channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                # [Mamba-3] headdim 透传至 SSM 构造，SparseSS2D 会通过 **kwargs 忽略
                ssm_headdim=ssm_headdim,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # [核心] 传递模块选择与消融参数
                _SS2D=_SS2D_resolved,
                use_armg=use_armg,
                use_graph=use_graph,
                use_sparse_ssm=use_sparse_ssm,
                graph_num_nodes=graph_num_nodes,
                ssm_backend=ssm_backend,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=LayerNorm(self.num_features, channel_first=self.channel_first),
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, channel_first=False, version="v1"):
        if version == "v1":
            return nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
                nn.Identity(),
                (LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first) 
                    if patch_norm else (nn.Identity() if channel_first else Permute(0, 2, 3, 1))),
            )
        elif version == "v2":
            stride = patch_size // 2
            kernel_size = stride + 1
            padding = 1
            return nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Identity(),
                (LayerNorm(embed_dim // 2, channel_first=True) if patch_norm else nn.Identity()),
                nn.Identity(),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Identity(),
                (LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first) 
                    if patch_norm else (nn.Identity() if channel_first else Permute(0, 2, 3, 1))),
            )
        raise NotImplementedError

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm=True, channel_first=False, version="v1"):
        if version == "v1": 
            return nn.Sequential(
                PatchMerge(channel_first),
                LayerNorm(4 * dim, channel_first=channel_first) if norm else nn.Identity(),
                Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False, channel_first=channel_first),
            )
        elif version == "v2":
            return nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
                nn.Identity(),
                LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first) if norm else 
                    (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif version == "v3":
            return nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.Identity(),
                LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first) if norm else 
                    (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        raise NotImplementedError

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        downsample=nn.Identity(),
        channel_first=False,
        _SS2D=SparseSS2D,
        # [透传]
        use_armg=True,
        use_graph=True,
        use_sparse_ssm=True,
        graph_num_nodes=64,
        ssm_backend=None,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(EarthMambaBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                channel_first=channel_first,
                use_checkpoint=use_checkpoint,
                ssm_cls=_SS2D,
                use_armg=use_armg,
                use_graph=use_graph,
                use_sparse_ssm=use_sparse_ssm,
                graph_num_nodes=graph_num_nodes,
                ssm_backend=ssm_backend,
                **kwargs,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x
    
    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            self.load_state_dict(_ckpt[key], strict=False)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")


class BackboneEarthMamba(EarthMamba):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["ln2d"])
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = LayerNorm(self.dims[i], channel_first=self.channel_first)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        return outs
