"""下游评测公共约定：输出目录命名 + 终端结果横幅 + 训练加速工具。"""

from __future__ import annotations

import contextlib
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

# ── 默认路径（hy-tmp 单机）──────────────────────────────────────────────────
DEFAULT_DATA_ROOT = "/hy-tmp/task"
DEFAULT_OUTPUT_DIR = "/hy-tmp/downstream_result"
DEFAULT_CKPT = None  # 分任务 ckpt 由 run_all.sh 传入，不设全局默认

_RUN_TS_RE = re.compile(r"^\d{8}_\d{6}$")


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_run_timestamp(name: str) -> bool:
    return bool(_RUN_TS_RE.match(name))


def dataset_name_from(data_dir: str | None, fallback: str = "run") -> str:
    if data_dir:
        return Path(data_dir).resolve().name
    return fallback


def resolve_out_dir(
    output_dir: str | None,
    *,
    data_dir: str | None = None,
    dataset_name: str | None = None,
) -> Path:
    """
    解析最终写入目录（results.json / best.pth 直接落在此目录）。

    规则（与 run_all.sh 一致）::
      - 完整路径 ``.../20260612_223215/PatternNet`` → 直接使用
      - run 根 ``.../20260612_223215`` → 追加 ``{dataset}/``
      - 根 ``/hy-tmp/downstream_result`` → 自动 ``{根}/{新时间戳}/{dataset}/``
      - 实验文件夹 ``.../downstream_result/my_exp`` → ``my_exp/{dataset}/``
      - 其它任意路径 → 视为最终目录直接使用
    """
    ds = dataset_name or dataset_name_from(data_dir)
    base = Path(output_dir or DEFAULT_OUTPUT_DIR)
    default_root = Path(DEFAULT_OUTPUT_DIR).resolve()

    # .../YYYYMMDD_HHMMSS/PatternNet(_random|_knn) — shell 传入的叶子目录
    if base.parent != base and is_run_timestamp(base.parent.name):
        base.mkdir(parents=True, exist_ok=True)
        return base

    if base.name.lower() == ds.lower():
        base.mkdir(parents=True, exist_ok=True)
        return base

    # .../YYYYMMDD_HHMMSS — 一次 sh 跑批的时间戳根
    if is_run_timestamp(base.name):
        out = base / ds
        out.mkdir(parents=True, exist_ok=True)
        return out

    # /hy-tmp/downstream_result — 单独 python 跑时自动新建时间戳
    try:
        if base.resolve() == default_root:
            out = base / new_run_id() / ds
            out.mkdir(parents=True, exist_ok=True)
            return out
    except OSError:
        pass

    # /hy-tmp/downstream_result/tune 或自定义实验名
    try:
        if base.parent.resolve() == default_root:
            out = base / ds
            out.mkdir(parents=True, exist_ok=True)
            return out
    except OSError:
        pass

    base.mkdir(parents=True, exist_ok=True)
    return base


def make_out_dir(
    output_dir: str,
    task_type: str,
    dataset_name: str,
    *,
    prefix: Optional[str] = None,
) -> Path:
    """兼容旧调用；新代码请用 resolve_out_dir。"""
    _ = task_type, prefix
    return resolve_out_dir(output_dir, dataset_name=dataset_name)


def add_output_args(parser, *, default: str = DEFAULT_OUTPUT_DIR) -> None:
    parser.add_argument(
        "--output_dir",
        default=default,
        help=(
            f"结果目录。可直接传 run 叶子路径（如 .../20260612_120000/PatternNet）；"
            f"或传时间戳 run 根（脚本追加数据集子目录）；"
            f"默认 {default}/{{时间戳}}/{{数据集}}/"
        ),
    )


def ckpt_label(ckpt: Optional[str]) -> str:
    if not ckpt:
        return "random_init"
    return Path(ckpt).name


def print_final_summary(
    *,
    dataset: str,
    task: str,
    script: str,
    ckpt: Optional[str],
    metrics: Dict[str, Any],
    out_dir: Path,
    split: str = "",
    epoch: Optional[int] = None,
    extra_lines: Optional[list[str]] = None,
) -> None:
    """训练结束在终端打印可复制到记录.md 的结果块。"""
    width = 72
    print("\n" + "=" * width)
    print(f"  [{dataset}] {task}  |  {script}")
    print(f"  ckpt: {ckpt or '(none) random init'}")
    if split:
        print(f"  split: {split}")
    if epoch is not None:
        print(f"  best_epoch: {epoch}")
    print("-" * width)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    if extra_lines:
        for line in extra_lines:
            print(f"  {line}")
    print("-" * width)
    print(f"  输出目录: {out_dir}")
    print(f"  结果文件: {out_dir / 'results.json'}")
    print("=" * width + "\n")


# ── GPU 能力检测 ──────────────────────────────────────────────────────────────


def auto_amp_dtype() -> str:
    """自动检测最优混合精度类型。
    A100/H100 (sm_80+) → bf16；V100/T4 等 (sm_70/75) → fp16；无 GPU → none。
    """
    if not torch.cuda.is_available():
        return "none"
    major, _ = torch.cuda.get_device_capability()
    return "bf16" if major >= 8 else "fp16"


def gpu_info() -> str:
    """返回当前 GPU 名称及计算能力字符串（用于日志）。"""
    if not torch.cuda.is_available():
        return "CPU"
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return f"{name} (sm_{major}{minor})"


# ── 训练加速（A100/V100 bf16/fp16 自适应 / DataLoader）──────────────────────


def setup_perf() -> None:
    """固定输入尺寸时启用 cuDNN autotune；A100 额外开 TF32。"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def add_perf_args(parser) -> None:
    """为下游脚本添加通用性能参数。"""
    g = parser.add_argument_group("performance")
    g.add_argument(
        "--amp_dtype",
        default="auto",
        choices=["auto", "none", "bf16", "fp16"],
        help=(
            "混合精度：auto=自动选择（A100→bf16，V100→fp16）；"
            "A100 推荐 bf16（与预训练一致）；V100 用 fp16；none=FP32"
        ),
    )
    g.add_argument("--prefetch_factor", type=int, default=4,
                   help="DataLoader prefetch（num_workers>0 时生效）")
    g.add_argument("--persistent_workers", action="store_true", default=True,
                   help="DataLoader 持久 worker，减少 epoch 间重启开销")
    g.add_argument("--no_persistent_workers", action="store_false",
                   dest="persistent_workers",
                   help="关闭 persistent_workers（内存紧张时用）")
    # 实验性：EarthMamba 含 Mamba3 自定义 Triton kernel，torch.compile 通常会在首步报错
    g.add_argument("--compile", action="store_true",
                   help="（不推荐）torch.compile；EarthMamba/Mamba3 与 inductor 不兼容，会自动跳过")


def maybe_compile(
    model: nn.Module,
    do_compile: bool,
    device: Optional[torch.device] = None,
    mode: str = "reduce-overhead",
) -> nn.Module:
    """可选 torch.compile。EarthMamba + Mamba3 自定义 kernel 与 inductor 不兼容，默认跳过。

    A100 加速请用 ``--amp_dtype auto``（bf16），不要依赖 ``--compile``。
    若用户显式传 ``--compile``，此处打印警告并回退 eager，避免首步训练崩溃。
    """
    if not do_compile:
        return model
    print(
        "  [跳过] torch.compile：EarthMamba/Mamba3 自定义算子与 PyTorch inductor 不兼容\n"
        "         （典型报错 BackendCompilerFailed / cannot extract sympy expressions）\n"
        "         A100 请用 --amp_dtype auto（bf16）加速，无需 --compile"
    )
    return model


def loader_kwargs(
    num_workers: int,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> dict:
    """DataLoader 公共 kwargs（pin_memory 由调用方加）。"""
    kw: dict = {}
    if num_workers > 0:
        kw["prefetch_factor"] = prefetch_factor
        if persistent_workers:
            kw["persistent_workers"] = True
    return kw


class AmpHelper:
    """bf16/fp16 混合精度；bf16 在 A100 上无需 GradScaler。auto 模式自动选最优精度。"""

    def __init__(self, amp_dtype: str = "auto"):
        raw = (amp_dtype or "auto").lower()
        self.amp_dtype = auto_amp_dtype() if raw == "auto" else raw
        self.enabled = (
            self.amp_dtype in ("bf16", "fp16")
            and torch.cuda.is_available()
        )
        self.use_scaler = self.enabled and self.amp_dtype == "fp16"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        clip_grad: float = 0.0,
    ) -> None:
        if self.use_scaler:
            self.scaler.scale(loss).backward()
            if clip_grad > 0:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

    def attach(self, args) -> "AmpHelper":
        """挂到 args 上供 fine_tune_cls 等模块读取。"""
        args.amp_helper = self
        return self


def to_device(x, device, non_blocking: bool = True):
    if hasattr(x, "to"):
        return x.to(device, non_blocking=non_blocking)
    return x


def args_to_dict(args) -> dict:
    """将 argparse.Namespace 转为可 JSON 序列化的字典。
    排除 amp_helper 等不可序列化的对象。"""
    d = vars(args).copy()
    # 移除不可序列化的对象
    d.pop("amp_helper", None)
    return d


# ── Warmup + Cosine LR Scheduler ─────────────────────────────────────────────


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """线性 warmup + cosine 衰减（顶会标准：MAE/SimMIM/BEiT 均采用此策略）。

    Args:
        warmup_epochs: 线性 warmup 的 epoch 数，0 = 禁用 warmup。
        total_epochs: 总训练 epoch 数。
        min_lr_ratio: cosine 最低 lr 与初始 lr 的比值，默认 0（衰减至 0）。
    """
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def add_warmup_args(parser, default_warmup: int = 5) -> None:
    """为下游脚本添加 warmup 参数（LambdaLR warmup cosine）。"""
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=default_warmup,
        help=f"线性 warmup epoch 数，0=禁用（默认 {default_warmup}）",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.0,
        help="cosine 衰减终点 lr / 初始 lr 比值（默认 0.0）",
    )


def add_eval_interval_args(parser, default: int = 1) -> None:
    """每隔 N epoch 验证一次（最后一 epoch 必验证）。"""
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=default,
        help=f"验证间隔 epoch 数，1=每 epoch 验证（默认 {default}）；末 epoch 必验证",
    )


# ── 向量化 mIoU（替代逐像素 Python 循环，快 100x+）────────────────────────────


class SegMIoUMeter:
    """分割 mIoU 累计器：batch 级 GPU bincount，epoch 末一次汇总。"""

    def __init__(self, num_classes: int, ignore_index: int = -1, device: Optional[torch.device] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device or torch.device("cpu")
        self.conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=self.device)

    def reset(self) -> None:
        self.conf.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred/target: (B,H,W) 或 (H,W)，可在 GPU 上。"""
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        pred = pred.reshape(-1).to(torch.int64)
        target = target.reshape(-1).to(torch.int64)
        if self.ignore_index >= 0:
            valid = target != self.ignore_index
            pred = pred[valid]
            target = target[valid]
        if pred.numel() == 0:
            return
        k = self.num_classes
        idx = target * k + pred
        batch_conf = torch.bincount(idx, minlength=k * k).reshape(k, k)
        self.conf += batch_conf.to(self.conf.device)

    def compute(self) -> float:
        conf = self.conf.float()
        inter = torch.diag(conf)
        union = conf.sum(dim=1) + conf.sum(dim=0) - inter
        valid = union > 0
        if not bool(valid.any()):
            return 0.0
        iou = inter[valid] / union[valid].clamp(min=1e-6)
        return float(iou.mean().item())


# ── SECOND 四指标累计器（OA / F1 / mIoU_sc / Sek）─────────────────────────────


class SECONDMetricMeter:
    """SECOND 语义变化检测四指标累计器。

    标签约定: 0 = 未变化; 1~(K-1) = 变化后语义类别（共 K-1 = 6 类）。

    - OA    : 全像素准确率（K 类）
    - F1    : 二值变化检测 F1（changed vs unchanged）
    - mIoU  : 语义变化 mIoU（仅 class 1~K-1，**不含** class 0，与 ChangeMamba 口径一致）
    - Sek   : Separated kappa（仅在 GT=changed 的像素上、(K-1) 类问题的 κ 系数）
    """

    def __init__(self, num_classes: int = 7, device: Optional[torch.device] = None):
        self.K = num_classes
        self.device = device or torch.device("cpu")
        self.conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=self.device)

    def reset(self) -> None:
        self.conf.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.reshape(-1).long().clamp(0, self.K - 1)
        target = target.reshape(-1).long()
        valid = (target >= 0) & (target < self.K)
        pred, target = pred[valid], target[valid]
        idx = target * self.K + pred
        self.conf += torch.bincount(idx, minlength=self.K * self.K).reshape(
            self.K, self.K
        ).to(self.conf.device)

    def compute(self) -> dict:
        C = self.conf.float()
        K = self.K

        # ── OA (Overall Accuracy, all K classes) ─────────────────────────
        total = C.sum().clamp(min=1)
        oa = float((C.diagonal().sum() / total).item())

        # ── F1 (binary: changed vs unchanged) ────────────────────────────
        tp = C[1:, 1:].sum()
        fp = C[0, 1:].sum()
        fn = C[1:, 0].sum()
        f1 = float((2 * tp / (2 * tp + fp + fn).clamp(min=1)).item())

        # ── mIoU over semantic change classes (1~K-1, excluding class 0) ──
        # 必须用完整混淆矩阵的行/列，否则 class0 贡献的 FP/FN 被丢弃
        # 导致 union 虚小、IoU 虚高（极端情况 mIoU=1.0 的 bug 根源）
        inter = C.diagonal()[1:]                      # TP for class 1..K-1
        union = C[1:, :].sum(1) + C[:, 1:].sum(0) - inter   # FN + FP 均来自全矩阵
        valid_cls = union > 0
        miou_sc = float(
            (inter[valid_cls] / union[valid_cls].clamp(min=1e-6)).mean().item()
        ) if bool(valid_cls.any()) else 0.0

        # ── Sek (Separated kappa on GT-changed pixels, K-1 classes) ──────
        # Rows = GT class 1~K-1 (changed GT pixels), Cols = pred 0~K-1
        C_ch = C[1:, :]                              # (K-1) × K
        n_ch = C_ch.sum().clamp(min=1)
        # Correct: diag of C_ch[:, 1:] (pred correctly as changed class)
        po = C_ch[:, 1:].diagonal().sum() / n_ch
        row_sum = C_ch.sum(1)                         # [K-1]
        col_sum = C_ch[:, 1:].sum(0)                  # [K-1]
        pe = (row_sum @ col_sum) / (n_ch * n_ch)
        sek = float(((po - pe) / (1.0 - pe).clamp(min=1e-6)).item())

        return {"OA": oa, "F1": f1, "mIoU": miou_sc, "Sek": sek}


# ── 参数组（bias + norm 不加 weight_decay）───────────────────────────────────


def param_groups_weight_decay(
    model: nn.Module,
    weight_decay: float,
    no_decay_names: Optional[List[str]] = None,
) -> List[dict]:
    """将模型参数分为 decay 组 和 no_decay 组（顶会 AdamW 标准做法）。

    bias 和 1-D 参数（LayerNorm/BatchNorm weight/bias）不加 weight_decay。
    额外可通过 no_decay_names 指定名称包含特定字符串的参数组。
    """
    if no_decay_names is None:
        no_decay_names = []
    decay_params: List[torch.Tensor] = []
    no_decay_params: List[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or any(nd in name for nd in no_decay_names)
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def param_groups_layerwise_decay(
    model: nn.Module,
    weight_decay: float,
    lr: float,
    layer_decay: float = 0.75,
    num_layers: Optional[int] = None,
) -> List[dict]:
    """Layer-wise LR decay（ELECTRA/BEiT/MAE 微调时常用）。

    每层 lr = base_lr * layer_decay^(num_layers - layer_idx)。
    适用于 EarthMamba 的 4 个 stage（depths=[2,2,27,2]）。
    """
    if num_layers is None:
        num_layers = 4  # EarthMamba 默认 4 stages

    def _get_layer_id(name: str) -> int:
        if "patch_embed" in name or "pos_embed" in name:
            return 0
        for i in range(num_layers):
            if f"layers.{i}" in name:
                return i + 1
        return num_layers  # classifier / head

    groups: dict[str, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _get_layer_id(name)
        layer_lr = lr * (layer_decay ** (num_layers - layer_id))
        is_no_decay = param.ndim <= 1 or name.endswith(".bias")
        key = f"layer{layer_id}_{'nodecay' if is_no_decay else 'decay'}"
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": layer_lr,
                "weight_decay": 0.0 if is_no_decay else weight_decay,
            }
        groups[key]["params"].append(param)

    return list(groups.values())
