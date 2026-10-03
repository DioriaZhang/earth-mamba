"""
Earth-Mamba 专用训练实现（由仓库根目录 train.py 调用）。

从 train_ddp_configs_benchmark_v2_monitor.py 迁移：仅保留 EarthMamba、DDP/FSDP、
数据加载、监控与日志逻辑；不依赖父目录或其它对比模型仓库。
"""
from __future__ import annotations

import argparse
import builtins
import glob
import inspect
import io
import json
import math
import os
import sys
import threading
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml

    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import lmdb

    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import selective_scan_cuda_oflex

    sys.modules["selective_scan_cuda"] = selective_scan_cuda_oflex
except ImportError:
    pass

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class FlatFolderDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.root_dir = root_dir
        self.img_size = img_size
        self.images = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.images = [
            x
            for x in self.images
            if x.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"))
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert("RGB")
            image = self.transform(image)
            return image, 0
        except Exception:
            return torch.zeros(3, self.img_size, self.img_size), 0


class LmdbImageDataset(Dataset):
    """LMDB：键 __len__ + 索引；多进程下 per-worker 延迟打开 env。"""

    def __init__(self, lmdb_path, img_size=224, key_fmt="decimal"):
        if not HAS_LMDB:
            raise RuntimeError("未安装 lmdb，请: pip install lmdb")
        self.img_size = img_size
        self.key_fmt = key_fmt
        self._lmdb_path = lmdb_path
        self.env = None
        _env_tmp = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        with _env_tmp.begin(write=False) as txn:
            raw = txn.get(b"__len__")
            if raw is None:
                raise RuntimeError("LMDB 缺少键 __len__，无法确定数据集大小")
            self.length = int(raw.decode("ascii"))
        _env_tmp.close()
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self._lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )

    def __len__(self):
        return self.length

    def _key(self, idx):
        if self.key_fmt == "padded8":
            return f"{idx:08d}".encode("ascii")
        return str(idx).encode("ascii")

    def __getitem__(self, idx):
        self._get_env()
        with self.env.begin(write=False) as txn:
            buf = txn.get(self._key(idx))
            if buf is None and self.key_fmt == "decimal":
                buf = txn.get(f"{idx:08d}".encode("ascii"))
        if buf is None:
            return torch.zeros(3, self.img_size, self.img_size), 0
        try:
            image = Image.open(io.BytesIO(buf)).convert("RGB")
            return self.transform(image), 0
        except Exception:
            return torch.zeros(3, self.img_size, self.img_size), 0


def setup_ddp():
    if "RANK" not in os.environ:
        return 0, False
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, True


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_master():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def build_earth_mamba_model(args, cfg, num_classes):
    from earth_mamba.models.earth_mamba import EarthMamba

    ssm_version = getattr(args, "ssm_version", "mamba1")
    ssm_d_state_default = 64 if ssm_version == "mamba3" else 16
    ssm_d_state = getattr(args, "ssm_d_state", ssm_d_state_default)
    if ssm_d_state <= 0:
        ssm_d_state = ssm_d_state_default

    init_kwargs = {
        "depths": cfg["depths"],
        "dims": cfg["dims"],
        "patch_size": getattr(args, "patch_size", 4),
        "in_chans": 3,
        "num_classes": num_classes,
        "ssm_d_state": ssm_d_state,
        "ssm_ratio": 2.0,
        "use_armg": True,
        "use_graph": True,
        "norm_layer": "ln",
        "imgsize": getattr(args, "img_size", 224),
    }
    sig = inspect.signature(EarthMamba.__init__).parameters
    if "forward_type" in sig:
        init_kwargs["forward_type"] = "v2"
    if "use_checkpoint" in sig:
        init_kwargs["use_checkpoint"] = bool(getattr(args, "use_grad_checkpoint", False))
    if "ssm_backend" in sig and getattr(args, "ssm_backend", "").strip():
        init_kwargs["ssm_backend"] = args.ssm_backend.strip()
    if "ssm_version" in sig:
        init_kwargs["ssm_version"] = ssm_version
    if "ssm_headdim" in sig:
        init_kwargs["ssm_headdim"] = getattr(args, "ssm_headdim", 64)
    return EarthMamba(**init_kwargs)


def build_model(args):
    size = args.model_size.lower()
    num_classes = 1000
    cfgs = {
        "tiny": {"depths": [2, 2, 9, 2], "dims": [96, 192, 384, 768]},
        "small": {"depths": [2, 2, 27, 2], "dims": [96, 192, 384, 768]},
        "base": {"depths": [2, 2, 27, 2], "dims": [128, 256, 512, 1024]},
    }
    cfg = cfgs.get(size, cfgs["tiny"])
    if is_master():
        print(f"构建模型: EARTH_MAMBA | size={size.upper()} | cfg={cfg}")
    return build_earth_mamba_model(args, cfg, num_classes)


def build_scheduler(args, optimizer, steps_per_epoch):
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ResourceMonitor:
    def __init__(self, device, interval_sec=1.0):
        self.device = device
        self.interval_sec = max(0.2, float(interval_sec))
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._start_time = None
        self._ps_proc = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        self._nvml_ready = False
        self._gpu_count = 0

    def _init_nvml(self):
        if not HAS_PYNVML:
            return
        try:
            pynvml.nvmlInit()
            self._gpu_count = int(pynvml.nvmlDeviceGetCount())
            self._nvml_ready = self._gpu_count > 0
        except Exception:
            self._gpu_count = 0
            self._nvml_ready = False

    def _gpu_stats(self):
        out = {
            "gpu_util_mean_percent": None,
            "gpu_mem_used_percent_mean": None,
            "gpu_mem_used_mb_mean": None,
            "gpus": [],
            "gpu_power_w_sum": None,
            "torch_mem_alloc_mb": None,
            "torch_mem_reserved_mb": None,
        }
        if self.device.type == "cuda":
            idx = self.device.index if self.device.index is not None else 0
            try:
                out["torch_mem_alloc_mb"] = round(torch.cuda.memory_allocated(idx) / (1024**2), 2)
                out["torch_mem_reserved_mb"] = round(torch.cuda.memory_reserved(idx) / (1024**2), 2)
            except Exception:
                pass

        if not self._nvml_ready or self._gpu_count <= 0:
            if self.device.type == "cuda":
                _util_fn = getattr(torch.cuda, "utilization", None)
                if callable(_util_fn):
                    try:
                        _di = self.device.index if self.device.index is not None else 0
                        u = float(_util_fn(device=_di))
                        out["gpu_util_mean_percent"] = round(u, 2)
                        mem_total, mem_free = torch.cuda.mem_get_info(_di)
                        used = mem_total - mem_free
                        used_pct = (100.0 * float(used) / float(mem_total)) if mem_total else 0.0
                        out["gpu_mem_used_percent_mean"] = round(used_pct, 2)
                        out["gpu_mem_used_mb_mean"] = round(used / (1024**2), 2)
                        out["gpus"] = [
                            {
                                "index": _di,
                                "gpu_util_percent": u,
                                "gpu_mem_used_mb": round(used / (1024**2), 2),
                                "gpu_mem_total_mb": round(mem_total / (1024**2), 2),
                                "gpu_mem_used_percent": round(used_pct, 2),
                            }
                        ]
                    except Exception:
                        pass
            return out

        gpus = []
        power_sum = 0.0
        power_any = False
        try:
            for gi in range(self._gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gi)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_pct = (100.0 * float(mem.used) / float(mem.total)) if mem.total else 0.0
                one = {
                    "index": gi,
                    "gpu_util_percent": float(util.gpu),
                    "gpu_mem_used_mb": round(mem.used / (1024**2), 2),
                    "gpu_mem_total_mb": round(mem.total / (1024**2), 2),
                    "gpu_mem_used_percent": round(used_pct, 2),
                }
                try:
                    pw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    one["gpu_power_w"] = round(pw, 2)
                    power_sum += pw
                    power_any = True
                except Exception:
                    one["gpu_power_w"] = None
                gpus.append(one)
        except Exception:
            return out

        out["gpus"] = gpus
        if gpus:
            out["gpu_util_mean_percent"] = round(sum(g["gpu_util_percent"] for g in gpus) / len(gpus), 2)
            out["gpu_mem_used_percent_mean"] = round(
                sum(g["gpu_mem_used_percent"] for g in gpus) / len(gpus), 2
            )
            out["gpu_mem_used_mb_mean"] = round(sum(g["gpu_mem_used_mb"] for g in gpus) / len(gpus), 2)
        if power_any:
            out["gpu_power_w_sum"] = round(power_sum, 2)
        return out

    def _cpu_stats(self):
        out = {
            "cpu_percent_mean": None,
            "cpu_percent_max_core": None,
            "ram_percent": None,
            "proc_cpu_percent": None,
            "proc_rss_mb": None,
        }
        if not HAS_PSUTIL:
            return out
        try:
            percpu = psutil.cpu_percent(interval=None, percpu=True)
            if percpu:
                out["cpu_percent_mean"] = round(sum(percpu) / len(percpu), 2)
                out["cpu_percent_max_core"] = round(max(percpu), 2)
            out["ram_percent"] = round(psutil.virtual_memory().percent, 2)
            out["proc_cpu_percent"] = self._ps_proc.cpu_percent(interval=None)
            if out["proc_cpu_percent"] is not None:
                out["proc_cpu_percent"] = round(float(out["proc_cpu_percent"]), 2)
            out["proc_rss_mb"] = round(self._ps_proc.memory_info().rss / (1024**2), 2)
        except Exception:
            pass
        return out

    def sample_once(self):
        if self._start_time is None:
            self._start_time = time.time()
        sample = {
            "time_s": round(time.time() - self._start_time, 3),
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        sample.update(self._cpu_stats())
        sample.update(self._gpu_stats())
        self.samples.append(sample)
        return sample

    def _loop(self):
        self.sample_once()
        while not self._stop.is_set():
            time.sleep(self.interval_sec)
            self.sample_once()

    def start(self):
        self._start_time = time.time()
        self._init_nvml()
        if HAS_PSUTIL:
            try:
                self._ps_proc.cpu_percent(interval=0.05)
                psutil.cpu_percent(interval=0.1, percpu=True)
            except Exception:
                try:
                    self._ps_proc.cpu_percent(interval=None)
                    psutil.cpu_percent(interval=None, percpu=True)
                except Exception:
                    pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _deps_hint():
        if not HAS_PSUTIL:
            print("[监控] 未安装 psutil：CPU/内存占用率将为 None。安装: pip install psutil")
        if not HAS_PYNVML:
            print("[监控] 未安装 pynvml：全卡 NVML 统计不可用。安装: pip install nvidia-ml-py3 或 pynvml")
        elif torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                pynvml.nvmlShutdown()
            except Exception:
                print(
                    "[监控] NVML 初始化失败（权限/驱动）：将尝试用 torch.cuda.utilization / mem_get_info（若 PyTorch 版本支持）"
                )

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_ready = False

    def summary(self):
        if not self.samples:
            return {}
        out = {
            "num_samples": len(self.samples),
            "gpu_count_nvml": self._gpu_count,
        }

        def series_avg_max_min(key):
            vals = [x[key] for x in self.samples if isinstance(x.get(key), (int, float))]
            if not vals:
                return
            out[f"{key}_time_avg"] = round(sum(vals) / len(vals), 3)
            out[f"{key}_time_max"] = round(max(vals), 3)
            out[f"{key}_time_min"] = round(min(vals), 3)

        series_avg_max_min("cpu_percent_mean")
        series_avg_max_min("cpu_percent_max_core")
        series_avg_max_min("ram_percent")
        series_avg_max_min("gpu_util_mean_percent")
        series_avg_max_min("gpu_mem_used_percent_mean")
        series_avg_max_min("gpu_mem_used_mb_mean")
        series_avg_max_min("torch_mem_alloc_mb")
        series_avg_max_min("torch_mem_reserved_mb")

        core_peaks = [
            x["cpu_percent_max_core"]
            for x in self.samples
            if isinstance(x.get("cpu_percent_max_core"), (int, float))
        ]
        if core_peaks:
            out["cpu_single_core_percent_peak"] = round(max(core_peaks), 3)

        ram_peaks = [x["ram_percent"] for x in self.samples if isinstance(x.get("ram_percent"), (int, float))]
        if ram_peaks:
            out["ram_percent_peak"] = round(max(ram_peaks), 3)

        gpu_util_peak = None
        gpu_mem_pct_peak = None
        for x in self.samples:
            for g in x.get("gpus") or []:
                u = g.get("gpu_util_percent")
                m = g.get("gpu_mem_used_percent")
                if isinstance(u, (int, float)):
                    gpu_util_peak = u if gpu_util_peak is None else max(gpu_util_peak, u)
                if isinstance(m, (int, float)):
                    gpu_mem_pct_peak = m if gpu_mem_pct_peak is None else max(gpu_mem_pct_peak, m)
        if gpu_util_peak is not None:
            out["gpu_single_card_util_percent_peak"] = round(gpu_util_peak, 3)
        if gpu_mem_pct_peak is not None:
            out["gpu_single_card_mem_used_percent_peak"] = round(gpu_mem_pct_peak, 3)

        out["metrics_for_plotting"] = {
            "cpu_percent_mean_time_avg": out.get("cpu_percent_mean_time_avg"),
            "ram_percent_time_avg": out.get("ram_percent_time_avg"),
            "gpu_util_mean_percent_time_avg": out.get("gpu_util_mean_percent_time_avg"),
            "gpu_mem_used_percent_mean_time_avg": out.get("gpu_mem_used_percent_time_avg"),
            "cpu_single_core_percent_peak": out.get("cpu_single_core_percent_peak"),
            "ram_percent_peak": out.get("ram_percent_peak"),
            "gpu_single_card_util_percent_peak": out.get("gpu_single_card_util_percent_peak"),
            "gpu_single_card_mem_used_percent_peak": out.get("gpu_single_card_mem_used_percent_peak"),
        }
        return out


def save_resource_curve_plots(output_dir, samples, monitor_summary):
    if not HAS_MATPLOTLIB or not samples:
        return
    import matplotlib.pyplot as plt

    times = [s["time_s"] for s in samples]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_xlabel("time_s")
    ax1.set_ylabel("CPU % (mean cores)", color="tab:blue")
    cpu_m = [s.get("cpu_percent_mean") for s in samples]
    if any(isinstance(v, (int, float)) for v in cpu_m):
        ax1.plot(times, cpu_m, color="tab:blue", label="cpu_mean")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("RAM %", color="tab:green")
    ram = [s.get("ram_percent") for s in samples]
    if any(isinstance(v, (int, float)) for v in ram):
        ax2.plot(times, ram, color="tab:green", label="ram_percent")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    fig.suptitle("CPU (mean) & RAM vs time")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "resource_cpu_ram_curve.png"))
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_xlabel("time_s")
    ax1.set_ylabel("GPU util % (mean all cards)", color="tab:orange")
    gu = [s.get("gpu_util_mean_percent") for s in samples]
    if any(isinstance(v, (int, float)) for v in gu):
        ax1.plot(times, gu, color="tab:orange", label="gpu_util_mean")
    ax1.tick_params(axis="y", labelcolor="tab:orange")

    ax2 = ax1.twinx()
    ax2.set_ylabel("GPU VRAM used % (mean all cards)", color="tab:red")
    gm = [s.get("gpu_mem_used_percent_mean") for s in samples]
    if any(isinstance(v, (int, float)) for v in gm):
        ax2.plot(times, gm, color="tab:red", label="gpu_mem_used_pct_mean")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.suptitle("GPU mean util & VRAM % vs time (NVML all devices)")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "resource_gpu_mean_curve.png"))
    plt.close()

    max_g = 0
    for s in samples:
        max_g = max(max_g, len(s.get("gpus") or []))
    if max_g > 1:
        fig, axes = plt.subplots(max_g, 1, figsize=(10, 2.2 * max_g), sharex=True)
        if max_g == 1:
            axes = [axes]
        for gi in range(max_g):
            u_series = []
            m_series = []
            for s in samples:
                glist = s.get("gpus") or []
                if gi < len(glist):
                    u_series.append(glist[gi].get("gpu_util_percent"))
                    m_series.append(glist[gi].get("gpu_mem_used_percent"))
                else:
                    u_series.append(None)
                    m_series.append(None)
            axu = axes[gi]
            axm = axu.twinx()
            axu.plot(times, u_series, color="tab:orange", label="util%")
            axm.plot(times, m_series, color="tab:red", label="vram%")
            axu.set_ylabel(f"GPU{gi} util %")
            axm.set_ylabel(f"GPU{gi} VRAM %")
        axes[-1].set_xlabel("time_s")
        fig.suptitle("Per-GPU util & VRAM % vs time")
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, "resource_gpu_per_card_curve.png"))
        plt.close()


def wrap_fsdp_model(model, local_rank, use_amp, amp_dtype):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from earth_mamba.models.earth_mamba_block import EarthMambaBlock

    import functools

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=frozenset({EarthMambaBlock}),
    )

    mp = None
    if use_amp and amp_dtype == "bf16":
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif use_amp and amp_dtype == "fp16":
        mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    return FSDP(
        model,
        device_id=local_rank,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap,
        use_orig_params=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Earth-Mamba 训练（DDP / FSDP，本仓库自包含）")
    # 与旧版多模型脚本兼容：仅接受 earth，其它值请用仓库外的对比脚本
    parser.add_argument(
        "--model",
        type=str,
        default="earth",
        choices=["earth"],
        help="固定为 earth（保留该参数以便沿用旧命令行）",
    )
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--data_dir", type=str, default="", help="图像目录（与 --lmdb_path 二选一）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--patch_size",
        type=int,
        default=4,
        help="patch 大小；设为 16 可减少 token",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true", default=False)
    parser.add_argument("--lmdb_path", type=str, default="")
    parser.add_argument("--lmdb_key_fmt", type=str, default="decimal", choices=["decimal", "padded8"])
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["bf16", "fp16"])
    parser.add_argument("--use_grad_checkpoint", action="store_true", default=False)
    parser.add_argument("--ssm_fp32", action="store_true", default=False)
    parser.add_argument("--ssm_backend", type=str, default="")
    parser.add_argument(
        "--ssm_version",
        type=str,
        default="mamba1",
        choices=["mamba1", "mamba3"],
    )
    parser.add_argument("--ssm_headdim", type=int, default=64)
    parser.add_argument(
        "--ssm_d_state",
        type=int,
        default=0,
        help="0=自动（mamba1→16, mamba3→64）",
    )
    parser.add_argument("--fsdp", action="store_true", default=False)
    parser.add_argument("--sync_bn", action="store_true", default=False)
    parser.add_argument("--find_unused_parameters", action="store_true", default=True)
    parser.add_argument("--no_find_unused_parameters", action="store_false", dest="find_unused_parameters")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--save_ckpt", action="store_true", default=False)
    parser.add_argument("--monitor_interval", type=float, default=1.0)
    parser.add_argument("--disable_monitor", action="store_true", default=False)
    parser.add_argument("--step_log_every", type=int, default=1)
    args = parser.parse_args()

    if not args.lmdb_path and not args.data_dir:
        print("错误：请指定 --data_dir 或 --lmdb_path")
        return
    if args.ssm_fp32:
        os.environ["EARTH_MAMBA_SELECTIVE_SCAN_FP32"] = "1"
    if args.ssm_backend.strip():
        os.environ["EARTH_MAMBA_SSM_BACKEND"] = args.ssm_backend.strip()

    local_rank, use_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    if device.type == "cuda" and args.amp and args.amp_dtype == "bf16":
        _cc = torch.cuda.get_device_capability(device)
        if _cc[0] < 8:
            print(
                f"[警告] 当前 GPU compute capability={_cc[0]}.{_cc[1]}，"
                "BF16 无硬件原生支持时请改用 --amp_dtype fp16。\n"
                "  → mamba3 Triton 核可能自动降级为 FP16。"
            )

    if not is_master():
        builtins.print = lambda *a, **k: None

    if is_master():
        print("====== Earth-Mamba 训练 ======")
        print("model_size:", args.model_size)
        print("data:", args.lmdb_path or args.data_dir)
        print("batch_size:", args.batch_size, "img_size:", args.img_size, "patch_size:", args.patch_size)
        print("epochs:", args.epochs, "steps_per_epoch:", args.steps_per_epoch)
        print("lr:", args.lr, "weight_decay:", args.weight_decay)
        print("amp:", args.amp, "amp_dtype:", args.amp_dtype if args.amp else "—", "ssm_fp32:", args.ssm_fp32)
        print("grad_checkpoint:", args.use_grad_checkpoint, "fsdp:", args.fsdp)
        print("repo_root:", _REPO_ROOT)
        if use_ddp:
            print("world_size:", dist.get_world_size())

    if args.lmdb_path:
        if not HAS_LMDB:
            if is_master():
                print("错误：使用 --lmdb_path 需要 pip install lmdb")
            cleanup_ddp()
            return
        dataset = LmdbImageDataset(args.lmdb_path, img_size=args.img_size, key_fmt=args.lmdb_key_fmt)
    else:
        dataset = FlatFolderDataset(args.data_dir, img_size=args.img_size)
    if len(dataset) == 0:
        if is_master():
            print("错误：数据集为空")
        cleanup_ddp()
        return

    sampler = DistributedSampler(dataset) if use_ddp else None
    dl_kwargs = dict(
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = args.prefetch_factor
        dl_kwargs["persistent_workers"] = args.persistent_workers
    loader = DataLoader(dataset, **dl_kwargs)

    model = build_model(args).to(device)
    if use_ddp and args.sync_bn and not args.fsdp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    fsdp_used = False
    if use_ddp and args.fsdp and dist.get_world_size() > 1:
        try:
            model = wrap_fsdp_model(model, local_rank, args.amp, args.amp_dtype)
            fsdp_used = True
            if is_master():
                print("[并行] 已启用 FSDP FULL_SHARD")
        except Exception as e:
            if is_master():
                print("FSDP 包装失败，回退 DDP:", e)
            fsdp_used = False
    if use_ddp and not fsdp_used:
        eff_find_unused = bool(args.find_unused_parameters)
        if args.use_grad_checkpoint:
            eff_find_unused = False
            if is_master():
                print(
                    "[DDP] --use_grad_checkpoint：find_unused_parameters=False（避免与重入 checkpoint 冲突）"
                )
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=eff_find_unused,
        )
        if args.use_grad_checkpoint and hasattr(model, "_set_static_graph"):
            model._set_static_graph()
            if is_master():
                print("[DDP] 已启用 _set_static_graph()")
        elif args.use_grad_checkpoint and is_master():
            print(
                "[DDP][警告] 无 DDP._set_static_graph；+ checkpoint 可能 backward 失败，请 --fsdp 或关闭 checkpoint。"
            )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = None
    if args.amp and args.amp_dtype == "fp16" and device.type == "cuda":
        if fsdp_used:
            try:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

                scaler = ShardedGradScaler()
                if is_master():
                    print("[AMP] FSDP+FP16: 已启用 ShardedGradScaler")
            except ImportError:
                if is_master():
                    print("[警告] ShardedGradScaler 不可用，建议升级 PyTorch 或改用 bf16（需 A100+）")
        else:
            scaler = torch.amp.GradScaler("cuda")

    steps_per_epoch = min(args.steps_per_epoch, len(loader))
    scheduler = build_scheduler(args, optimizer, steps_per_epoch)

    loss_history = []
    lr_history = []
    total_start = time.time()

    step_log_f = None
    monitor = None
    step_log_path = None
    resource_log_path = None

    if is_master() and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        step_log_path = os.path.join(args.output_dir, "step_log.jsonl")
        resource_log_path = os.path.join(args.output_dir, "resource_log.jsonl")
        step_log_f = open(step_log_path, "w", encoding="utf-8")

    if is_master() and (not args.disable_monitor):
        ResourceMonitor._deps_hint()
        monitor = ResourceMonitor(device=device, interval_sec=args.monitor_interval)
        monitor.start()

    try:
        for epoch in range(args.epochs):
            if use_ddp and sampler is not None:
                sampler.set_epoch(epoch)
            epoch_start = time.time()
            model.train()
            iterator = tqdm(
                loader,
                total=steps_per_epoch,
                disable=not is_master(),
                desc=f"Epoch {epoch + 1}/{args.epochs}",
            )

            for step, (images, labels) in enumerate(iterator):
                if step >= steps_per_epoch:
                    break
                step_start = time.time()

                images = images.to(device, non_blocking=(device.type == "cuda"))
                labels = labels.to(device, non_blocking=(device.type == "cuda"))
                optimizer.zero_grad()

                use_autocast = args.amp and device.type == "cuda" and not fsdp_used
                amp_dtype_torch = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

                if scaler is not None:
                    with torch.amp.autocast("cuda", dtype=amp_dtype_torch):
                        outputs = model(images)
                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                elif use_autocast:
                    with torch.amp.autocast("cuda", dtype=amp_dtype_torch):
                        outputs = model(images)
                        if isinstance(outputs, (tuple, list)):
                            outputs = outputs[0]
                        loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    outputs = model(images)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                cur_lr = optimizer.param_groups[0]["lr"]

                if is_master():
                    loss_value = float(loss.item())
                    loss_history.append(loss_value)
                    lr_history.append(cur_lr)

                    if step % 10 == 0:
                        iterator.set_postfix({"loss": f"{loss_value:.4f}", "lr": f"{cur_lr:.2e}"})

                    if step_log_f is not None and step % max(1, args.step_log_every) == 0:
                        row = {
                            "epoch": epoch + 1,
                            "step_in_epoch": step,
                            "global_step": epoch * steps_per_epoch + step,
                            "loss": loss_value,
                            "lr": cur_lr,
                            "step_time_s": round(time.time() - step_start, 4),
                        }
                        if device.type == "cuda":
                            idx = device.index if device.index is not None else 0
                            row["torch_mem_alloc_mb"] = round(torch.cuda.memory_allocated(idx) / (1024**2), 2)
                            row["torch_mem_reserved_mb"] = round(torch.cuda.memory_reserved(idx) / (1024**2), 2)
                        step_log_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if is_master():
                print(f"  Epoch {epoch + 1} 耗时: {time.time() - epoch_start:.2f}s")

    finally:
        if monitor is not None:
            monitor.stop()
        if step_log_f is not None:
            step_log_f.close()

    total_time = time.time() - total_start
    if is_master():
        max_mem = 0.0
        if device.type == "cuda":
            max_mem = torch.cuda.max_memory_allocated() / 1024**3
        world_size = dist.get_world_size() if use_ddp else 1
        total_images = steps_per_epoch * args.epochs * args.batch_size * world_size
        images_per_sec = total_images / total_time if total_time > 0 else 0

        monitor_summary = {}
        if monitor is not None:
            monitor_summary = monitor.summary()

        print("====== 训练结束 ======")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  吞吐: {images_per_sec:.1f} images/s")
        if device.type == "cuda":
            print(f"  单卡最大显存: {max_mem:.2f} GB")
        if monitor_summary:
            m = monitor_summary.get("metrics_for_plotting") or {}
            print("  资源监控摘要:", monitor_summary)
            if m:
                print(
                    "  [采样时间平均] CPU均值%:",
                    m.get("cpu_percent_mean_time_avg"),
                    " 内存%:",
                    m.get("ram_percent_time_avg"),
                    " GPU利用率均值%:",
                    m.get("gpu_util_mean_percent_time_avg"),
                    " GPU显存占用率均值%:",
                    m.get("gpu_mem_used_percent_mean_time_avg"),
                )
                print(
                    "  [全过程峰值] 单核CPU%:",
                    m.get("cpu_single_core_percent_peak"),
                    " 内存%:",
                    m.get("ram_percent_peak"),
                    " 单卡GPU利用率%:",
                    m.get("gpu_single_card_util_percent_peak"),
                    " 单卡显存占用率%:",
                    m.get("gpu_single_card_mem_used_percent_peak"),
                )

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            log = {
                "model": "earth",
                "model_size": args.model_size,
                "data_dir": args.data_dir,
                "lmdb_path": args.lmdb_path or None,
                "earth_mamba_repo_root": _REPO_ROOT,
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "patch_size": args.patch_size,
                "epochs": args.epochs,
                "steps_per_epoch": steps_per_epoch,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "scheduler": "cosine",
                "amp": args.amp,
                "amp_dtype": args.amp_dtype if args.amp else None,
                "ssm_fp32": args.ssm_fp32,
                "ssm_backend": args.ssm_backend or None,
                "ssm_version": args.ssm_version,
                "ssm_headdim": args.ssm_headdim,
                "use_grad_checkpoint": args.use_grad_checkpoint,
                "fsdp": fsdp_used,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
                "total_time_s": round(total_time, 2),
                "images_per_sec": round(images_per_sec, 2),
                "max_memory_gb": round(max_mem, 2),
                "world_size": world_size,
                "loss_history": loss_history,
                "lr_history": lr_history,
                "monitor_interval_s": args.monitor_interval,
                "monitor_enabled": (not args.disable_monitor),
                "monitor_summary": monitor_summary,
                "resource_metrics": (monitor_summary or {}).get("metrics_for_plotting"),
                "step_log_path": step_log_path,
                "resource_log_path": resource_log_path,
            }
            with open(os.path.join(args.output_dir, "train_log.json"), "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2, ensure_ascii=False)

            if monitor is not None and monitor.samples:
                with open(resource_log_path, "w", encoding="utf-8") as f:
                    for s in monitor.samples:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
                with open(os.path.join(args.output_dir, "resource_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(monitor_summary, f, indent=2, ensure_ascii=False)

            if HAS_MATPLOTLIB:
                import matplotlib.pyplot as plt

                plt.figure()
                plt.plot(loss_history)
                plt.title("Loss curve")
                plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
                plt.close()

                plt.figure()
                plt.plot(lr_history)
                plt.title("LR curve (cosine)")
                plt.savefig(os.path.join(args.output_dir, "lr_curve.png"))
                plt.close()
                print("  已保存 loss_curve.png, lr_curve.png")

            if monitor is not None and monitor.samples:
                save_resource_curve_plots(args.output_dir, monitor.samples, monitor_summary)
                print("  已保存 resource_*_curve.png（若 matplotlib 可用）")

            if args.save_ckpt:
                if fsdp_used:
                    _sd = model.state_dict()
                elif use_ddp:
                    _sd = model.module.state_dict()
                else:
                    _sd = model.state_dict()
                state = {
                    "model": "earth",
                    "model_size": args.model_size,
                    "img_size": args.img_size,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "fsdp": fsdp_used,
                    "state_dict": _sd,
                }
                ckpt = os.path.join(args.output_dir, f"earth_{args.model_size}_cfg.pth")
                torch.save(state, ckpt)
                print("  已保存 checkpoint:", ckpt)

            print("  已保存日志:")
            print("   -", os.path.join(args.output_dir, "train_log.json"))
            if step_log_path is not None:
                print("   -", step_log_path)
            if resource_log_path is not None and monitor is not None:
                print("   -", resource_log_path)
                print("   -", os.path.join(args.output_dir, "resource_summary.json"))

    cleanup_ddp()


if __name__ == "__main__":
    main()
