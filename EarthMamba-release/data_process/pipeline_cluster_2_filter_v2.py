"""
pipeline_cluster_2_filter_v2.py — Step 2（GPU 加速版，自包含、无外部 dataset 依赖）

为什么有这个文件？
  上一个 _fix 版本用 importlib 加载外部 pipeline_cluster_dataset_fix.py，
  导致 DataLoader 多进程 spawn 时 pickle 类引用失效（PicklingError）。
  本文件**所有依赖都内嵌在脚本本身**，不需要任何外部 dataset 模块，
  就不会再出现「J 盘有个旧文件抢先 import」的问题。

加速点（vs. 原版 pipeline_cluster_2_filter.py）
  1. ImageEmbedDatasetFast（cv2 解码，跳过 PIL，1.5-2.5× 读图速度）
  2. GPU 超球筛选（torch.matmul 替代 numpy，5-20×）
  3. embed_batch_size 默认 512（原 128），GPU 利用率显著提升
  4. num_workers 默认 16，prefetch_factor=8
  5. cv2.setNumThreads(0) 让 worker 不抢核
  6. channels_last 内存格式，conv 推理更快
  7. AMP 用新式 torch.amp.autocast（消 FutureWarning）
  8. 默认禁用 torch.compile（部分 PyTorch 版本编译会报错）；如需启用 --use_compile

调用命令（参数与原版完全一致）：
  python pipeline_cluster_2_filter_v2.py --pool_dir "G:\\images_v2" \\
      --feature_root J:\\datasets\\feature --seed_mult 2.2 --ds_mult 0.5
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import re
import sys
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# cv2 是关键加速点；没装就退回 PIL
try:
    import cv2 as _cv2
    _HAS_CV2 = True
    _cv2.setNumThreads(0)
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

# ImageNet 归一化（预乘 255 后直接作用于 uint8→float32）
_MEAN = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
_STD  = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)


# ═════════════════════════ Dataset（顶层定义，spawn-safe） ═════════════════════════

class ImageEmbedDatasetFast(Dataset):
    """cv2 解码 + numpy 归一化，输出 (3, img_size, img_size) float32 Tensor。"""

    def __init__(self, paths: list[str], img_size: int) -> None:
        self.paths = paths
        self.img_size = int(img_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        if _HAS_CV2:
            t = self._load_cv2(p)
            if t is not None:
                return idx, t
        return idx, self._load_pil(p)

    def _load_cv2(self, p: str):
        try:
            bgr = _cv2.imread(p, _cv2.IMREAD_COLOR)  # type: ignore[union-attr]
            if bgr is None:
                return None
            s = self.img_size
            if bgr.shape[0] != s or bgr.shape[1] != s:
                bgr = _cv2.resize(bgr, (s, s), interpolation=_cv2.INTER_LINEAR)
            rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB).astype(np.float32, copy=False)
            rgb = (rgb - _MEAN) / _STD
            return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        except Exception:
            return None

    def _load_pil(self, p: str):
        try:
            import torchvision.transforms.functional as TF
            s = self.img_size
            img = Image.open(p).convert("RGB")
            if img.size != (s, s):
                try:
                    resample = Image.Resampling.BILINEAR
                except AttributeError:
                    resample = Image.BILINEAR  # type: ignore[attr-defined]
                img = img.resize((s, s), resample)
            t = TF.to_tensor(img)
            t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return t
        except Exception:
            return None


def embed_collate(batch):
    idxs: list[int] = []
    tensors: list[torch.Tensor] = []
    for idx, t in batch:
        if t is None:
            continue
        idxs.append(int(idx))
        tensors.append(t)
    if not tensors:
        return None
    return idxs, torch.stack(tensors, dim=0)


# ═════════════════════════ 工具函数 ═════════════════════════

def collect_image_paths(root: str) -> list[str]:
    root = os.path.abspath(os.path.normpath(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"目录不存在: {root}")
    out: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(dirpath, fn))
    return out


def load_checkpoint(weight_path: str) -> dict:
    print("Loading checkpoint...", flush=True)
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    print(f"  Checkpoint keys: {len(ckpt) if hasattr(ckpt, '__len__') else 'n/a'}", flush=True)
    return ckpt


def parse_img_size_from_ckpt(ckpt: dict) -> tuple[int, int]:
    for k in ckpt.keys():
        if "pos_embed" in k:
            pos_embed = ckpt[k]
            token_num = pos_embed.shape[1]
            embed_dim = pos_embed.shape[2]
            grid_size = int(math.sqrt(token_num - 1))
            img_size = grid_size * 16
            print(f"  embed_dim={embed_dim}  grid={grid_size}  img_size={img_size}", flush=True)
            return img_size, embed_dim
    raise RuntimeError("pos_embed not found in checkpoint")


def build_model(
    img_size: int, ckpt: dict, device: str,
    multi_gpu: bool = True, use_compile: bool = False,
):
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0, img_size=img_size
    )
    model_dict = model.state_dict()
    new_ckpt = {
        k: v for k, v in ckpt.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model.load_state_dict(new_ckpt, strict=False)
    model.to(device)
    model.eval()

    # channels_last 在 conv-based 模型上更快；ViT 也兼容
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    if use_compile:
        try:
            model = torch.compile(model)
            print("  [torch.compile] 编译成功", flush=True)
        except Exception as e:
            print(f"  [torch.compile] 跳过: {e}", flush=True)

    if multi_gpu and device == "cuda":
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = nn.DataParallel(model)
            print(f"  [multi-gpu] DataParallel，GPUs={n_gpu}", flush=True)

    return model


# ═════════════════════════ Embedding 提取 ═════════════════════════

@torch.no_grad()
def extract_embeddings(
    img_list: list[str],
    img_size: int,
    model,
    device: str,
    batch_size: int = 512,
    desc: str = "",
    *,
    num_workers: int = 16,
    use_amp: bool = True,
    pin_memory: bool | None = None,
) -> tuple[list[str], np.ndarray]:
    if not img_list:
        return [], np.empty((0, 0), dtype=np.float32)

    if pin_memory is None:
        pin_memory = device == "cuda"

    ds = ImageEmbedDatasetFast(img_list, img_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=embed_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=8 if num_workers > 0 else None,
    )

    print(
        f"  [embed] {desc}: {len(img_list):,} 张 → {len(loader)} batch"
        f"（batch_size={batch_size}, workers={num_workers}, prefetch=8）",
        flush=True,
    )

    n_total = len(img_list)
    all_feat: np.ndarray | None = None
    ok_mask = np.zeros(n_total, dtype=bool)

    if use_amp and device == "cuda":
        autocast_ctx = lambda: torch.amp.autocast("cuda")  # noqa: E731
    else:
        autocast_ctx = nullcontext

    pbar = tqdm(loader, desc=desc or "embed", total=len(loader), leave=False, unit="batch")
    for batch in pbar:
        if batch is None:
            continue
        idxs, x = batch
        x = x.to(device, non_blocking=pin_memory)
        try:
            x = x.to(memory_format=torch.channels_last)
        except Exception:
            pass
        with autocast_ctx():
            out = model(x)
        out_np = out.float().cpu().numpy()
        norms = np.linalg.norm(out_np, axis=1)
        valid = norms >= 1e-12
        denom = np.maximum(norms, 1e-12).reshape(-1, 1)
        out_np = (out_np / denom).astype(np.float32)

        if all_feat is None:
            all_feat = np.empty((n_total, out_np.shape[1]), dtype=np.float32)

        idxs_arr = np.asarray(idxs, dtype=np.int64)
        if valid.any():
            keep_idx = idxs_arr[valid]
            all_feat[keep_idx] = out_np[valid]
            ok_mask[keep_idx] = True

    if all_feat is None or not ok_mask.any():
        if n_total > 0:
            print(f"  [embed] 全部 {n_total:,} 张解码/embedding 失败", flush=True)
        return [], np.empty((0, 0), dtype=np.float32)

    ok_pos = np.where(ok_mask)[0]
    n_fail = n_total - len(ok_pos)
    if n_fail > 0:
        print(f"  [embed] 解码失败 {n_fail:,}/{n_total:,} 张（已跳过）", flush=True)
    return [img_list[int(i)] for i in ok_pos], np.ascontiguousarray(all_feat[ok_pos])


# ═════════════════════════ 聚类特征加载 ═════════════════════════

def load_all_cluster_features(feature_dir: str) -> tuple[np.ndarray, np.ndarray]:
    feature_dir = os.path.abspath(feature_dir)
    if not os.path.isdir(feature_dir):
        print(f"  [WARN] 特征目录不存在: {feature_dir}", flush=True)
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    files = sorted(glob.glob(os.path.join(feature_dir, "**", "clusters.npz"), recursive=True))
    if not files:
        print(f"  [WARN] 未找到任何 clusters.npz: {feature_dir}", flush=True)
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    centers_list: list[np.ndarray] = []
    radii_list: list[np.ndarray] = []
    total_images = 0
    for fp in files:
        data = np.load(fp)
        c = data["centers"].astype(np.float32)
        r = data["radii"].astype(np.float32)
        n = int(data.get("n_images", 0))
        rel = os.path.relpath(fp, feature_dir)
        print(
            f"  加载: {rel:50s}  K={len(r):3d}  半径[{r.min():.3f}~{r.max():.3f}]  n_images={n:,}",
            flush=True,
        )
        centers_list.append(c)
        radii_list.append(r)
        total_images += n

    all_centers = np.concatenate(centers_list, axis=0)
    all_radii = np.concatenate(radii_list, axis=0)
    print(
        f"  合计簇数: {len(all_radii):,}  特征维度: {all_centers.shape[1]}  "
        f"覆盖图片: {total_images:,}",
        flush=True,
    )
    return all_centers, all_radii


# ═════════════════════════ 超球筛选（GPU） ═════════════════════════

@torch.no_grad()
def sphere_filter(
    feat_np: np.ndarray,
    centers_np: np.ndarray,
    radii_np: np.ndarray,
    multiplier: float,
    chunk: int,
    device: str,
) -> np.ndarray:
    N = feat_np.shape[0]
    if centers_np.shape[0] == 0:
        return np.ones(N, dtype=bool)

    if device == "cuda" and torch.cuda.is_available():
        dev = "cuda"
        centers = torch.from_numpy(centers_np).to(dev)
        radii_sq = torch.from_numpy((radii_np * multiplier) ** 2).to(dev)

        result = torch.zeros(N, dtype=torch.bool, device="cpu")
        desc = f"sphere_filter_gpu(mult={multiplier:.2f})"
        for start in tqdm(range(0, N, chunk), desc=desc, leave=False):
            end = min(start + chunk, N)
            c = torch.from_numpy(feat_np[start:end]).to(dev)
            dots = c @ centers.T
            dist_sq = torch.clamp(2.0 - 2.0 * dots, min=0.0)
            mask = (dist_sq <= radii_sq.unsqueeze(0)).any(dim=1)
            result[start:end] = mask.cpu()
        return result.numpy()
    else:
        radii_sq = (radii_np * multiplier) ** 2
        result = np.zeros(N, dtype=bool)
        desc = f"sphere_filter_cpu(mult={multiplier:.2f})"
        for start in tqdm(range(0, N, chunk), desc=desc, leave=False):
            end = min(start + chunk, N)
            ck = feat_np[start:end]
            dots = ck @ centers_np.T
            dist_sq = np.clip(2.0 - 2.0 * dots, 0.0, None)
            result[start:end] = (dist_sq <= radii_sq[None, :]).any(axis=1)
        return result


# ═════════════════════════ KMeans 均衡采样 ═════════════════════════

def kmeans_balanced_sample(
    paths: list[str],
    feats: np.ndarray,
    k: int,
    per_cluster_ratio: float,
    random_state: int,
) -> tuple[list[str], np.ndarray]:
    N = len(paths)
    if N == 0:
        return [], np.empty((0, feats.shape[1] if feats.ndim == 2 else 0), dtype=np.float32)

    K = max(1, min(k, N))
    print(f"  KMeans K={K}，per_cluster_ratio={per_cluster_ratio:.2f}…", flush=True)

    if K == 1:
        rng = random.Random(random_state)
        n_keep = max(1, round(N * per_cluster_ratio))
        idx = list(range(N))
        rng.shuffle(idx)
        idx = sorted(idx[:n_keep])
        return [paths[i] for i in idx], feats[np.asarray(idx, dtype=np.int64)]

    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init="auto", max_iter=300)
    labels = kmeans.fit_predict(feats)
    print("  KMeans 完成，开始按簇采样…", flush=True)

    rng = random.Random(random_state)
    selected: list[int] = []
    cluster_stats: list[tuple[int, int, int]] = []
    for kk in range(K):
        cluster_idx = np.where(labels == kk)[0].tolist()
        if not cluster_idx:
            continue
        n_keep = max(1, round(len(cluster_idx) * per_cluster_ratio))
        rng.shuffle(cluster_idx)
        chosen = sorted(cluster_idx[:n_keep])
        selected.extend(chosen)
        cluster_stats.append((kk, len(cluster_idx), len(chosen)))

    selected.sort()
    sel_paths = [paths[i] for i in selected]
    sel_feats = np.ascontiguousarray(feats[np.asarray(selected, dtype=np.int64)])
    sizes = [s for _, s, _ in cluster_stats]
    kepts = [kp for _, _, kp in cluster_stats]
    print(
        f"  采样后: {len(sel_paths):,} 张（{K} 簇，"
        f"各簇大小 {min(sizes)}~{max(sizes)}，各簇保留 {min(kepts)}~{max(kepts)}）",
        flush=True,
    )
    return sel_paths, sel_feats


def write_txt(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"  已写: {path}  ({len(lines):,} 行)", flush=True)


# ═════════════════════════ argparse ═════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="三步 pool 筛选（GPU 加速）：超球过滤 → KMeans 采样 → downstream 去重"
    )
    p.add_argument("--pool_dir", type=str, required=True)
    p.add_argument("--feature_root", default=r"J:\datasets\feature")
    p.add_argument("--weight_path", default=r"J:\datasets\weights\satmae-pretrain-vit-base-e199.pth")
    p.add_argument("--output_dir", default=r"J:\datasets\results")
    # Step 1
    p.add_argument("--seed_mult", type=float, default=1.5)
    # Step 2
    p.add_argument("--kmeans_k", type=int, default=200)
    p.add_argument("--per_cluster_ratio", type=float, default=0.5)
    # Step 3
    p.add_argument("--ds_mult", type=float, default=0.5)
    # 推理参数
    p.add_argument("--embed_batch_size", type=int, default=512,
                   help="GPU 推理 batch（默认 512；显存不够可 256/384）")
    p.add_argument("--num_workers", type=int, default=16,
                   help="DataLoader 读图进程数（默认 16）")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_multi_gpu", action="store_true")
    p.add_argument("--use_compile", action="store_true",
                   help="启用 torch.compile（部分版本可能编译失败，默认关闭）")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--filter_chunk_size", type=int, default=32768,
                   help="GPU 球筛分块大小（默认 32768，更大利用率更高）")
    return p.parse_args()


# ═════════════════════════ main ═════════════════════════

def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pool_name = re.sub(r"[^\w\-.]+", "_", os.path.basename(
        os.path.normpath(args.pool_dir)
    )).strip("._") or "pool"

    out_abs = os.path.abspath(os.path.join(args.output_dir, pool_name))
    os.makedirs(out_abs, exist_ok=True)
    print(f"输出目录: {out_abs}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    print(
        f"[filter_pool_v2] device={device}  GPUs={n_gpu}  "
        f"AMP={not args.no_amp}  DataParallel={not args.no_multi_gpu and n_gpu > 1}  "
        f"compile={args.use_compile}  num_workers={args.num_workers}  "
        f"embed_batch_size={args.embed_batch_size}",
        flush=True,
    )
    if device == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    print("\n━━━ 加载 seed 聚类特征 ━━━", flush=True)
    seed_centers, seed_radii = load_all_cluster_features(
        os.path.join(args.feature_root, "seed")
    )
    print("\n━━━ 加载 downstream 聚类特征 ━━━", flush=True)
    ds_centers, ds_radii = load_all_cluster_features(
        os.path.join(args.feature_root, "downstream")
    )

    print("\n━━━ 加载模型 ━━━", flush=True)
    ckpt = load_checkpoint(args.weight_path)
    img_size, _ = parse_img_size_from_ckpt(ckpt)
    model = build_model(
        img_size, ckpt, device,
        multi_gpu=not args.no_multi_gpu,
        use_compile=args.use_compile,
    )

    print(f"\n━━━ 扫描 pool: {args.pool_dir} ━━━", flush=True)
    pool_all_paths = collect_image_paths(args.pool_dir)
    print(f"  pool 图片总数: {len(pool_all_paths):,}", flush=True)
    if not pool_all_paths:
        print("[ERROR] 无图片，退出", flush=True)
        return

    print("\n━━━ 提取 pool embedding ━━━", flush=True)
    ok_paths, ok_feat = extract_embeddings(
        pool_all_paths,
        img_size,
        model,
        device,
        batch_size=args.embed_batch_size,
        desc="pool embed",
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )
    print(f"  成功 embedding: {len(ok_paths):,} / {len(pool_all_paths):,}", flush=True)
    if not ok_paths:
        print("[ERROR] 无有效 embedding，退出", flush=True)
        return

    # Step 1
    print(f"\n━━━ Step 1：seed 超球筛选  seed_mult={args.seed_mult} ━━━", flush=True)
    if seed_centers.shape[0] == 0:
        print("  [WARN] 无 seed 聚类特征，Step 1 跳过", flush=True)
        step1_mask = np.ones(len(ok_paths), dtype=bool)
    else:
        step1_mask = sphere_filter(
            ok_feat, seed_centers, seed_radii,
            multiplier=args.seed_mult,
            chunk=args.filter_chunk_size,
            device=device,
        )
    step1_pos = np.where(step1_mask)[0]
    step1_paths_raw = [ok_paths[int(i)] for i in step1_pos]
    seen: dict[str, int] = {}
    dedup_rel_idx: list[int] = []
    for rel_i, p in enumerate(step1_paths_raw):
        if p not in seen:
            seen[p] = rel_i
            dedup_rel_idx.append(rel_i)
    n_dup = len(step1_paths_raw) - len(dedup_rel_idx)
    step1_paths = [step1_paths_raw[i] for i in dedup_rel_idx]
    step1_feat = np.ascontiguousarray(
        ok_feat[step1_pos[np.asarray(dedup_rel_idx, dtype=np.int64)]]
    )
    print(
        f"  Step 1 保留: {len(step1_paths):,} / {len(ok_paths):,} "
        f"（{100*len(step1_paths)/max(len(ok_paths),1):.1f}%）"
        + (f"  [去重 {n_dup:,} 条重复]" if n_dup else ""),
        flush=True,
    )
    step1_txt = os.path.join(out_abs, f"{pool_name}_{ts}_step1_seed_filtered.txt")
    write_txt(step1_txt, step1_paths)
    if not step1_paths:
        return

    # Step 2
    print(
        f"\n━━━ Step 2：KMeans 均衡采样  "
        f"kmeans_k={args.kmeans_k}  per_cluster_ratio={args.per_cluster_ratio} ━━━",
        flush=True,
    )
    step2_paths, step2_feat = kmeans_balanced_sample(
        step1_paths, step1_feat,
        k=args.kmeans_k,
        per_cluster_ratio=args.per_cluster_ratio,
        random_state=args.random_state,
    )
    print(f"  Step 2 保留: {len(step2_paths):,} / {len(step1_paths):,}", flush=True)
    step2_txt = os.path.join(out_abs, f"{pool_name}_{ts}_step2_kmeans_sampled.txt")
    write_txt(step2_txt, step2_paths)
    if not step2_paths:
        return

    # Step 3
    print(f"\n━━━ Step 3：downstream 去重  ds_mult={args.ds_mult} ━━━", flush=True)
    if ds_centers.shape[0] == 0:
        print("  [WARN] 无 downstream 聚类特征，Step 3 跳过", flush=True)
        final_paths = step2_paths
    else:
        ds_mask = sphere_filter(
            step2_feat, ds_centers, ds_radii,
            multiplier=args.ds_mult,
            chunk=args.filter_chunk_size,
            device=device,
        )
        final_paths = [step2_paths[i] for i in range(len(step2_paths)) if not ds_mask[i]]
        print(
            f"  Step 3 去除: {ds_mask.sum():,} 张  保留: {len(final_paths):,} / {len(step2_paths):,}",
            flush=True,
        )

    final_txt = os.path.join(out_abs, f"{pool_name}_{ts}_step3_final_dataset.txt")
    write_txt(final_txt, final_paths)

    dup_note = f"  （去重 {n_dup:,} 条）" if n_dup else ""
    print(
        f"\n{'='*60}\n[汇总] {pool_name}\n"
        f"  pool 扫描        : {len(pool_all_paths):,} 张\n"
        f"  有效 embedding   : {len(ok_paths):,} 张\n"
        f"  Step1 seed 过滤后: {len(step1_paths):,} 张{dup_note}\n"
        f"    → {step1_txt}\n"
        f"  Step2 KMeans 后  : {len(step2_paths):,} 张\n"
        f"    → {step2_txt}\n"
        f"  Step3 ds 去重后  : {len(final_paths):,} 张\n"
        f"    → {final_txt}\n"
        f"  输出目录: {out_abs}\n"
        f"{'='*60}",
        flush=True,
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
