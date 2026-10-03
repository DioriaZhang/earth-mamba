"""
pipeline_cluster_1_build.py — Step 1

对 seed / downstream 各二级子文件夹提取聚类特征 + 最小覆盖半径，
结果保存到：
    {feature_root}/seed/{子文件夹名}/clusters.npz
    {feature_root}/downstream/{子文件夹名}/clusters.npz

聚类半径定义：cluster k 的半径 = max_{i∈cluster_k} ||feat_i - center_k||_2
（所有 feat 均已 L2 归一化；center 在聚类后重新 L2 归一化）

.npz 字段：
    centers  : (K, D) float32  L2 归一化簇中心
    radii    : (K,)  float32   对应簇最小覆盖半径（L2 距离）
    n_images : int             成功 embedding 的图片数量

用法：
    python build_cluster_features.py
    python build_cluster_features.py --n_clusters 100 --overwrite
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from contextlib import nullcontext

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from pipeline_cluster_dataset import ImageEmbedDataset, embed_collate

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


# ─────────────────────────── 图片扫描 ────────────────────────────


def collect_image_paths(root: str) -> list[str]:
    root = os.path.abspath(os.path.normpath(root))
    if not os.path.isdir(root):
        return []
    out: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(dirpath, fn))
    return out


def find_immediate_subdirs(root: str) -> list[tuple[str, str]]:
    """返回 root 下所有直接子文件夹的 (名称, 绝对路径) 列表，按名称排序。"""
    root = os.path.abspath(os.path.normpath(root))
    if not os.path.isdir(root):
        print(f"[WARN] 目录不存在: {root}", flush=True)
        return []
    result = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            result.append((name, path))
    return result


# ─────────────────────────── 模型加载 ────────────────────────────


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


def build_model(img_size: int, ckpt: dict, device: str, multi_gpu: bool = True):
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
    if multi_gpu and device == "cuda":
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = nn.DataParallel(model)
            print(f"  [multi-gpu] DataParallel，GPUs={n_gpu}", flush=True)
    return model


# ─────────────────────────── Embedding 提取 ──────────────────────


@torch.no_grad()
def extract_embeddings(
    img_list: list[str],
    transform,
    model,
    device: str,
    batch_size: int = 128,
    desc: str = "",
    *,
    num_workers: int = 8,
    use_amp: bool = True,
    pin_memory: bool | None = None,
) -> tuple[list[str], np.ndarray]:
    """返回 (成功路径列表, L2 归一化 embedding 矩阵 (N,D) float32)。"""
    if not img_list:
        return [], np.empty((0, 0), dtype=np.float32)

    if pin_memory is None:
        pin_memory = device == "cuda"

    prefetch = 2 if sys.platform == "win32" else 4
    ds = ImageEmbedDataset(img_list, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=embed_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch if num_workers > 0 else None,
    )

    n_total = len(img_list)
    all_feat: np.ndarray | None = None
    ok_mask = np.zeros(n_total, dtype=bool)

    autocast_ctx = torch.cuda.amp.autocast if (use_amp and device == "cuda") else nullcontext

    pbar = tqdm(loader, desc=desc or "embed", total=len(loader), leave=False, unit="batch")
    for batch in pbar:
        if batch is None:
            continue
        idxs, x = batch
        x = x.to(device, non_blocking=pin_memory)
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
        return [], np.empty((0, 0), dtype=np.float32)

    ok_pos = np.where(ok_mask)[0]
    n_fail = n_total - len(ok_pos)
    if n_fail > 0:
        print(f"  [embed] {n_fail:,}/{n_total:,} 张解码失败（已跳过）", flush=True)
    return [img_list[int(i)] for i in ok_pos], np.ascontiguousarray(all_feat[ok_pos])


# ─────────────────────────── 聚类 + 最小覆盖半径 ─────────────────


def compute_cluster_features(
    feats: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对 L2 归一化特征做 KMeans，计算每簇最小覆盖半径。

    Returns:
        centers : (K, D) float32  L2 归一化后的簇中心
        radii   : (K,)  float32   r_k = max_{i∈cluster_k} ||feat_i - center_k||_2
    """
    N, D = feats.shape
    K = max(1, min(n_clusters, N))

    if K == 1:
        center = feats.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(center, axis=1, keepdims=True)
        center = (center / np.maximum(norm, 1e-12)).astype(np.float32)
        dists = np.linalg.norm(feats - center[0], axis=1)
        return center, np.array([float(dists.max()) if N > 0 else 0.0], dtype=np.float32)

    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init="auto", max_iter=300)
    labels = kmeans.fit_predict(feats)
    raw_centers = kmeans.cluster_centers_  # (K, D)，不是 L2 归一化的

    # 重归一化簇中心（均值不保证单位范数）
    norms = np.linalg.norm(raw_centers, axis=1, keepdims=True)
    centers = (raw_centers / np.maximum(norms, 1e-12)).astype(np.float32)

    # 各簇最小覆盖半径
    radii = np.zeros(K, dtype=np.float32)
    for k in range(K):
        mask = labels == k
        if not mask.any():
            continue
        dists = np.linalg.norm(feats[mask] - centers[k], axis=1)
        radii[k] = float(dists.max())

    return centers, radii


# ─────────────────────────── 单子文件夹处理 ──────────────────────


def process_one_subdir(
    subdir_name: str,
    subdir_path: str,
    out_dir: str,
    model,
    transform,
    device: str,
    args,
) -> bool:
    out_file = os.path.join(out_dir, "clusters.npz")
    if os.path.exists(out_file) and not args.overwrite:
        print(f"  [skip] 已存在（--overwrite 覆盖）: {out_file}", flush=True)
        return True

    paths = collect_image_paths(subdir_path)
    if not paths:
        print(f"  [skip] 无图片: {subdir_path}", flush=True)
        return False
    print(f"  [{subdir_name}] 扫描到图片: {len(paths):,}", flush=True)

    ok_paths, feats = extract_embeddings(
        paths,
        transform,
        model,
        device,
        batch_size=args.embed_batch_size,
        desc=f"embed {subdir_name}",
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )
    if not ok_paths:
        print(f"  [skip] 无有效 embedding: {subdir_name}", flush=True)
        return False

    K_actual = min(args.n_clusters, len(ok_paths))
    print(
        f"  [{subdir_name}] 有效 embedding: {len(ok_paths):,}，"
        f"开始 KMeans K={K_actual}…",
        flush=True,
    )
    centers, radii = compute_cluster_features(feats, args.n_clusters, args.random_state)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out_file,
        centers=centers,
        radii=radii,
        n_images=np.int64(len(ok_paths)),
    )
    print(
        f"  [{subdir_name}] 已保存: {out_file}\n"
        f"    K={len(centers)}  半径[min={radii.min():.4f}  mean={radii.mean():.4f}  max={radii.max():.4f}]",
        flush=True,
    )
    return True


# ─────────────────────────── 主函数 ──────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "构建 seed/downstream 聚类特征 + 最小覆盖半径，供 pipeline_cluster_2_filter.py 使用。\n"
            "seed 直接传入两个数据集目录（可在任意盘），输出特征固定命名为 Globe230k / LuoJia-CDKD。\n"
            "至少传入 --globe230k 或 --luojia 其中一个。"
        )
    )
    # ── seed：直接传路径，固定输出名 ─────────────────────────────
    p.add_argument(
        "--globe230k",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Globe230k 预处理后的图片目录\n"
            "（preprocess_seed.py --globe230k 的输出，可在任意盘）\n"
            "特征写入 {feature_root}/seed/Globe230k/clusters.npz"
        ),
    )
    p.add_argument(
        "--luojia",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "LuoJia-CDKD 预处理后的图片目录\n"
            "（preprocess_seed.py --luojia 的输出，可在任意盘）\n"
            "特征写入 {feature_root}/seed/LuoJia-CDKD/clusters.npz"
        ),
    )
    p.add_argument(
        "--downstream_dirs", nargs="+", default=[r"J:\datasets\Downstream-datasets"],
        help="downstream 根目录列表（递归到二级子文件夹，可多个，可在任意盘）",
    )
    p.add_argument(
        "--feature_root", default=r"J:\datasets\feature",
        help="特征输出根目录；seed→feature/seed/，downstream→feature/downstream/",
    )
    p.add_argument(
        "--weight_path", default=r"J:\datasets\weights\satmae-pretrain-vit-base-e199.pth",
    )
    p.add_argument(
        "--n_clusters", type=int, default=50,
        help="每个子文件夹的 KMeans 簇数（图片数不足时自动收紧）",
    )
    p.add_argument("--embed_batch_size", type=int, default=128)
    p.add_argument(
        "--num_workers", type=int, default=8,
        help="DataLoader 读图并行进程数（Windows 建议 4-8）",
    )
    p.add_argument("--no_amp", action="store_true", help="关闭混合精度推理")
    p.add_argument("--no_multi_gpu", action="store_true", help="关闭 DataParallel")
    p.add_argument(
        "--overwrite", action="store_true",
        help="重新计算已存在的 clusters.npz（默认跳过）",
    )
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    print(
        f"[build_cluster_features] device={device}  GPUs={n_gpu}  "
        f"AMP={not args.no_amp}  DataParallel={not args.no_multi_gpu and n_gpu > 1}  "
        f"num_workers={args.num_workers}  embed_batch_size={args.embed_batch_size}",
        flush=True,
    )

    ckpt = load_checkpoint(args.weight_path)
    img_size, _ = parse_img_size_from_ckpt(ckpt)
    model = build_model(img_size, ckpt, device, multi_gpu=not args.no_multi_gpu)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if device == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    # ── seed：固定两个数据集，直接按名处理 ──────────────────────
    seed_named: list[tuple[str, str]] = []   # (固定输出名, 图片目录路径)
    if args.globe230k:
        seed_named.append(("Globe230k", os.path.abspath(args.globe230k)))
    if args.luojia:
        seed_named.append(("LuoJia-CDKD", os.path.abspath(args.luojia)))

    if not seed_named:
        print("[WARN] 未传入 --globe230k 或 --luojia，跳过 seed 特征构建", flush=True)
    else:
        sep = "=" * 60
        print(f"\n{sep}\n处理 [seed] 数据集\n{sep}", flush=True)
        seed_out_root = os.path.join(args.feature_root, "seed")
        os.makedirs(seed_out_root, exist_ok=True)
        for name, path in seed_named:
            out_dir = os.path.join(seed_out_root, name)
            print(f"\n→ {name}  ({path})", flush=True)
            try:
                process_one_subdir(name, path, out_dir, model, transform, device, args)
            except Exception as e:
                print(
                    f"  [ERROR] {name} 处理失败，已跳过: {type(e).__name__}: {e}",
                    flush=True,
                )

    # ── downstream：保持遍历二级子文件夹逻辑 ─────────────────────
    sep = "=" * 60
    print(f"\n{sep}\n处理 [downstream] 数据集\n{sep}", flush=True)
    ds_out_root = os.path.join(args.feature_root, "downstream")
    os.makedirs(ds_out_root, exist_ok=True)

    for root in args.downstream_dirs:
        subdirs = find_immediate_subdirs(root)
        if not subdirs:
            print(f"[WARN] 无子文件夹或目录不存在: {root}", flush=True)
            continue
        print(f"\n[downstream] 根目录: {root}\n子文件夹数: {len(subdirs)}", flush=True)
        for subdir_name, subdir_path in subdirs:
            out_dir = os.path.join(ds_out_root, subdir_name)
            print(f"\n→ {subdir_name}  ({subdir_path})", flush=True)
            try:
                process_one_subdir(
                    subdir_name, subdir_path, out_dir,
                    model, transform, device, args,
                )
            except Exception as e:
                print(
                    f"  [ERROR] {subdir_name} 处理失败，已跳过: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

    print("\n全部完成。特征根目录:", os.path.abspath(args.feature_root), flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

