"""
pipeline_cluster_2_filter.py — Step 2

基于 pipeline_cluster_1_build.py 预计算的聚类特征对 pool 数据集进行三步筛选：

  Step 1  超球筛选   —— 保留在任意 seed 簇中心 × seed_mult 半径内的 pool 图
                        → 输出 {pool}_{ts}_step1_seed_filtered.txt
  Step 2  KMeans 均衡采样 —— 对候选集聚类，每簇按 per_cluster_ratio 比例随机抽样
  Step 3  downstream 去重 —— 去除在任意 downstream 簇中心 × ds_mult 半径内的图片
                        → 输出 {pool}_{ts}_final_dataset.txt

距离度量：L2-归一化特征空间上的 L2 距离（等价于 cosine 距离的单调变换）。
         dist²(a,b) = 2 - 2·(a·b)（a,b 均为单位向量）

用法：
    python filter_pool_by_clusters.py --pool_dir J:\\datasets\\pool_datasets\\opensatmap
    python filter_pool_by_clusters.py \\
        --pool_dir J:\\datasets\\pool_datasets\\opensatmap \\
        --seed_mult 2.0 --ds_mult 0.8 \\
        --kmeans_k 200 --per_cluster_ratio 0.5
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
        raise FileNotFoundError(f"目录不存在: {root}")
    out: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(dirpath, fn))
    return out


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
    """返回 (成功路径列表, L2 归一化 embedding 矩阵 (N, D) float32)。"""
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

    print(
        f"  [embed] {desc}: {len(img_list):,} 张 → {len(loader)} batch"
        f"（batch_size={batch_size}, workers={num_workers}）",
        flush=True,
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
        if n_total > 0:
            print(f"  [embed] 全部 {n_total:,} 张解码/embedding 失败", flush=True)
        return [], np.empty((0, 0), dtype=np.float32)

    ok_pos = np.where(ok_mask)[0]
    n_fail = n_total - len(ok_pos)
    if n_fail > 0:
        print(f"  [embed] 解码失败 {n_fail:,}/{n_total:,} 张（已跳过）", flush=True)
    return [img_list[int(i)] for i in ok_pos], np.ascontiguousarray(all_feat[ok_pos])


# ─────────────────────────── 聚类特征加载 ────────────────────────


def load_all_cluster_features(
    feature_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    递归加载 feature_dir 下所有 clusters.npz。
    返回：
        all_centers : (K_total, D) float32  拼接后所有簇中心
        all_radii   : (K_total,)  float32  对应簇半径
    若目录为空或无文件，返回 shape=(0,0) 和 shape=(0,) 的空数组。
    """
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
            f"  加载: {rel:50s}  K={len(r):3d}  "
            f"半径[{r.min():.3f}~{r.max():.3f}]  n_images={n:,}",
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


# ─────────────────────────── 超球筛选 ────────────────────────────


def sphere_filter(
    feat: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    multiplier: float,
    chunk_size: int = 4096,
) -> np.ndarray:
    """
    向量化超球筛选。
    返回 bool mask (N,)：True 表示该图片在 ANY 簇的 radius×multiplier 球内。

    利用 L2 归一化向量性质：dist²(a,b) = 2 - 2·(a·b)，避免显式开根号。
    内存峰值约 chunk_size × K_total × 4 bytes（chunk_size=4096, K=500 → 8 MB）。
    """
    N = feat.shape[0]
    if centers.shape[0] == 0:
        # 无聚类特征 → 全部通过（保守）
        return np.ones(N, dtype=bool)

    radii_sq = (radii * multiplier) ** 2  # (K,)
    result = np.zeros(N, dtype=bool)

    desc = f"sphere_filter(mult={multiplier:.1f})"
    for start in tqdm(range(0, N, chunk_size), desc=desc, leave=False):
        end = min(start + chunk_size, N)
        chunk = feat[start:end]               # (cb, D)
        dots = chunk @ centers.T              # (cb, K)
        dist_sq = np.clip(2.0 - 2.0 * dots, 0.0, None)  # (cb, K)
        result[start:end] = (dist_sq <= radii_sq[None, :]).any(axis=1)

    return result


# ─────────────────────────── KMeans 均衡采样 ─────────────────────


def kmeans_balanced_sample(
    paths: list[str],
    feats: np.ndarray,
    k: int,
    per_cluster_ratio: float,
    random_state: int,
) -> tuple[list[str], np.ndarray]:
    """
    对候选集做 KMeans，每簇按 per_cluster_ratio 比例随机抽样（至少 1 张）。
    保证最终样本在视觉空间中均衡分布（多样性）。

    Args:
        k                  : KMeans 簇数，自动收紧到 min(k, len(paths))
        per_cluster_ratio  : (0,1]，每簇保留比例，如 0.5 表示每簇留一半

    Returns:
        sel_paths : 抽样后的路径列表
        sel_feats : 对应 embedding (M, D)
    """
    N = len(paths)
    if N == 0:
        return [], np.empty((0, feats.shape[1] if feats.ndim == 2 else 0), dtype=np.float32)

    K = max(1, min(k, N))
    print(f"  KMeans K={K}，per_cluster_ratio={per_cluster_ratio:.2f}…", flush=True)

    if K == 1:
        # 只有一簇，直接按比例抽
        rng = random.Random(random_state)
        n_keep = max(1, round(N * per_cluster_ratio))
        idx = list(range(N))
        rng.shuffle(idx)
        idx = sorted(idx[:n_keep])
        return [paths[i] for i in idx], feats[np.asarray(idx, dtype=np.int64)]

    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init="auto", max_iter=300)
    labels = kmeans.fit_predict(feats)
    print(f"  KMeans 完成，开始按簇采样…", flush=True)

    rng = random.Random(random_state)
    selected: list[int] = []
    cluster_stats: list[tuple[int, int, int]] = []   # (cluster_id, size, kept)

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
        f"各簇大小 {min(sizes)}~{max(sizes)}，"
        f"各簇保留 {min(kepts)}~{max(kepts)}）",
        flush=True,
    )
    return sel_paths, sel_feats


# ─────────────────────────── 写 txt ──────────────────────────────


def write_txt(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"  已写: {path}  ({len(lines):,} 行)", flush=True)


# ─────────────────────────── 参数 ────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="三步 pool 筛选：超球过滤 → KMeans 均衡采样 → downstream 去重"
    )
    p.add_argument(
        "--pool_dir", type=str, required=True,
        help="pool 数据集目录（直接读取其中图片，不进子文件夹）",
    )
    p.add_argument(
        "--feature_root", default=r"J:\datasets\feature",
        help="pipeline_cluster_1_build.py 的输出根目录（含 seed/ 和 downstream/ 子目录）",
    )
    p.add_argument(
        "--weight_path", default=r"J:\datasets\weights\satmae-pretrain-vit-base-e199.pth",
    )
    p.add_argument(
        "--output_dir", default=r"J:\datasets\results",
        help="结果根目录；每次运行在其下创建 {pool_name}/ 子目录存放 3 份 txt（默认 J:\\datasets\\results）",
    )
    # Step 1
    p.add_argument(
        "--seed_mult", type=float, default=1.5,
        help="seed 聚类半径倍数（越大越宽松，默认 1.5；推荐 1.5~2.0）",
    )
    # Step 2
    p.add_argument(
        "--kmeans_k", type=int, default=200,
        help="KMeans 聚类数，超过候选数量时自动收紧（默认 200）",
    )
    p.add_argument(
        "--per_cluster_ratio", type=float, default=0.5,
        help="每簇保留比例，(0,1]，如 0.5 表示每簇留一半（默认 0.5）",
    )
    # Step 3
    p.add_argument(
        "--ds_mult", type=float, default=0.5,
        help="downstream 聚类半径倍数（越小越宽松，去除更少；默认 0.5；推荐 0.5~1.0）",
    )
    # 通用
    p.add_argument("--embed_batch_size", type=int, default=128)
    p.add_argument(
        "--num_workers", type=int, default=8,
        help="DataLoader 并行进程数（Windows 建议 4-8）",
    )
    p.add_argument("--no_amp", action="store_true", help="关闭混合精度推理")
    p.add_argument("--no_multi_gpu", action="store_true", help="关闭 DataParallel")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--filter_chunk_size", type=int, default=4096,
        help="超球筛选分块大小（调小可降低峰值内存，默认 4096）",
    )
    return p.parse_args()


# ─────────────────────────── 主函数 ──────────────────────────────


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

    # 每个 pool 的 3 份 txt 统一放到 results/{pool_name}/ 下
    out_abs = os.path.abspath(os.path.join(args.output_dir, pool_name))
    os.makedirs(out_abs, exist_ok=True)
    print(f"输出目录: {out_abs}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count() if device == "cuda" else 0
    print(
        f"[filter_pool_by_clusters] device={device}  GPUs={n_gpu}  "
        f"AMP={not args.no_amp}  DataParallel={not args.no_multi_gpu and n_gpu > 1}  "
        f"num_workers={args.num_workers}  embed_batch_size={args.embed_batch_size}",
        flush=True,
    )
    if device == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    # ── 加载聚类特征 ──────────────────────────────────────────────
    print("\n━━━ 加载 seed 聚类特征 ━━━", flush=True)
    seed_centers, seed_radii = load_all_cluster_features(
        os.path.join(args.feature_root, "seed")
    )

    print("\n━━━ 加载 downstream 聚类特征 ━━━", flush=True)
    ds_centers, ds_radii = load_all_cluster_features(
        os.path.join(args.feature_root, "downstream")
    )

    # ── 构建模型 ──────────────────────────────────────────────────
    print("\n━━━ 加载模型 ━━━", flush=True)
    ckpt = load_checkpoint(args.weight_path)
    img_size, _ = parse_img_size_from_ckpt(ckpt)
    model = build_model(img_size, ckpt, device, multi_gpu=not args.no_multi_gpu)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── 扫描 pool ─────────────────────────────────────────────────
    print(f"\n━━━ 扫描 pool: {args.pool_dir} ━━━", flush=True)
    pool_all_paths = collect_image_paths(args.pool_dir)
    print(f"  pool 图片总数: {len(pool_all_paths):,}", flush=True)
    if not pool_all_paths:
        print("[ERROR] 无图片，退出", flush=True)
        return

    # ── 提取 pool embedding ───────────────────────────────────────
    print("\n━━━ 提取 pool embedding ━━━", flush=True)
    ok_paths, ok_feat = extract_embeddings(
        pool_all_paths,
        transform,
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

    # ══════════════════════════════════════════════════════════════
    # Step 1: seed 超球筛选
    # ══════════════════════════════════════════════════════════════
    print(f"\n━━━ Step 1：seed 超球筛选  seed_mult={args.seed_mult} ━━━", flush=True)

    if seed_centers.shape[0] == 0:
        print("  [WARN] 无 seed 聚类特征，Step 1 跳过（保留全部）", flush=True)
        step1_mask = np.ones(len(ok_paths), dtype=bool)
    else:
        step1_mask = sphere_filter(
            ok_feat, seed_centers, seed_radii,
            multiplier=args.seed_mult,
            chunk_size=args.filter_chunk_size,
        )

    step1_pos = np.where(step1_mask)[0]
    step1_paths_raw = [ok_paths[int(i)] for i in step1_pos]

    # 路径去重（保序），同时对齐 embedding；避免递归扫描时同一文件重复出现
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
        + (f"  [去重 {n_dup:,} 条重复路径]" if n_dup else ""),
        flush=True,
    )

    step1_txt = os.path.join(out_abs, f"{pool_name}_{ts}_step1_seed_filtered.txt")
    write_txt(step1_txt, step1_paths)

    if not step1_paths:
        print("[WARN] Step 1 后无候选，流水线结束", flush=True)
        return

    # ══════════════════════════════════════════════════════════════
    # Step 2: KMeans 均衡采样（多样性保障）
    # ══════════════════════════════════════════════════════════════
    print(
        f"\n━━━ Step 2：KMeans 均衡采样  "
        f"kmeans_k={args.kmeans_k}  per_cluster_ratio={args.per_cluster_ratio} ━━━",
        flush=True,
    )
    step2_paths, step2_feat = kmeans_balanced_sample(
        step1_paths,
        step1_feat,
        k=args.kmeans_k,
        per_cluster_ratio=args.per_cluster_ratio,
        random_state=args.random_state,
    )
    print(f"  Step 2 保留: {len(step2_paths):,} / {len(step1_paths):,}", flush=True)

    step2_txt = os.path.join(out_abs, f"{pool_name}_{ts}_step2_kmeans_sampled.txt")
    write_txt(step2_txt, step2_paths)

    if not step2_paths:
        print("[WARN] Step 2 后无候选，流水线结束", flush=True)
        return

    # ══════════════════════════════════════════════════════════════
    # Step 3: downstream 超球去重
    # ══════════════════════════════════════════════════════════════
    print(f"\n━━━ Step 3：downstream 去重  ds_mult={args.ds_mult} ━━━", flush=True)

    if ds_centers.shape[0] == 0:
        print("  [WARN] 无 downstream 聚类特征，Step 3 跳过（保留全部）", flush=True)
        final_paths = step2_paths
    else:
        ds_mask = sphere_filter(
            step2_feat, ds_centers, ds_radii,
            multiplier=args.ds_mult,
            chunk_size=args.filter_chunk_size,
        )
        # 去掉落在 downstream 球内的图片
        final_paths = [step2_paths[i] for i in range(len(step2_paths)) if not ds_mask[i]]
        print(
            f"  Step 3 去除: {ds_mask.sum():,} 张  "
            f"保留: {len(final_paths):,} / {len(step2_paths):,}",
            flush=True,
        )

    # ── 写最终结果 ────────────────────────────────────────────────
    final_txt = os.path.join(out_abs, f"{pool_name}_{ts}_step3_final_dataset.txt")
    write_txt(final_txt, final_paths)

    # ── 汇总 ──────────────────────────────────────────────────────
    dup_note = f"  （去重 {n_dup:,} 条重复）" if n_dup else ""
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

