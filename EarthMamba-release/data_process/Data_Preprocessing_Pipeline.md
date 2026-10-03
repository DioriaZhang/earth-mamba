# Remote Sensing Pre-training Data Preprocessing Pipeline


> **Scope of this document.** The datasets, commands, and statistics below are **illustrative excerpts** from our full curation effort. They are included to make the pipeline reproducible and auditable, not to enumerate every source or every batch we processed. In practice, we ran the same three-stage pipeline **repeatedly across many more Pool batches, disks, and hyperparameter settings** (storage was processed incrementally: preprocess → filter → package → upload). The final pre-training corpus is at the **~4M-image scale**; the tables here report only a **representative subset** of completed runs.

---

## 1. Objectives and Design Principles

The pre-training corpus is constructed to satisfy four requirements:

1. **Scale and diversity.** Images span multiple sensors, geographic regions, and land-cover semantics, with a target scale of at least ~4M images (built and filtered in batches).
2. **Downstream isolation.** Near-duplicates between pre-training samples and downstream benchmark images are suppressed in feature space to reduce data leakage.
3. **Unified on-disk format.** All images are stored as \(512 \times 512\) RGB PNG files for efficient caching and I/O; mild resolution jitter is applied online during training.
4. **Minimal augmentation, semantic curation.** Following the DINO-style philosophy, we prioritize distribution coverage and deduplication over heavy data augmentation.

### Data roles

| Role | Description | Representative sources |
|------|-------------|------------------------|
| **Seed** | Target semantic support set (distribution anchors) | Globe230k, LuoJia-CDKD |
| **Downstream** | Benchmark-related semantics to avoid | PatternNet, DIOR, DOTA, LoveDA, etc. |
| **Pool** | Large uncurated candidate pool | OpenSatMap, IRSAMap, 8KDehaze, regional imagery, *and additional sources processed in later batches* |

---

## 2. Pipeline Overview

```
Raw remote sensing imagery
    │
    ▼
[A] preprocess_downstream.py
    Center-crop / resize → 512×512 PNG
    │
    ├── On-disk layout: Seed / Downstream / Pool
    │
    ▼
[B] pipeline_cluster_1_build.py
    SatMAE/ViT embeddings (L2-normalized) → KMeans → cluster centers + covering radii
    Output: feature/{seed|downstream}/*/clusters.npz
    │
    ▼
[C] pipeline_cluster_2_filter.py
    Three-stage Pool filtering: Seed hypersphere retention → balanced KMeans sampling → Downstream deduplication
    Output: results/{pool_name}/*_step{1,2,3}_*.txt
```

| Script | Purpose |
|--------|---------|
| `preprocess_downstream.py` | Recursively read images and normalize to \(512 \times 512\) PNG |
| `pipeline_cluster_dataset.py` | Multiprocessing-safe embedding DataLoader |
| `pipeline_cluster_1_build.py` | Build Seed / Downstream cluster hyperspheres |
| `pipeline_cluster_2_filter.py` | Three-stage Pool filtering (main pipeline) |
| `pipeline_cluster_2_filter_v2.py` / `_fix.py` | Throughput-optimized variants (same logic) |

**Feature encoder.** SatMAE pre-trained ViT-Base (`vit_base_patch16_224`, checkpoint: `satmae-pretrain-vit-base-e199.pth`). After L2 normalization, Euclidean distance is monotonically equivalent to cosine distance:

\[
\|a-b\|_2^2 = 2 - 2\,a^\top b,\quad \|a\|_2=\|b\|_2=1.
\]

---

## 3. Stage A: Resolution Normalization to \(512 \times 512\)

**Script:** `preprocess_downstream.py`

### 3.1 Processing rules

- **Supported formats:** `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.webp`
- **Multi-band / high bit-depth TIFF:** Use the first three bands (or replicate grayscale to RGB) and linearly rescale to 8-bit
- **Default resize mode (`crop_resize`):**
  - Non-square images: center-crop to a square
  - LANCZOS resize to \(512 \times 512\)
- **Alternative (`stretch`):** Directly resize to \(512 \times 512\) (may alter aspect ratio)
- **Output:** PNG; filenames encode relative path segments and an 8-character MD5 hash of the absolute path to avoid collisions across subdirectories

### 3.2 Dataset partitions

**Downstream benchmarks** (used only for feature construction and deduplication; **not** included as pre-training positives)

| Dataset | Task | Normalized output (example) |
|---------|------|----------------------------|
| PatternNet | Scene classification | `Downstream-datasets/PatternNet` |
| DFC15 | Multi-label classification | `Downstream-datasets/DFC15` |
| DIOR | Horizontal bounding-box detection | `Downstream-datasets/DIOR` |
| DOTAv1.0 | Oriented object detection | `Downstream-datasets/DOTAv1.0` |
| INRIA | Building segmentation | `Downstream-datasets/INRIA` |
| LoveDA | Semantic segmentation / domain adaptation | `Downstream-datasets/LoveDA` |
| LEVIR | Change detection | `Downstream-datasets/LEVIR` |
| SECOND | Change detection | `Downstream-datasets/second_dataset` |

After normalization, the Downstream split contains **92,281** images in total.

**Seed set (semantic anchors)**

| Dataset | Description | Valid embeddings |
|---------|-------------|------------------|
| Globe230k | Global high-resolution land-cover imagery (~1 m/px) | 232,812 |
| LuoJia-CDKD | Change-detection / knowledge-distillation subset | 83,789 |
| **Total** | | **316,601** |

**Pool (candidate sources, filtered in batches)**

The following are **representative** Pool sources. Additional regional and web-scale collections were normalized and filtered using the same scripts; only a subset is listed here for brevity.

| Dataset | Description |
|---------|-------------|
| OpenSatMap | High-resolution road / scene imagery |
| IRSAMap_v2 | High spatial-resolution map imagery |
| 8KDehaze | Dehazing-related samples |
| GD_dataset | Regional imagery |
| images_v1 / images_v2 | Multi-city / large-area imagery (stored across disks) |
| *(others)* | Further Pool batches processed off-document due to space and disk sharding |

**Example command:**

```bash
python preprocess_downstream.py <DATASET_DIR_1> <DATASET_DIR_2> \
  --output_root <OUT_ROOT>/Downstream-datasets
```

---

## 4. Stage B: Seed / Downstream Cluster Feature Construction

**Script:** `pipeline_cluster_1_build.py`

Each dataset subdirectory is treated as an independent distribution:

1. Scan all images; extract ViT embeddings and L2-normalize
2. Run KMeans (default \(K=50\); automatically reduced if insufficient samples)
3. Re-normalize cluster centers to unit length
4. Compute the minimum covering radius per cluster:
   \[
   r_k = \max_{i \in C_k} \| f_i - c_k \|_2
   \]
5. Save `clusters.npz` with fields `centers`, `radii`, and `n_images`

**Output layout:**

```
feature/
├── seed/
│   ├── Globe230k/clusters.npz
│   └── LuoJia-CDKD/clusters.npz
└── downstream/
    ├── DFC15/clusters.npz
    ├── DIOR/clusters.npz
    └── ...
```

### 4.1 Example run and statistics *(partial demonstration)*

The cluster statistics below are from **one completed feature-build run** on the Seed and Downstream splits documented above. The same procedure was applied consistently; we do not reproduce every intermediate log in this supplement.

```bash
python pipeline_cluster_1_build.py \
  --globe230k <GLOBE_DIR> \
  --luojia <LUOJIA_DIR> \
  --downstream_dirs <DOWNSTREAM_ROOT> \
  --feature_root <FEATURE_ROOT> \
  --n_clusters 50
```

**Seed cluster radius statistics**

| Dataset | #Images | K | Radius (min / mean / max) |
|---------|---------|---|---------------------------|
| Globe230k | 232,812 | 50 | 0.1554 / 0.2836 / 0.6947 |
| LuoJia-CDKD | 83,789 | 50 | 0.1401 / 0.2724 / 0.5898 |

**Downstream cluster radius statistics**

| Dataset | #Images | K | Radius (min / mean / max) |
|---------|---------|---|---------------------------|
| DFC15 | 3,342 | 50 | 0.0903 / 0.1256 / 0.2625 |
| DIOR | 23,463 | 50 | 0.1028 / 0.1918 / 0.4277 |
| DOTAv1.0 | 3,799 | 50 | 0.0000 / 0.1204 / 0.2289 |
| INRIA | 540 | 50 | 0.0264 / 0.0644 / 0.1110 |
| LEVIR | 1,911 | 50 | 0.0456 / 0.1013 / 0.2359 |
| LoveDA | 10,178 | 50 | 0.0724 / 0.1138 / 0.1986 |
| PatternNet | 30,400 | 50 | 0.0961 / 0.1724 / 0.2808 |
| second_dataset | 18,648 | 50 | 0.0762 / 0.1442 / 0.2917 |
| **Total** | **92,281** | **400 clusters** | — |

In aggregate: **100** Seed clusters covering **316,601** images; **400** Downstream clusters covering **92,281** images.

---

## 5. Stage C: Three-Stage Pool Filtering

**Script:** `pipeline_cluster_2_filter.py`

For each Pool directory, embeddings are extracted and filtered in three stages:

### Step 1 — Seed hypersphere retention (high recall)

A Pool image is **kept** if it falls inside **any** Seed cluster hypersphere:

\[
\exists\, k:\quad \|f - c_k^{\text{seed}}\|_2 \le r_k^{\text{seed}} \cdot \texttt{seed\_mult}
\]

- Recommended `seed_mult ∈ [1.5, 2.0]` (larger values are more permissive)
- Output: `*_step1_seed_filtered.txt`

### Step 2 — Balanced KMeans sampling (diversity)

Candidates from Step 1 are re-clustered (default \(K=200\)). Within each cluster, `per_cluster_ratio` (default 0.5) of images are randomly sampled to prevent a single semantic mode from dominating.

- Output: `*_step2_kmeans_sampled.txt`

### Step 3 — Downstream hypersphere deduplication (benchmark isolation)

A remaining image is **removed** if it falls inside **any** Downstream cluster hypersphere:

\[
\exists\, k:\quad \|f - c_k^{\text{ds}}\|_2 \le r_k^{\text{ds}} \cdot \texttt{ds\_mult}
\]

- Recommended `ds_mult ∈ [0.5, 0.8]` (smaller values enforce stricter deduplication)
- Output: `*_step3_final_dataset.txt` (final list of image paths)

**Example command:**

```bash
python pipeline_cluster_2_filter.py \
  --pool_dir <POOL_SUBDIR> \
  --feature_root <FEATURE_ROOT> \
  --seed_mult 2.0 --ds_mult 0.8 \
  --kmeans_k 200 --per_cluster_ratio 0.5
```

### 5.1 Per-Pool filtering results *(partial demonstration)*

Table 5.1 reports **five representative Pool runs** with full step-by-step counts. Many additional Pool subdirectories were filtered with the same pipeline (often varying `seed_mult`, `ds_mult`, and batch boundaries). Those runs are omitted here to keep the supplement readable.

| Pool | Scanned | Step 1 (Seed) | Step 2 (KMeans) | Step 3 (Final) | Key hyperparameters |
|------|---------|---------------|-----------------|----------------|---------------------|
| opensatmap | 16,778 | 16,778 (100%) | 8,386 | **2,934** | seed_mult=2.0, ds_mult=0.8 |
| IRSAMap_v2 | 20,868 | 20,868 (100%) | 10,431 | **10,283** | seed_mult=2.0, ds_mult=0.6 |
| 8KDehaze | 13,535 | 13,535 (100%) | 6,763 | **6,732** | seed_mult=2.0, ds_mult=0.6 |
| GD_dataset | 32,488 | 32,488 (100%) | 16,247 | **16,247** | seed_mult=2.0, ds_mult=0.6 |
| images_v2 | — | — | — | **90,698** | Aggregated after batch filtering |

**Notes.** Some Pools retain 100% of images at Step 1 under `seed_mult=2.0`, indicating that candidates already lie within the Seed semantic neighborhood. Subsequent balanced sampling and Downstream deduplication still reduce corpus size and suppress benchmark-adjacent samples (e.g., opensatmap removes 5,452 images at Step 3). **These rows are not exhaustive:** they exemplify typical retention patterns; other batches showed similar or stricter filtering depending on source semantics and `ds_mult`.

### 5.2 Pre-training corpus composition *(example batch summary, not the full corpus)*

The table below sums **only the Seed set plus the five Pool batches in §5.1**. It is meant to show how filtered batches compose into a training list, **not** to claim that pre-training used only these sources.

| Source | #Images |
|--------|---------|
| Seed: Globe230k + LuoJia-CDKD | 316,601 |
| Pool: opensatmap | 2,934 |
| Pool: IRSAMap_v2 | 10,283 |
| Pool: 8KDehaze | 6,732 |
| Pool: GD_dataset | 16,247 |
| Pool: images_v2 | 90,698 |
| **Subtotal (documented example batches only)** | **≈ 443,495** |

> **Full corpus vs. this document.** Pre-training ultimately draws from **many more filtered Pool batches** merged with the Seed set, reaching the **~4M-image** target. Building that corpus required extensive offline work—recursive normalization of heterogeneous sources, multi-GPU embedding passes, repeated three-stage filtering, and cross-disk packaging—that cannot be fully listed in a short supplement. The statistics above are **faithful records of real runs** on a **deliberately selected subset**; the same scripts and hyperparameters were applied at scale beyond what is tabulated here.

---

## 6. Connection to Pre-training Data Loading

All images are stored at \(512 \times 512\) on disk. During SimMIM pre-training, `MultiScaleCollator` samples target resolutions at the batch level:

- **~85%** probability: keep 512
- **~15%** probability: sample a size within \(\pm 15\%\) of 512, rounded to a multiple of 16
- Apply horizontal/vertical flip, color jitter, and ImageNet normalization; stack to `(B, 3, H, W)`

The backbone uses `posembed=False` by default; the classification head employs `AdaptiveAvgPool2d(1)`; and the SSM branch flattens features according to the current spatial size. This enables variable-resolution training and inference without locking to a single resolution.

---

## 7. Minimal Reproduction Checklist

```bash
# 1) Normalize to 512×512
python preprocess_downstream.py <DATASET_DIRS...> --output_root <OUT_ROOT>

# 2) Build Seed / Downstream cluster hyperspheres
python pipeline_cluster_1_build.py \
  --globe230k <GLOBE_DIR> --luojia <LUOJIA_DIR> \
  --downstream_dirs <DOWNSTREAM_ROOT> \
  --feature_root <FEATURE_ROOT> \
  --n_clusters 50

# 3) Filter each Pool subdirectory
python pipeline_cluster_2_filter.py \
  --pool_dir <POOL_SUBDIR> \
  --feature_root <FEATURE_ROOT> \
  --seed_mult 2.0 --ds_mult 0.6 \
  --kmeans_k 200 --per_cluster_ratio 0.5
```

**Dependencies:** `torch`, `timm`, `scikit-learn`, `Pillow`, `numpy`, `tqdm`. GPU is optional (DataParallel and AMP supported).

---

## 8. Artifact Index

| Artifact | Description |
|----------|-------------|
| `Downstream-datasets/{name}/*.png` | Normalized 512×512 Downstream images |
| `feature/seed/*/clusters.npz` | Seed cluster centers and radii |
| `feature/downstream/*/clusters.npz` | Downstream cluster centers and radii |
| `results/{pool}/*_step1_*.txt` | Paths after Seed filtering |
| `results/{pool}/*_step2_*.txt` | Paths after balanced sampling |
| `results/{pool}/*_step3_final_*.txt` | Final pre-training image path list |

---

*This document matches the scripts in `data_process/`. Reported numbers are taken from actual pipeline logs on **representative batches**; they may be cited in the paper's methodology or supplementary material. The **complete** curation covered substantially more data and batches than the examples shown.*
