# Data Preparation

This document describes the data organization expected by EarthMamba. Raw datasets, labels, private manifests, LMDBs, tar shards, embeddings, and generated caches are not redistributed in this repository.

## Overview

EarthMamba uses:

1. approximately **3.67M unlabeled remote-sensing images** for SimMIM-style masked image modeling pretraining;
2. six public downstream benchmarks for classification, semantic segmentation, horizontal-box detection, and oriented-box detection.

The complete pretraining manifest is not included because it is large and may contain private absolute paths. Public users should obtain source datasets from their official providers and generate local manifests.

## Pretraining Inputs

The pretraining entry points are:

```text
train/pretrain_mae.py
train/pretrain_mae_base.py
```

Confirmed input modes:

| Input mode | Argument | Expected format |
|---|---|---|
| Text manifest | `--data_list` | one sample per line, tab-separated ID and image/file/folder path |
| LMDB | `--lmdb` | LMDB-backed image store |
| Tar shards | `--tar_dir` | directory containing tar shards and `tar_index.jsonl` |

Example text manifest:

```text
sample_000001	/path/to/public_dataset/images/image_000001.jpg
sample_000002	/path/to/public_dataset/images/image_000002.jpg
scene_000003	/path/to/public_dataset/scene_folder_000003
```

See [examples/train_list.example.txt](examples/train_list.example.txt). The real training list is not public.

## Data Processing Scripts

| Script | Purpose |
|---|---|
| `data_process/preprocess_downstream.py` | preprocess downstream datasets into a normalized image organization |
| `data_process/pipeline_cluster_1_build.py` | build seed and downstream feature clusters |
| `data_process/pipeline_cluster_2_filter.py` | filter candidate pool images using feature-space criteria |
| `data_process/pipeline_cluster_2_filter_fix.py` | retained alternative filtering variant |
| `data_process/pipeline_cluster_2_filter_v2.py` | retained alternative GPU-oriented filtering variant |
| `data_process/pipeline_cluster_dataset.py` | dataset helpers for feature extraction |
| `data_process/Data_Preprocessing_Pipeline.md` | detailed pipeline notes |

Multiple Step-2 filtering variants exist. Treat `pipeline_cluster_2_filter.py` as the documented main script unless an experiment record explicitly selects a variant.

## Public Dataset Placeholders

Fill official acquisition links and redistribution terms before non-anonymous public release.

| Dataset/source | Used for | Official acquisition | Redistribution status |
|---|---|---|---|
| Pretraining mixture | masked image modeling | `TODO_OFFICIAL_URLS` | `TODO_VERIFY` |
| DFC15 | multi-label classification | `TODO_OFFICIAL_URL` | `TODO_VERIFY` |
| PatternNet | scene classification | `TODO_OFFICIAL_URL` | `TODO_VERIFY` |
| INRIA Aerial Image Labeling | building segmentation | `TODO_OFFICIAL_URL` | `TODO_VERIFY` |
| DeepGlobe Land Cover | land-cover segmentation | `TODO_OFFICIAL_URL` | `TODO_VERIFY` |
| DIOR | horizontal-box detection | `TODO_OFFICIAL_URL` | `TODO_VERIFY` |
| DIOR-R | oriented-box detection | `TODO_OFFICIAL_URL` | `TODO_VERIFY` |

## Downstream Layouts

The downstream suite resolves these layouts from code:

| Task | Entry script | Expected local layout |
|---|---|---|
| DFC15 | `downstream_suite/run_dfc15.py` | either `multilabel.csv` with `images_tr/` and `images_test/`, or split folders with image and label text files |
| PatternNet | `downstream_suite/run_patternnet.py` | `train/<class>/*` and `val/<class>/*`, or `images/<class>/*`, or root-level `<class>/*` folders |
| INRIA | `downstream_suite/run_inria.py` | `train/images/*.tif` and `train/gt/*.tif` under the dataset root or a parent wrapper directory |
| DeepGlobe | `downstream_suite/run_deepglobe.py` | `train/*_sat.jpg` paired with `train/*_mask.png`; optional `valid/` split with the same naming |
| DIOR HBB | `downstream_suite/run_dior.py` | `Annotations/` plus `JPEGImages-trainval/`, `JPEGImages-test/`, `JPEGImages/`, or `images/` |
| DIOR-R OBB | `downstream_suite/run_dior_r.py` | oriented annotations under `Annotations/`, image folders as above, optional `ImageSets/Main/train.txt` and `val.txt` |

Dataset preparation files should contain only local user paths and should not be committed.
