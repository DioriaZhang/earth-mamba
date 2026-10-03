"""
下游数据集 512 预处理脚本（preprocess_downstream.py）

功能：
    接受一个或多个下游数据集根目录，递归读取所有图片，
    将每张图统一处理为 512×512 PNG。

输出结构（便于分库复用、并行跑多个文件夹）：
    --output_root 指定根目录（默认：<当前工作目录>/Downstream-datasets）
    每个输入 DATASET_DIR 写入独立子文件夹：
        {output_root}/{数据集名}/  例：J:\\output\\Downstream-datasets\\LoveDA

    单独输入 J:\\datasets\\LoveDA 且 --output_root J:\\output\\Downstream-datasets
    → 输出到 J:\\output\\Downstream-datasets\\LoveDA

依赖（pip install）：
    Pillow>=9.0
    tqdm
    numpy

PIL 大图像素：默认不限制（遥感）；需上限时设环境变量 PIL_MAX_IMAGE_PIXELS=正整数。

文件名（在各自数据集子目录内平铺）：
    若含子目录：{sub1}-{sub2}--{stem}_{hash8}.png
    若在根目录下：{stem}_{hash8}.png
    （数据集名已在父目录中，文件名不再重复加 LoveDA- 前缀）

512 缩放与多波段 TIFF：同前版（crop_resize / stretch、open_as_rgb）。

用法示例（PowerShell）：
    # 单数据集
    python preprocess_downstream.py J:\\datasets\\LoveDA --output_root J:\\output\\Downstream-datasets

    # 多数据集（各自子目录；可开多终端并行跑不同数据集加快速度）
    python preprocess_downstream.py J:\\datasets\\LoveDA J:\\datasets\\DIOR \\
        --output_root J:\\output\\Downstream-datasets
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_pil_max = os.environ.get("PIL_MAX_IMAGE_PIXELS", "").strip().lower()
if _pil_max in ("", "0", "none", "unlimited"):
    Image.MAX_IMAGE_PIXELS = None
else:
    Image.MAX_IMAGE_PIXELS = int(_pil_max)
try:
    from PIL.Image import DecompressionBombWarning

    warnings.filterwarnings("ignore", category=DecompressionBombWarning)
except Exception:
    pass

# --------------------------------------------------
# 常量
# --------------------------------------------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
OUT_HW = (512, 512)
MAX_STEM_LEN = 60
MAX_SEG_LEN = 40
PROGRESS_EVERY = 50_000


# --------------------------------------------------
# 路径 / 命名工具
# --------------------------------------------------

def sanitize(s: str, maxlen: int = 80) -> str:
    """去除文件名非法字符，替换为 _，截断到 maxlen。"""
    s = re.sub(r"[^\w\-.]", "_", s).strip("._")
    return (s[:maxlen] if s else "x")


def dataset_folder_name(dataset_root: str) -> str:
    """输出子目录名 = 输入根目录 basename，经 sanitize。"""
    return sanitize(os.path.basename(os.path.normpath(dataset_root)) or "dataset")


def make_output_name(
    root: str,
    img_path: str,
    *,
    dataset_in_parent_dir: bool,
) -> str:
    """
    生成输出文件名（不含路径）。
    dataset_in_parent_dir=True：输出已在 {root名}/ 下，文件名不再加数据集前缀。
    """
    root = os.path.abspath(os.path.normpath(root))
    abs_img = os.path.abspath(os.path.normpath(img_path))

    try:
        rel = os.path.relpath(abs_img, root)
    except ValueError:
        rel = os.path.basename(abs_img)

    parts = Path(rel).parts
    sub_parts = parts[:-1]
    filename = parts[-1]
    stem = os.path.splitext(filename)[0]

    sub_segs = [sanitize(p, MAX_SEG_LEN) for p in sub_parts]
    stem_safe = sanitize(stem, MAX_STEM_LEN)
    h8 = hashlib.md5(abs_img.encode("utf-8", errors="replace")).hexdigest()[:8]

    if dataset_in_parent_dir:
        if sub_segs:
            return f"{'-'.join(sub_segs)}--{stem_safe}_{h8}.png"
        return f"{stem_safe}_{h8}.png"

    # 旧版平铺根目录：文件名里保留数据集名前缀（与某根目录合并时避免重名）
    dataset_name = sanitize(os.path.basename(root) or "dataset")
    prefix = (
        f"{dataset_name}-{'-'.join(sub_segs)}" if sub_segs else dataset_name
    )
    return f"{prefix}--{stem_safe}_{h8}.png"


# --------------------------------------------------
# 图片收集（递归）
# --------------------------------------------------

def collect_images(root: str) -> list[str]:
    root = os.path.abspath(os.path.normpath(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"目录不存在: {root}")

    out: list[str] = []
    n_all = 0
    print(f"  递归扫描中: {root}", flush=True)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            n_all += 1
            if fn.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(dirpath, fn))
            if n_all % PROGRESS_EVERY == 0:
                print(
                    f"    已遍历 {n_all:,} 个文件，找到图片 {len(out):,} 张 …",
                    flush=True,
                )
    print(f"  扫描完成: 共 {n_all:,} 个文件，图片 {len(out):,} 张", flush=True)
    return out


# --------------------------------------------------
# 图片读取（兼容多波段 TIFF）
# --------------------------------------------------

def open_as_rgb(src: str) -> Image.Image | None:
    try:
        img = Image.open(src)
        return img.convert("RGB")
    except Exception:
        pass

    try:
        img = Image.open(src)
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        else:
            return None

        if arr.dtype != np.uint8:
            vmin, vmax = float(arr.min()), float(arr.max())
            if vmax > vmin:
                arr = ((arr.astype(np.float32) - vmin) / (vmax - vmin) * 255).astype(
                    np.uint8
                )
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)

        return Image.fromarray(arr.astype(np.uint8), mode="RGB")
    except Exception:
        return None


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def resize_to_512(img: Image.Image, mode: str) -> Image.Image:
    w, h = img.size

    if mode == "stretch":
        if (w, h) == OUT_HW:
            return img
        return img.resize(OUT_HW, _resample())

    if (w, h) == OUT_HW:
        return img
    if w != h:
        img = center_crop_square(img)
    if img.size != OUT_HW:
        img = img.resize(OUT_HW, _resample())
    return img


def _resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def process_image(src: str, dst: str, mode: str) -> bool:
    img = open_as_rgb(src)
    if img is None:
        print(f"  [SKIP] 无法读取为 RGB: {src}", flush=True)
        return False

    try:
        out_img = resize_to_512(img, mode)
    except Exception as e:
        print(f"  [SKIP] resize 失败: {src}  ({e})", flush=True)
        return False

    if out_img is img and src.lower().endswith(".png"):
        try:
            shutil.copy2(src, dst)
            return True
        except Exception:
            pass

    try:
        out_img.save(dst, format="PNG", optimize=False)
        return True
    except Exception as e:
        print(f"  [SKIP] 保存失败: {dst}  ({e})", flush=True)
        return False


def process_one_dataset(
    root: str,
    output_root: str,
    mode: str,
    skip_existing: bool,
    dataset_in_parent_dir: bool,
    report: dict,
) -> None:
    """在 output_root 下创建 {数据集名}/ 子目录并写入 PNG。"""
    name = dataset_folder_name(root)
    output_dir = os.path.join(output_root, name)

    print(f"\n{'='*60}", flush=True)
    print(f"[{name}] 输入: {root}", flush=True)
    print(f"[{name}] 输出: {os.path.abspath(output_dir)}", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    paths = collect_images(root)
    if not paths:
        print(f"[{name}] 未找到任何图片，跳过。", flush=True)
        report[name] = {
            "total": 0,
            "ok": 0,
            "skip": 0,
            "err": 0,
            "output_dir": output_dir,
        }
        return

    ok = skip = err = 0
    for p in tqdm(paths, desc=name, unit="img", dynamic_ncols=True):
        out_name = make_output_name(
            root,
            p,
            dataset_in_parent_dir=dataset_in_parent_dir,
        )
        dst = os.path.join(output_dir, out_name)

        if skip_existing and os.path.isfile(dst):
            skip += 1
            continue

        if process_image(p, dst, mode):
            ok += 1
        else:
            err += 1

    report[name] = {
        "total": len(paths),
        "ok": ok,
        "skip": skip,
        "err": err,
        "output_dir": output_dir,
    }
    print(
        f"[{name}] 完成 ── ok={ok:,}  skip={skip:,}  err={err:,}  total={len(paths):,}",
        flush=True,
    )


def count_png_in_dir(directory: str) -> int:
    n = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(".png"):
                    n += 1
    except OSError:
        pass
    return n


def count_png_under_output_root(output_root: str) -> tuple[int, list[tuple[str, int]]]:
    """统计 output_root 下各一级子目录内 PNG 数量及总和。"""
    total = 0
    per_sub: list[tuple[str, int]] = []
    try:
        for name in sorted(os.listdir(output_root)):
            sub = os.path.join(output_root, name)
            if os.path.isdir(sub):
                c = count_png_in_dir(sub)
                per_sub.append((name, c))
                total += c
    except OSError:
        pass
    return total, per_sub


def parse_args() -> argparse.Namespace:
    default_root = os.path.join(os.getcwd(), "Downstream-datasets")

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "递归读取下游数据集，输出 512×512 PNG。\n"
            "每个输入目录对应 {output_root}/{数据集名}/，便于复用与并行处理。"
        ),
    )
    p.add_argument(
        "input_dirs",
        nargs="+",
        metavar="DATASET_DIR",
        help="一个或多个数据集根目录（递归读图；原数据不修改）",
    )
    p.add_argument(
        "--output_root",
        default=default_root,
        help=(
            "输出根目录（默认：<当前工作目录>/Downstream-datasets）。\n"
            "实际写入路径为：{output_root}/{每个输入目录的 basename}/"
        ),
    )
    p.add_argument(
        "--flat_filename",
        action="store_true",
        help=(
            "文件名仍带「数据集名-」前缀（旧版平铺风格）；\n"
            "默认关闭，因数据集名已在子目录中"
        ),
    )
    p.add_argument(
        "--resize_mode",
        choices=["crop_resize", "stretch"],
        default="crop_resize",
        help="crop_resize（默认）或 stretch",
    )
    p.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        default=False,
        help="强制覆盖已存在输出（默认跳过以支持断点续跑）",
    )
    p.set_defaults(skip_existing=True)
    return p.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args()
    output_root = os.path.abspath(os.path.normpath(args.output_root))
    os.makedirs(output_root, exist_ok=True)

    # 默认 True：输出在 {root}/{数据集名}/ 下，文件名不再重复加数据集前缀
    dataset_in_parent_dir = not args.flat_filename

    print("=" * 60, flush=True)
    print("[preprocess_downstream] 启动", flush=True)
    print(f"  输出根目录     : {output_root}", flush=True)
    print(f"  输入数据集数量 : {len(args.input_dirs)}", flush=True)
    for i, d in enumerate(args.input_dirs, 1):
        sub = dataset_folder_name(d)
        print(f"    {i}. {d}  →  {output_root}\\{sub}", flush=True)
    print(f"  缩放策略       : {args.resize_mode}", flush=True)
    print(f"  跳过已有文件   : {args.skip_existing}", flush=True)
    print(f"  文件名含数据集前缀（旧平铺风格）: {args.flat_filename}", flush=True)
    print("=" * 60, flush=True)

    report: dict[str, dict] = {}
    for d in args.input_dirs:
        try:
            process_one_dataset(
                root=d,
                output_root=output_root,
                mode=args.resize_mode,
                skip_existing=args.skip_existing,
                dataset_in_parent_dir=dataset_in_parent_dir,
                report=report,
            )
        except FileNotFoundError as e:
            print(f"[ERROR] {e}，已跳过该数据集", flush=True)
        except Exception as e:
            print(f"[ERROR] 处理 {d} 时发生未预期异常: {e}，已跳过", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("汇总", flush=True)
    print(f"  {'数据集':<24} {'total':>8} {'ok':>8} {'skip':>7} {'err':>6}", flush=True)
    print(f"  {'-'*58}", flush=True)
    for name, r in report.items():
        print(
            f"  {name:<24} {r['total']:>8,} {r['ok']:>8,} {r['skip']:>7,} {r['err']:>6,}",
            flush=True,
        )
        print(f"    └ {os.path.abspath(r['output_dir'])}", flush=True)

    grand_total, per_sub = count_png_under_output_root(output_root)
    print(f"\n  {output_root} 下 PNG 合计: {grand_total:,}", flush=True)
    for sub_name, c in per_sub:
        if c:
            print(f"    · {sub_name}: {c:,}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
