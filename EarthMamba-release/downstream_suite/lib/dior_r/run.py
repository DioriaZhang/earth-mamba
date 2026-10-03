from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .backbones import SUPPORTED_BACKBONES, build_backbone
from .data import CLASSES, build_records, load_official_split, parse_annotation, resize_polygons, split_records
from .engine import load_trained_model, run_eval, seed_everything, train
from .model import FastOrientedDetector, decode_predictions, detection_loss


def add_common_arguments(parser):
    parser.add_argument("--data-dir", default="/hy-tmp/task/DIOR")
    parser.add_argument("--backbone", required=True, choices=SUPPORTED_BACKBONES)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--project-root", default="/hy-tmp")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssm-version", default="mamba3", choices=("mamba2", "mamba3"))
    parser.add_argument(
        "--earth-adapter",
        default="gated_pyramid",
        choices=("gated_pyramid", "detail_pyramid", "semantic_pyramid", "none"),
        help="earth-mamba only: dense feature adapter (ignored for other backbones)",
    )
    parser.add_argument(
        "--adapter-lr",
        type=float,
        default=2e-4,
        help="earth-mamba adapter + outnorm LR when --earth-adapter is not none",
    )
    parser.add_argument(
        "--debug-shapes",
        action="store_true",
        help="earth-mamba: print adapter output feature shapes once",
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Fast DIOR-R oriented detection benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="validate data, checkpoint and one forward pass")
    add_common_arguments(check)

    training = subparsers.add_parser("train", help="train and evaluate OBB mAP@0.5")
    add_common_arguments(training)
    training.add_argument("--epochs", type=int, default=36)
    training.add_argument("--batch-size", type=int, default=12)
    training.add_argument("--val-batch-size", type=int, default=16)
    training.add_argument("--workers", type=int, default=10)
    training.add_argument("--lr", type=float, default=2e-4)
    training.add_argument("--encoder-lr", type=float, default=2e-5)
    training.add_argument("--weight-decay", type=float, default=0.05)
    training.add_argument("--warmup-epochs", type=int, default=3)
    training.add_argument("--clip-grad", type=float, default=5.0)
    training.add_argument("--eval-interval", type=int, default=6)
    training.add_argument("--log-interval", type=int, default=50)
    training.add_argument("--score-threshold", type=float, default=0.05)
    training.add_argument("--nms-threshold", type=float, default=0.1)
    training.add_argument("--amp-dtype", choices=("fp16", "bf16", "none"), default="bf16")
    training.add_argument("--val-subset", type=int, default=0,
                          help="if >0, evaluate on at most this many val images (faster mid-run)")
    training.add_argument("--resume")
    training.add_argument("--work-dir", required=True)

    evaluation = subparsers.add_parser(
        "eval", help="evaluate a trained checkpoint (full val + optional threshold sweep)"
    )
    add_common_arguments(evaluation)
    evaluation.add_argument("--checkpoint", required=True, help="best.pt or last.pt from training")
    evaluation.add_argument("--val-batch-size", type=int, default=24)
    evaluation.add_argument("--workers", type=int, default=10)
    evaluation.add_argument("--score-threshold", type=float, default=0.05)
    evaluation.add_argument("--nms-threshold", type=float, default=0.1)
    evaluation.add_argument("--amp-dtype", choices=("fp16", "bf16", "none"), default="bf16")
    evaluation.add_argument("--val-subset", type=int, default=0,
                            help="0 = full val; >0 = first N val images only")
    evaluation.add_argument("--sweep-thresholds", action="store_true",
                            help="grid-search score/nms thresholds on val set")
    return parser


def _build_model(args):
    encoder, channels = build_backbone(
        args.backbone,
        args.ckpt,
        args.project_root,
        args.image_size,
        args.ssm_version,
        earth_adapter=getattr(args, "earth_adapter", "gated_pyramid"),
        debug_shapes=getattr(args, "debug_shapes", False),
    )
    model = FastOrientedDetector(encoder, channels, len(CLASSES))
    total = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"[model] backbone={args.backbone} channels={channels} total={total:.1f}M")
    return model


def check(args, train_records, val_records):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    annotation = parse_annotation(train_records[0].annotation_path)
    object_count = sum(len(parse_annotation(record.annotation_path).labels) for record in train_records)
    val_objects = sum(len(parse_annotation(record.annotation_path).labels) for record in val_records)
    print(
        f"[data] train_images={len(train_records)} train_objects={object_count} "
        f"val_images={len(val_records)} val_objects={val_objects}"
    )
    print(
        f"[xml] sample={train_records[0].annotation_path.name} "
        f"size={annotation.image_size} objects={len(annotation.labels)}"
    )
    model = _build_model(args).cuda().eval()
    image = torch.zeros((1, 3, args.image_size, args.image_size), device="cuda")
    scaled_polygons = resize_polygons(
        annotation.polygons,
        annotation.image_size,
        (args.image_size, args.image_size),
    )
    target = [{
        "polygons": scaled_polygons,
        "labels": annotation.labels,
        "difficult": annotation.difficult,
    }]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(image)
        losses = detection_loss(outputs, target, len(CLASSES))
    decoded = decode_predictions(outputs, args.image_size)
    shapes = [(tuple(item["logits"].shape), tuple(item["quads"].shape)) for item in outputs]
    print(f"[forward] output_shapes={shapes}")
    print(f"[targets] positives={int(losses['positives'])} loss_finite={bool(torch.isfinite(losses['loss']))}")
    print(f"[decode] detections={len(decoded[0]['scores'])}")
    print("[check] OK")


def main():
    args = build_parser().parse_args()
    if args.image_size % 32:
        raise ValueError("--image-size must be divisible by 32")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    records = build_records(args.data_dir)
    train_records, val_records = split_records(
        records, args.val_ratio, args.seed, data_dir=args.data_dir
    )
    official = load_official_split(args.data_dir)
    split_note = "official ImageSets/Main" if official is not None else f"random {1-args.val_ratio:.0%}/{args.val_ratio:.0%}"
    print(
        f"[DIOR-R] data={Path(args.data_dir).resolve()} labeled={len(records)} "
        f"train={len(train_records)} val={len(val_records)} image_size={args.image_size} "
        f"split={split_note}"
    )
    if args.command == "check":
        check(args, train_records, val_records)
    elif args.command == "eval":
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        print(f"[eval] checkpoint={checkpoint}", flush=True)
        model, state = load_trained_model(checkpoint, args)
        epoch = state.get("epoch", "?")
        best_map = state.get("best_map50", float("nan"))
        print(f"[eval] saved epoch={epoch} training_best_mAP50={best_map:.4f}", flush=True)
        run_eval(model, val_records, args)
    else:
        print(
            f"[protocol] epochs={args.epochs} batch={args.batch_size} "
            f"lr={args.lr:g} encoder_lr={args.encoder_lr:g} amp={args.amp_dtype} "
            f"metric=oriented_polygon_mAP@0.5"
        )
        if is_earth_mamba(args.backbone):
            print(
                f"[protocol] earth_adapter={args.earth_adapter} adapter_lr={args.adapter_lr:g}",
                flush=True,
            )
        model = _build_model(args)
        train(model, train_records, val_records, args)


if __name__ == "__main__":
    main()
