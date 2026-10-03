from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


CLASSES = (
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "expressway-service-area", "expressway-toll-station",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship", "stadium",
    "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


def normalize_class_name(name: str) -> str:
    value = name.strip().lower().replace("_", "-").replace(" ", "-")
    if value in CLASS_TO_INDEX:
        return value
    # Official DIOR XML mixes "baseballfield" and "baseball-field" spellings.
    compact = value.replace("-", "")
    if compact in CLASS_TO_INDEX:
        return compact
    return value


def resolve_obb_annotation_dir(root: Path) -> Path:
    """Return the OBB annotation folder under ``Annotations/``."""
    ann_root = root / "Annotations"
    if not ann_root.is_dir():
        raise FileNotFoundError(f"DIOR-R Annotations directory not found: {ann_root}")
    candidates = (
        "Oriented Bounding Boxes",  # official DIOR-R release
        "OrientedBoundingBoxes",
        "Oriented_Bounding_Boxes",
    )
    for name in candidates:
        directory = ann_root / name
        if directory.is_dir() and any(directory.glob("*.xml")):
            return directory
    raise FileNotFoundError(
        f"DIOR-R OBB annotation directory not found under {ann_root}; "
        f"tried: {', '.join(candidates)}"
    )


@dataclass(frozen=True)
class Annotation:
    filename: str
    image_size: tuple[int, int]
    polygons: torch.Tensor
    labels: torch.Tensor
    difficult: torch.Tensor


@dataclass(frozen=True)
class Record:
    image_path: Path
    annotation_path: Path


_POINT_FIELDS = (
    ("x_left_top", "y_left_top"),
    ("x_right_top", "y_right_top"),
    ("x_right_bottom", "y_right_bottom"),
    ("x_left_bottom", "y_left_bottom"),
)


@lru_cache(maxsize=32768)
def parse_annotation(path: str | Path) -> Annotation:
    path = Path(path)
    root = ET.parse(path).getroot()
    filename = (root.findtext("filename") or f"{path.stem}.jpg").strip()
    width = int(float(root.findtext("size/width", "800")))
    height = int(float(root.findtext("size/height", "800")))
    polygons: list[list[list[float]]] = []
    labels: list[int] = []
    difficult: list[bool] = []

    for obj in root.findall("object"):
        name = normalize_class_name(obj.findtext("name", ""))
        if name not in CLASS_TO_INDEX:
            raise ValueError(f"unknown DIOR class {name!r} in {path}")
        box = obj.find("robndbox")
        if box is None:
            continue
        polygon = []
        for x_name, y_name in _POINT_FIELDS:
            x_text, y_text = box.findtext(x_name), box.findtext(y_name)
            if x_text is None or y_text is None:
                raise ValueError(f"missing {x_name}/{y_name} in {path}")
            polygon.append([float(x_text), float(y_text)])
        polygons.append(polygon)
        labels.append(CLASS_TO_INDEX[name])
        difficult.append(bool(int(obj.findtext("difficult", "0"))))

    polygon_tensor = torch.tensor(polygons, dtype=torch.float32)
    if not polygons:
        polygon_tensor = torch.empty((0, 4, 2), dtype=torch.float32)
    return Annotation(
        filename=filename,
        image_size=(height, width),
        polygons=polygon_tensor,
        labels=torch.tensor(labels, dtype=torch.long),
        difficult=torch.tensor(difficult, dtype=torch.bool),
    )


def resize_polygons(
    polygons: torch.Tensor,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> torch.Tensor:
    old_h, old_w = old_size
    new_h, new_w = new_size
    scale = polygons.new_tensor((new_w / old_w, new_h / old_h))
    return polygons * scale


def _find_image_map(root: Path) -> dict[str, Path]:
    image_map: dict[str, Path] = {}
    preferred = [
        root / "JPEGImages-trainval",
        root / "JPEGImages-test",
        root / "JPEGImages",
        root / "images",
    ]
    directories = [path for path in preferred if path.is_dir()]
    if not directories:
        directories = [root]
    for directory in directories:
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
            for path in directory.rglob(extension):
                image_map.setdefault(path.stem, path)
    return image_map


def build_records(data_dir: str | Path) -> list[Record]:
    root = Path(data_dir).expanduser().resolve()
    annotation_dir = resolve_obb_annotation_dir(root)
    image_map = _find_image_map(root)
    official = load_official_split(root)
    allowed_stems: set[str] | None = None
    if official is not None:
        # Exclude official test IDs from training/eval pools.
        test_ids = set(_read_id_list(root / "ImageSets" / "Main" / "test.txt"))
        allowed_stems = {
            stem for stem in image_map
            if stem not in test_ids
        }
    records = [
        Record(image_map[path.stem], path)
        for path in sorted(annotation_dir.glob("*.xml"))
        if path.stem in image_map and (allowed_stems is None or path.stem in allowed_stems)
    ]
    if not records:
        raise FileNotFoundError(
            f"no matched DIOR-R images/XML under {root}; images={len(image_map)}"
        )
    return records


def _read_id_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_official_split(data_dir: str | Path) -> tuple[list[str], list[str]] | None:
    """Return (train_ids, val_ids) from ``ImageSets/Main`` when present."""
    split_dir = Path(data_dir).expanduser().resolve() / "ImageSets" / "Main"
    train_ids = _read_id_list(split_dir / "train.txt")
    val_ids = _read_id_list(split_dir / "val.txt")
    if train_ids and val_ids:
        return train_ids, val_ids
    return None


def split_records(
    records: list[Record],
    val_ratio: float = 0.15,
    seed: int = 42,
    data_dir: str | Path | None = None,
) -> tuple[list[Record], list[Record]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    by_stem = {record.image_path.stem: record for record in records}
    official = load_official_split(data_dir) if data_dir is not None else None
    if official is not None:
        train_ids, val_ids = official
        train_records = [by_stem[stem] for stem in train_ids if stem in by_stem]
        val_records = [by_stem[stem] for stem in val_ids if stem in by_stem]
        if train_records and val_records:
            return train_records, val_records
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


# ImageNet mean × 255 used as background fill so padding normalises near zero.
_IMAGENET_FILL = (123, 116, 103)


class DiorRDataset(Dataset):
    def __init__(self, records: list[Record], image_size: int, training: bool):
        self.records = records
        self.image_size = int(image_size)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.records)

    def _scale_crop(
        self,
        image: Image.Image,
        polygons: torch.Tensor,
        labels: torch.Tensor,
        difficult: torch.Tensor,
    ):
        """Random-scale + random-crop/pad to self.image_size × self.image_size.

        Scale factor is sampled from [0.5, 1.5].  For the downscale case the
        resized image is centre-pasted onto an ImageNet-mean canvas.  For the
        upscale case a random crop is taken.  Polygons whose centre ends up
        outside the output canvas are removed; the rest are translated.
        """
        T = self.image_size
        orig_h, orig_w = image.height, image.width
        scale = random.uniform(0.5, 1.5)
        new_h = max(1, int(round(orig_h * scale)))
        new_w = max(1, int(round(orig_w * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        polygons = resize_polygons(polygons, (orig_h, orig_w), (new_h, new_w))

        # Compute crop/pad offsets.
        if new_h >= T:
            top = random.randint(0, new_h - T)
            offset_y = -top
        else:
            top = 0
            offset_y = (T - new_h) // 2

        if new_w >= T:
            left = random.randint(0, new_w - T)
            offset_x = -left
        else:
            left = 0
            offset_x = (T - new_w) // 2

        crop_w = min(new_w, T)
        crop_h = min(new_h, T)
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        canvas = Image.new("RGB", (T, T), _IMAGENET_FILL)
        paste_x = max(0, offset_x)
        paste_y = max(0, offset_y)
        canvas.paste(cropped, (paste_x, paste_y))
        image = canvas

        # Update polygon coordinates.
        if len(polygons) > 0:
            polygons = polygons.clone()
            polygons[..., 0] += offset_x
            polygons[..., 1] += offset_y
            # Keep only objects whose centre lies inside the output image.
            centers = polygons.mean(dim=1)  # (N, 2)
            keep_mask = (
                (centers[:, 0] >= 0) & (centers[:, 0] < T) &
                (centers[:, 1] >= 0) & (centers[:, 1] < T)
            )
            polygons = polygons[keep_mask]
            labels = labels[keep_mask]
            difficult = difficult[keep_mask]

        return image, polygons, labels, difficult

    def __getitem__(self, index: int):
        record = self.records[index]
        annotation = parse_annotation(record.annotation_path)
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        orig_h, orig_w = image.height, image.width
        polygons = annotation.polygons.clone()
        labels = annotation.labels.clone()
        difficult = annotation.difficult.clone()

        if self.training:
            image, polygons, labels, difficult = self._scale_crop(
                image, polygons, labels, difficult
            )
            # Horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                w = image.width  # == self.image_size
                if len(polygons) > 0:
                    polygons = polygons.clone()
                    polygons[..., 0] = w - polygons[..., 0]
                    polygons = polygons[:, [1, 0, 3, 2]]
            # Vertical flip
            if random.random() < 0.5:
                image = TF.vflip(image)
                h = image.height  # == self.image_size
                if len(polygons) > 0:
                    polygons = polygons.clone()
                    polygons[..., 1] = h - polygons[..., 1]
                    polygons = polygons[:, [3, 2, 1, 0]]
            # Color jitter (each applied independently with 50 % probability)
            if random.random() < 0.8:
                image = TF.adjust_brightness(image, random.uniform(0.6, 1.4))
            if random.random() < 0.8:
                image = TF.adjust_contrast(image, random.uniform(0.6, 1.4))
            if random.random() < 0.8:
                image = TF.adjust_saturation(image, random.uniform(0.6, 1.4))
            if random.random() < 0.5:
                image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))
        else:
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            polygons = resize_polygons(polygons, (orig_h, orig_w), (self.image_size, self.image_size))

        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        target = {
            "polygons": polygons,
            "labels": labels,
            "difficult": difficult,
            "image_id": record.image_path.stem,
            "original_size": (orig_h, orig_w),
        }
        return tensor, target


def collate_batch(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
