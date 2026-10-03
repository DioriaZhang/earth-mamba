from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math


Point = tuple[float, float]
Polygon = list[Point]


def order_polygon(polygon: Sequence[Point]) -> Polygon:
    """Return four vertices in cyclic order, starting near the top-left."""
    if len(polygon) != 4:
        return list(polygon)
    center_x = sum(point[0] for point in polygon) / 4.0
    center_y = sum(point[1] for point in polygon) / 4.0
    ordered = sorted(
        polygon, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x)
    )
    start = min(range(4), key=lambda index: ordered[index][0] + ordered[index][1])
    return ordered[start:] + ordered[:start]


def signed_area(polygon: Sequence[Point]) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1])
    )


def polygon_area(polygon: Sequence[Point]) -> float:
    return abs(signed_area(polygon))


def _line_intersection(start: Point, end: Point, a: Point, b: Point) -> Point:
    dx1, dy1 = end[0] - start[0], end[1] - start[1]
    dx2, dy2 = b[0] - a[0], b[1] - a[1]
    denominator = dx1 * dy2 - dy1 * dx2
    if abs(denominator) < 1e-12:
        return end
    t = ((a[0] - start[0]) * dy2 - (a[1] - start[1]) * dx2) / denominator
    return start[0] + t * dx1, start[1] + t * dy1


def convex_intersection(subject: Sequence[Point], clipper: Sequence[Point]) -> Polygon:
    output, clip = list(subject), list(clipper)
    if signed_area(clip) < 0:
        clip.reverse()
    for a, b in zip(clip, clip[1:] + clip[:1]):
        if not output:
            break
        source, output = output, []

        def inside(point: Point) -> bool:
            return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= -1e-8

        start = source[-1]
        for end in source:
            if inside(end):
                if not inside(start):
                    output.append(_line_intersection(start, end, a, b))
                output.append(end)
            elif inside(start):
                output.append(_line_intersection(start, end, a, b))
            start = end
    return output


def rotated_iou(first: Sequence[Point], second: Sequence[Point]) -> float:
    area_first, area_second = polygon_area(first), polygon_area(second)
    if area_first <= 0.0 or area_second <= 0.0:
        return 0.0
    intersection = polygon_area(convex_intersection(first, second))
    union = area_first + area_second - intersection
    return intersection / union if union > 0.0 else 0.0


def rotated_nms(polygons, scores, labels, iou_threshold: float, max_detections: int = 300):
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)
    kept: list[int] = []
    for indices in grouped.values():
        order = sorted(indices, key=lambda index: float(scores[index]), reverse=True)
        while order:
            current = order.pop(0)
            kept.append(current)
            order = [
                index for index in order
                if rotated_iou(polygons[current], polygons[index]) <= iou_threshold
            ]
    kept.sort(key=lambda index: float(scores[index]), reverse=True)
    return kept[:max_detections]


def _voc_ap(recalls: list[float], precisions: list[float]) -> float:
    recall = [0.0, *recalls, 1.0]
    precision = [0.0, *precisions, 0.0]
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    return sum(
        (recall[index] - recall[index - 1]) * precision[index]
        for index in range(1, len(recall)) if recall[index] != recall[index - 1]
    )


def mean_average_precision(predictions, targets, num_classes: int, iou_threshold: float = 0.5):
    class_ap: list[float | None] = []
    for class_index in range(num_classes):
        ground_truth: dict[int, list[dict]] = defaultdict(list)
        positive_count = 0
        for image_index, target in enumerate(targets):
            difficult_values = target.get("difficult", [False] * len(target["labels"]))
            for polygon, label, difficult in zip(target["polygons"], target["labels"], difficult_values):
                if int(label) == class_index:
                    ground_truth[image_index].append(
                        {
                            "polygon": order_polygon(polygon),
                            "difficult": bool(difficult),
                            "matched": False,
                        }
                    )
                    positive_count += int(not difficult)
        if positive_count == 0:
            class_ap.append(None)
            continue

        detections = []
        for image_index, prediction in enumerate(predictions):
            for polygon, score, label in zip(
                prediction["polygons"], prediction["scores"], prediction["labels"]
            ):
                if int(label) == class_index:
                    detections.append((float(score), image_index, polygon))
        detections.sort(reverse=True, key=lambda item: item[0])
        true_positive, false_positive = [], []
        for _, image_index, polygon in detections:
            candidates = ground_truth[image_index]
            ordered = order_polygon(polygon)
            ious = [rotated_iou(ordered, item["polygon"]) for item in candidates]
            best = max(range(len(ious)), key=ious.__getitem__) if ious else None
            if best is not None and ious[best] >= iou_threshold:
                match = candidates[best]
                if match["difficult"]:
                    continue
                if not match["matched"]:
                    match["matched"] = True
                    true_positive.append(1)
                    false_positive.append(0)
                else:
                    true_positive.append(0)
                    false_positive.append(1)
            else:
                true_positive.append(0)
                false_positive.append(1)

        cumulative_tp, cumulative_fp, recalls, precisions = 0, 0, [], []
        for tp, fp in zip(true_positive, false_positive):
            cumulative_tp += tp
            cumulative_fp += fp
            recalls.append(cumulative_tp / positive_count)
            precisions.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1))
        class_ap.append(_voc_ap(recalls, precisions))
    valid = [value for value in class_ap if value is not None]
    return {"mAP50": sum(valid) / len(valid) if valid else 0.0, "AP50": class_ap}
