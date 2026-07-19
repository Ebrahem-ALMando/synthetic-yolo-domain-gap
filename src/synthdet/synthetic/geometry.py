"""Bounding-box and foreground-mask geometry used by copy-paste generation."""

from __future__ import annotations

import math

import numpy as np


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("YOLO box dimensions must be positive")
    if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
        raise ValueError("YOLO box values must be normalized to [0, 1]")
    left = (x_center - width / 2) * image_width
    top = (y_center - height / 2) * image_height
    right = (x_center + width / 2) * image_width
    bottom = (y_center + height / 2) * image_height
    if left < -1e-6 or top < -1e-6 or right > image_width + 1e-6 or bottom > image_height + 1e-6:
        raise ValueError("YOLO box extends outside the image")
    x1 = max(0, math.floor(left))
    y1 = max(0, math.floor(top))
    x2 = min(image_width, math.ceil(right))
    y2 = min(image_height, math.ceil(bottom))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("YOLO box maps to an empty pixel rectangle")
    return x1, y1, x2, y2


def xyxy_to_yolo(
    box: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise ValueError("Pixel box must be positive and fully inside the image")
    return (
        ((x1 + x2) / 2) / image_width,
        ((y1 + y2) / 2) / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def foreground_mask_bbox(mask: np.ndarray, threshold: int = 16) -> tuple[int, int, int, int]:
    if mask.ndim != 2:
        raise ValueError("Foreground mask must be two-dimensional")
    ys, xs = np.where(mask > threshold)
    if not len(xs):
        raise ValueError("Foreground mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def intersection_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    intersection = intersection_area(first, second)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def existing_object_occlusion(
    candidate: tuple[int, int, int, int], existing: tuple[int, int, int, int]
) -> float:
    existing_area = (existing[2] - existing[0]) * (existing[3] - existing[1])
    return intersection_area(candidate, existing) / existing_area if existing_area else 1.0
