"""Shared detector interface and data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One detected target.

    bbox uses (x, y, width, height) in pixel coordinates.
    """

    method: str
    center_x: float
    center_y: float
    radius: float | None
    bbox: BBox
    score: float | None = None
    ellipse_axes: tuple[float, float] | None = None
    ellipse_angle: float | None = None

    def as_dict(self) -> dict[str, float | str | None | BBox]:
        return {
            "method": self.method,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "bbox": self.bbox,
            "score": self.score,
            "ellipse_axes": self.ellipse_axes,
            "ellipse_angle": self.ellipse_angle,
        }


@dataclass(frozen=True)
class TimingInfo:
    """Detector latency in milliseconds.

    Image loading and output writing are intentionally excluded.
    """

    preprocessing_ms: float
    detection_ms: float
    total_ms: float


@dataclass(frozen=True)
class DetectorResult:
    """Complete detector output for one already-loaded BGR image."""

    method: str
    detections: list[Detection] = field(default_factory=list)
    timing: TimingInfo = field(default_factory=lambda: TimingInfo(0.0, 0.0, 0.0))
    image_shape: tuple[int, int, int] | None = None


class BaseTargetDetector(ABC):
    """Reusable detector interface.

    Callers own image loading, annotation evaluation, and output writing. The detector
    accepts one OpenCV BGR image and returns zero or more detections plus timing.
    """

    method_name: str

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @classmethod
    def from_config(cls, config_path: str | Path) -> "BaseTargetDetector":
        return cls(load_yaml_config(config_path))

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectorResult:
        """Detect targets in one OpenCV BGR image."""


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return data


def validate_bgr_image(image: np.ndarray) -> None:
    if image is None:
        raise ValueError("Image is None; expected an OpenCV BGR image.")
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if image.size == 0:
        raise ValueError("Image is empty.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have shape (height, width, 3) in BGR order.")
    if image.dtype != np.uint8:
        raise ValueError("Image dtype must be uint8.")


def circle_bbox(
    center_x: float,
    center_y: float,
    radius: float,
    image_width: int,
    image_height: int,
) -> BBox:
    x1 = max(0.0, center_x - radius)
    y1 = max(0.0, center_y - radius)
    x2 = min(float(image_width), center_x + radius)
    y2 = min(float(image_height), center_y + radius)
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def bbox_iou(first: BBox, second: BBox) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union
