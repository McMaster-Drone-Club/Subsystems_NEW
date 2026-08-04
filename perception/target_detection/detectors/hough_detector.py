"""Hough-circle target detector."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perception.target_detection.detectors.base import (
    BaseTargetDetector,
    Detection,
    DetectorResult,
    TimingInfo,
    bbox_iou,
    circle_bbox,
    load_yaml_config,
    validate_bgr_image,
)


class HoughCircleDetector(BaseTargetDetector):
    """OpenCV HoughCircles detector using one shared configuration."""

    method_name = "hough"

    @classmethod
    def from_config(cls, config_path: str | Path) -> "HoughCircleDetector":
        return cls(load_yaml_config(config_path))

    def detect(self, image: np.ndarray) -> DetectorResult:
        validate_bgr_image(image)
        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        processed = self._preprocess(image)
        preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000.0

        detector_start = time.perf_counter()
        circles = self._detect_circles(processed)
        detections = self._to_detections(circles, image.shape)
        detection_ms = (time.perf_counter() - detector_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return DetectorResult(
            method=self.method_name,
            detections=detections,
            timing=TimingInfo(preprocessing_ms, detection_ms, total_ms),
            image_shape=tuple(int(v) for v in image.shape),
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        preprocessing = self.config.get("preprocessing", {})
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if preprocessing.get("equalize_histogram", False):
            gray = cv2.equalizeHist(gray)

        blur = preprocessing.get("blur", {})
        blur_type = str(blur.get("type", "median")).lower()
        if blur_type == "none":
            return gray

        kernel_size = _odd_kernel_size(int(blur.get("kernel_size", 5)))
        if blur_type == "gaussian":
            sigma_x = float(blur.get("sigma_x", 0))
            return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma_x)
        if blur_type == "median":
            return cv2.medianBlur(gray, kernel_size)
        raise ValueError(f"Unsupported Hough blur type: {blur_type}")

    def _detect_circles(self, gray: np.ndarray) -> np.ndarray:
        hough = self.config.get("hough", {})
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=float(hough.get("dp", 1.2)),
            minDist=float(hough.get("min_dist", 40)),
            param1=float(hough.get("param1", 100)),
            param2=float(hough.get("param2", 24)),
            minRadius=int(hough.get("min_radius", 8)),
            maxRadius=int(hough.get("max_radius", 160)),
        )
        if circles is None:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(circles[0], dtype=np.float32)

    def _to_detections(
        self,
        circles: np.ndarray,
        image_shape: tuple[int, int, int],
    ) -> list[Detection]:
        height, width = image_shape[:2]
        raw = []
        for center_x, center_y, radius in circles:
            if radius <= 0:
                continue
            bbox = circle_bbox(float(center_x), float(center_y), float(radius), width, height)
            raw.append(
                Detection(
                    method=self.method_name,
                    center_x=float(center_x),
                    center_y=float(center_y),
                    radius=float(radius),
                    bbox=bbox,
                    score=None,
                )
            )

        return self._deduplicate(raw)

    def _deduplicate(self, detections: list[Detection]) -> list[Detection]:
        duplicate_config = self.config.get("duplicate_suppression", {})
        if not duplicate_config.get("enabled", True):
            return detections

        center_distance_factor = float(duplicate_config.get("center_distance_factor", 0.5))
        iou_threshold = float(duplicate_config.get("iou_threshold", 0.55))
        kept: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.radius or 0.0, reverse=True):
            is_duplicate = False
            for existing in kept:
                distance = math.hypot(
                    detection.center_x - existing.center_x,
                    detection.center_y - existing.center_y,
                )
                max_radius = max(detection.radius or 0.0, existing.radius or 0.0)
                if distance < max_radius * center_distance_factor:
                    is_duplicate = True
                    break
                if bbox_iou(detection.bbox, existing.bbox) >= iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(detection)
        return sorted(kept, key=lambda item: (item.center_y, item.center_x))


def _odd_kernel_size(value: int) -> int:
    if value < 1:
        raise ValueError("Blur kernel size must be positive.")
    return value if value % 2 == 1 else value + 1