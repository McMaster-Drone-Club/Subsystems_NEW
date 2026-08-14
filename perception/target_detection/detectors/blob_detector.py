"""SimpleBlobDetector target detector."""

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


class BlobDetector(BaseTargetDetector):
    """OpenCV SimpleBlobDetector wrapper using one shared configuration.

    OpenCV reports keypoint.size as blob diameter, so radius is size / 2.
    """

    method_name = "blob"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._detector = cv2.SimpleBlobDetector_create(self._build_params("blob"))
        small_config = self.config.get("small_blob", {})
        self._small_detector = (
            cv2.SimpleBlobDetector_create(self._build_params("small_blob"))
            if small_config.get("enabled", False)
            else None
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "BlobDetector":
        return cls(load_yaml_config(config_path))

    def detect(self, image: np.ndarray) -> DetectorResult:
        validate_bgr_image(image)
        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        processed = self._preprocess(image)
        edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000.0

        detector_start = time.perf_counter()
        keypoints = list(self._detector.detect(processed))
        if self._small_detector is not None:
            keypoints.extend(self._small_detector.detect(processed))
        detections = self._to_detections(keypoints, image.shape)
        detections = self._filter_by_boundary_edge(detections, edge_image)
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
        blur_type = str(blur.get("type", "gaussian")).lower()
        if blur_type == "none":
            return gray

        kernel_size = _odd_kernel_size(int(blur.get("kernel_size", 3)))
        if blur_type == "gaussian":
            sigma_x = float(blur.get("sigma_x", 0))
            return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma_x)
        if blur_type == "median":
            return cv2.medianBlur(gray, kernel_size)
        raise ValueError(f"Unsupported blob blur type: {blur_type}")

    def _to_detections(
        self,
        keypoints: tuple[cv2.KeyPoint, ...],
        image_shape: tuple[int, int, int],
    ) -> list[Detection]:
        height, width = image_shape[:2]
        detections: list[Detection] = []
        for keypoint in keypoints:
            center_x, center_y = keypoint.pt
            radius = keypoint.size / 2.0
            bbox = circle_bbox(float(center_x), float(center_y), float(radius), width, height)
            response = float(keypoint.response) if keypoint.response else None
            detections.append(
                Detection(
                    method=self.method_name,
                    center_x=float(center_x),
                    center_y=float(center_y),
                    radius=float(radius),
                    bbox=bbox,
                    score=response,
                )
            )
        return self._deduplicate(detections)

    def _filter_by_boundary_edge(
        self,
        detections: list[Detection],
        gray: np.ndarray,
    ) -> list[Detection]:
        edge_config = self.config.get("boundary_edge_filter", {})
        if not edge_config.get("enabled", False):
            return detections

        min_mean_gradient = float(edge_config.get("min_mean_gradient", 40.0))
        inner_factor = float(edge_config.get("inner_radius_factor", 0.8))
        outer_factor = float(edge_config.get("outer_radius_factor", 1.2))
        gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        kept: list[Detection] = []
        height, width = gray.shape
        for detection in detections:
            radius = detection.radius or 0.0
            outer_radius = max(1, int(math.ceil(radius * outer_factor)))
            x1 = max(0, int(math.floor(detection.center_x - outer_radius)))
            y1 = max(0, int(math.floor(detection.center_y - outer_radius)))
            x2 = min(width, int(math.ceil(detection.center_x + outer_radius + 1)))
            y2 = min(height, int(math.ceil(detection.center_y + outer_radius + 1)))
            yy, xx = np.ogrid[y1:y2, x1:x2]
            distance = np.sqrt(
                (xx - detection.center_x) ** 2 + (yy - detection.center_y) ** 2
            )
            ring = (distance >= radius * inner_factor) & (distance <= radius * outer_factor)
            values = gradient[y1:y2, x1:x2][ring]
            if values.size and float(np.mean(values)) >= min_mean_gradient:
                kept.append(detection)
        return kept

    def _deduplicate(self, detections: list[Detection]) -> list[Detection]:
        duplicate_config = self.config.get("duplicate_suppression", {})
        center_factor = float(duplicate_config.get("center_distance_factor", 0.5))
        iou_threshold = float(duplicate_config.get("iou_threshold", 0.5))
        kept: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.radius or 0.0, reverse=True):
            if any(
                math.hypot(
                    detection.center_x - existing.center_x,
                    detection.center_y - existing.center_y,
                ) < max(detection.radius or 0.0, existing.radius or 0.0) * center_factor
                or bbox_iou(detection.bbox, existing.bbox) >= iou_threshold
                for existing in kept
            ):
                continue
            kept.append(detection)
        return sorted(kept, key=lambda item: (item.center_y, item.center_x))

    def _build_params(self, section: str) -> cv2.SimpleBlobDetector_Params:
        params_config = self.config.get(section, {})
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = float(params_config.get("min_threshold", 10))
        params.maxThreshold = float(params_config.get("max_threshold", 240))
        params.thresholdStep = float(params_config.get("threshold_step", 10))
        params.minDistBetweenBlobs = float(params_config.get("min_dist_between_blobs", 10))

        params.filterByColor = bool(params_config.get("filter_by_color", True))
        params.blobColor = int(params_config.get("blob_color", 255))

        params.filterByArea = bool(params_config.get("filter_by_area", True))
        params.minArea = float(params_config.get("min_area", 40))
        params.maxArea = float(params_config.get("max_area", 30000))

        params.filterByCircularity = bool(params_config.get("filter_by_circularity", True))
        params.minCircularity = float(params_config.get("min_circularity", 0.55))
        params.maxCircularity = float(params_config.get("max_circularity", 1.0))

        params.filterByConvexity = bool(params_config.get("filter_by_convexity", False))
        params.minConvexity = float(params_config.get("min_convexity", 0.75))
        params.maxConvexity = float(params_config.get("max_convexity", 1.0))

        params.filterByInertia = bool(params_config.get("filter_by_inertia", False))
        params.minInertiaRatio = float(params_config.get("min_inertia_ratio", 0.1))
        params.maxInertiaRatio = float(params_config.get("max_inertia_ratio", 1.0))
        return params


def _odd_kernel_size(value: int) -> int:
    if value < 1:
        raise ValueError("Blur kernel size must be positive.")
    return value if value % 2 == 1 else value + 1
