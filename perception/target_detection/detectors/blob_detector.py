"""SimpleBlobDetector target detector."""

from __future__ import annotations

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
        self._detector = cv2.SimpleBlobDetector_create(self._build_params())

    @classmethod
    def from_config(cls, config_path: str | Path) -> "BlobDetector":
        return cls(load_yaml_config(config_path))

    def detect(self, image: np.ndarray) -> DetectorResult:
        validate_bgr_image(image)
        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        processed = self._preprocess(image)
        preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000.0

        detector_start = time.perf_counter()
        keypoints = self._detector.detect(processed)
        detections = self._to_detections(keypoints, image.shape)
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
        return sorted(detections, key=lambda item: (item.center_y, item.center_x))

    def _build_params(self) -> cv2.SimpleBlobDetector_Params:
        params_config = self.config.get("blob", {})
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