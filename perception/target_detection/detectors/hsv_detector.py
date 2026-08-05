"""HSV color-segmentation target detector."""

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


class HSVDetector(BaseTargetDetector):
    """HSV threshold + contour detector using one shared configuration.

    Unlike Hough/Blob, this detector works in color space rather than
    grayscale, so its bbox/radius come from `cv2.minEnclosingCircle` on
    each surviving contour rather than from an OpenCV keypoint or circle.
    """

    method_name = "hsv"

    def detect(self, image: np.ndarray) -> DetectorResult:
        validate_bgr_image(image)
        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        hsv = self._preprocess(image)
        mask = self._threshold(hsv)
        mask = self._apply_morphology(mask)
        preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000.0

        detector_start = time.perf_counter()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = self._to_detections(contours, image.shape)
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

        # Blur
        blur = preprocessing.get("blur", {})
        blur_type = str(blur.get("type", "median")).lower()
        blurred = image

        if blur_type != "none":
            kernel_size = _odd_kernel_size(int(blur.get("kernel_size", 5)))
            if blur_type == "median":
                blurred = cv2.medianBlur(image, kernel_size)
            elif blur_type == "gaussian":
                sigma_x = float(blur.get("sigma_x", 0))
                blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)
            else:
                raise ValueError(f"Unsupported HSV blur type: {blur_type}")

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # CLAHE
        clahe_config = preprocessing.get("clahe", {})
        if clahe_config.get("enabled", False):
            hsv = self._apply_clahe(hsv, clahe_config)

        return hsv

    def _apply_clahe(self, hsv: np.ndarray, clahe_config: dict[str, Any]) -> np.ndarray:

        #apply CLAHE to values channel only
        h, s, v = cv2.split(hsv)
        tile_size = int(clahe_config.get("tile_grid_size", 8))
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_config.get("clip_limit", 2.0)),
            tileGridSize=(tile_size, tile_size),
        )
        v = clahe.apply(v)
        return cv2.merge([h, s, v])

    def _threshold(self, hsv: np.ndarray) -> np.ndarray:
        hsv_config = self.config.get("hsv", {})
        ranges = hsv_config.get("ranges")

        if ranges:
            mask = None
            for color_range in ranges:
                lower = np.array(color_range["lower"], dtype=np.uint8)
                upper = np.array(color_range["upper"], dtype=np.uint8)
                partial = cv2.inRange(hsv, lower, upper)
                if mask is None:
                    mask = partial
                else:
                    cv2.bitwise_or(mask, partial)

            if mask is None:
                raise ValueError("HSV config 'ranges' must contain at least one entry.")
            return mask

        raise ValueError("HSV config ranges missing.")
    

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        morphology = self.config.get("morphology", {})

        open_kernel_size = int(morphology.get("open_kernel_size", 5))
        open_iterations = int(morphology.get("open_iterations", 1))
        if open_kernel_size > 0 and open_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

        close_kernel_size = int(morphology.get("close_kernel_size", 5))
        close_iterations = int(morphology.get("close_iterations", 1))
        if close_kernel_size > 0 and close_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

        return mask

    def _to_detections(self, contours: tuple[np.ndarray, ...], image_shape: tuple[int, int, int]) -> list[Detection]:
        contour_config = self.config.get("contour", {})
        min_area = float(contour_config.get("min_area", 40))
        min_circularity = float(contour_config.get("min_circularity", 0.55))
        height, width = image_shape[:2]

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            # Circularity of 1.0 --> perfect circle
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue

            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            bbox = circle_bbox(float(center_x), float(center_y), float(radius), width, height)
            detections.append(
                Detection(
                    method=self.method_name,
                    center_x=float(center_x),
                    center_y=float(center_y),
                    radius=float(radius),
                    bbox=bbox,
                    score=circularity,
                )
            )

        return sorted(detections, key=lambda item: (item.center_y, item.center_x))


def _odd_kernel_size(value: int) -> int:
    if value < 1:
        raise ValueError("Blur kernel size must be positive.")
    return value if value % 2 == 1 else value + 1