"""HSV-gated connected-region detector with ellipse-aware geometry."""

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
from perception.target_detection.detectors.blob_detector import BlobDetector
from perception.target_detection.detectors.hsv_detector import HSVDetector
from perception.target_detection.detectors.hough_detector import HoughCircleDetector


class HybridDetector(BaseTargetDetector):
    """Use the shared HSV mask, then recover blob regions and oval geometry."""

    method_name = "hybrid"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        hsv_config: dict[str, Any] | None = None,
        blob_config: dict[str, Any] | None = None,
        hough_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._hsv_detector = HSVDetector(hsv_config or {})
        self._blob_detector = BlobDetector(blob_config or {})
        self._hough_detector = HoughCircleDetector(hough_config or {})

    @classmethod
    def from_config(cls, config_path: str | Path) -> "HybridDetector":
        path = Path(config_path)
        config = load_yaml_config(path)
        hsv_path = path.parent / str(config.get("hsv_config", "hsv.yaml"))
        blob_path = path.parent / str(config.get("blob_config", "blob.yaml"))
        hough_path = path.parent / str(config.get("hough_config", "hough.yaml"))
        return cls(
            config,
            load_yaml_config(hsv_path),
            load_yaml_config(blob_path),
            load_yaml_config(hough_path),
        )

    def detect(self, image: np.ndarray) -> DetectorResult:
        validate_bgr_image(image)
        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        hsv = self._hsv_detector._preprocess(image)
        mask = self._hsv_detector._threshold(hsv)
        if self.config.get("use_hsv_morphology", False):
            mask = self._hsv_detector._apply_morphology(mask)
        mask = self._merge_nearby_regions(mask)
        preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000.0

        detector_start = time.perf_counter()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = self._to_detections(contours, image.shape)
        detections.extend(self._white_blob_fallback(image, hsv, detections))
        detections.extend(self._shadow_yellow_hough_fallback(image, hsv, detections))
        detections = sorted(detections, key=lambda item: (item.center_y, item.center_x))
        detection_ms = (time.perf_counter() - detector_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0
        return DetectorResult(
            method=self.method_name,
            detections=detections,
            timing=TimingInfo(preprocessing_ms, detection_ms, total_ms),
            image_shape=tuple(int(value) for value in image.shape),
        )

    def _merge_nearby_regions(self, mask: np.ndarray) -> np.ndarray:
        merge = self.config.get("region_merge", {})
        kernel_size = int(merge.get("close_kernel_size", 0))
        iterations = int(merge.get("close_iterations", 0))
        if kernel_size <= 0 or iterations <= 0:
            return mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_odd_kernel_size(kernel_size), _odd_kernel_size(kernel_size)),
        )
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    def _white_blob_fallback(
        self,
        image: np.ndarray,
        hsv: np.ndarray,
        existing: list[Detection],
    ) -> list[Detection]:
        fallback = self.config.get("white_blob_fallback", {})
        if not fallback.get("enabled", False):
            return []
        min_fraction = float(fallback.get("min_white_fraction", 0.6))
        max_saturation = int(fallback.get("max_saturation", 50))
        min_value = int(fallback.get("min_value", 140))
        min_area = float(fallback.get("min_bbox_area", 500))
        max_area = float(fallback.get("max_bbox_area", 5000))
        max_aspect_ratio = float(fallback.get("max_aspect_ratio", 0.8))
        min_refined_area = float(fallback.get("min_refined_bbox_area", 0))
        accepted: list[Detection] = []
        for candidate in self._blob_detector.detect(image).detections:
            x, y, box_width, box_height = candidate.bbox
            area = box_width * box_height
            if area < min_area or area > max_area:
                continue
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2 = min(hsv.shape[1], int(math.ceil(x + box_width)))
            y2 = min(hsv.shape[0], int(math.ceil(y + box_height)))
            crop = hsv[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            white = (crop[:, :, 1] <= max_saturation) & (crop[:, :, 2] >= min_value)
            white_fraction = float(np.mean(white))
            if white_fraction < min_fraction:
                continue
            contours, _ = cv2.findContours(
                (white.astype(np.uint8) * 255),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            local_x, local_y, local_width, local_height = cv2.boundingRect(hull)
            short_side = min(local_width, local_height)
            long_side = max(local_width, local_height)
            if long_side <= 0 or short_side / long_side > max_aspect_ratio:
                continue
            if local_width * local_height < min_refined_area:
                continue
            if len(hull) >= 5:
                (local_cx, local_cy), (axis_a, axis_b), ellipse_angle = cv2.fitEllipse(hull)
            else:
                (local_cx, local_cy), (axis_a, axis_b), ellipse_angle = cv2.minAreaRect(hull)
            refined_bbox = (
                float(x1 + local_x),
                float(y1 + local_y),
                float(local_width),
                float(local_height),
            )
            refined_radius = math.sqrt(max(axis_a, 1.0) * max(axis_b, 1.0)) / 2.0
            if any(bbox_iou(candidate.bbox, item.bbox) >= 0.25 for item in existing + accepted):
                continue
            accepted.append(
                Detection(
                    method=self.method_name,
                    center_x=float(x1 + local_cx),
                    center_y=float(y1 + local_cy),
                    radius=float(refined_radius),
                    bbox=refined_bbox,
                    score=white_fraction,
                    ellipse_axes=(float(axis_a / 2.0), float(axis_b / 2.0)),
                    ellipse_angle=float(ellipse_angle),
                )
            )
        return accepted

    def _shadow_yellow_hough_fallback(
        self,
        image: np.ndarray,
        hsv: np.ndarray,
        existing: list[Detection],
    ) -> list[Detection]:
        fallback = self.config.get("shadow_yellow_hough_fallback", {})
        if not fallback.get("enabled", False):
            return []
        if fallback.get("only_when_empty", True) and existing:
            return []
        min_yellow = float(fallback.get("min_shadow_yellow_fraction", 0.75))
        min_pale = float(fallback.get("min_pale_fraction", 0.80))
        radius_scale = float(fallback.get("radius_scale", 1.0))
        accepted: list[Detection] = []
        for candidate in self._hough_detector.detect(image).detections:
            x, y, box_width, box_height = candidate.bbox
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2 = min(hsv.shape[1], int(math.ceil(x + box_width)))
            y2 = min(hsv.shape[0], int(math.ceil(y + box_height)))
            crop = hsv[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            shadow_yellow = (
                (crop[:, :, 0] >= 12)
                & (crop[:, :, 0] <= 28)
                & (crop[:, :, 1] >= 20)
                & (crop[:, :, 1] <= 199)
                & (crop[:, :, 2] >= 120)
            )
            pale = (crop[:, :, 1] <= 120) & (crop[:, :, 2] >= 120)
            if float(np.mean(shadow_yellow)) < min_yellow or float(np.mean(pale)) < min_pale:
                continue
            center_x, center_y, refined_radius = self._refine_hough_circle(image, candidate)
            scaled_radius = float(refined_radius or (candidate.radius or 0.0) * radius_scale)
            accepted.append(
                Detection(
                    method=self.method_name,
                    center_x=center_x,
                    center_y=center_y,
                    radius=scaled_radius,
                    bbox=circle_bbox(
                        center_x,
                        center_y,
                        scaled_radius,
                        image.shape[1],
                        image.shape[0],
                    ),
                    score=float(np.mean(shadow_yellow)),
                )
            )
        return accepted

    def _refine_hough_circle(
        self,
        image: np.ndarray,
        candidate: Detection,
    ) -> tuple[float, float, float | None]:
        fallback = self.config.get("shadow_yellow_hough_fallback", {})
        candidate_radius = candidate.radius or 0.0
        if candidate_radius <= 0.0:
            return candidate.center_x, candidate.center_y, None
        hough = self._hough_detector.config.get("hough", {})
        gray = self._hough_detector._preprocess(image)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=float(hough.get("dp", 1.2)),
            minDist=float(hough.get("min_dist", 90)),
            param1=float(hough.get("param1", 120)),
            param2=float(hough.get("param2", 28)),
            minRadius=max(1, int(candidate_radius * float(fallback.get("refine_min_radius_scale", 1.1)))),
            maxRadius=max(1, int(candidate_radius * float(fallback.get("refine_max_radius_scale", 1.75)))),
        )
        if circles is None:
            return candidate.center_x, candidate.center_y, None
        nearest = min(
            circles[0],
            key=lambda circle: (circle[0] - candidate.center_x) ** 2
            + (circle[1] - candidate.center_y) ** 2,
        )
        shift = math.hypot(nearest[0] - candidate.center_x, nearest[1] - candidate.center_y)
        if shift > candidate_radius * float(fallback.get("max_center_shift_factor", 0.6)):
            return candidate.center_x, candidate.center_y, None
        return float(nearest[0]), float(nearest[1]), float(nearest[2])

    def _to_detections(
        self,
        contours: tuple[np.ndarray, ...],
        image_shape: tuple[int, int, int],
    ) -> list[Detection]:
        region = self.config.get("region", {})
        min_area = float(region.get("min_area", 150))
        max_area = float(region.get("max_area", 50000))
        small_max_area = float(region.get("small_region_max_area", min_area))
        large_min_area = float(region.get("large_region_min_area", min_area))
        min_solidity = float(region.get("min_solidity", 0.65))
        min_aspect_ratio = float(region.get("min_aspect_ratio", 0.2))
        ellipse_threshold = float(region.get("ellipse_aspect_ratio_threshold", 0.75))
        height, width = image_shape[:2]
        detections: list[Detection] = []

        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < min_area or contour_area > max_area:
                continue
            if small_max_area < contour_area < large_min_area:
                continue
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area <= 0.0:
                continue
            solidity = contour_area / hull_area
            if solidity < min_solidity:
                continue

            x, y, box_width, box_height = cv2.boundingRect(hull)
            if region.get("reject_border_touching", False) and (
                x <= 0 or y <= 0 or x + box_width >= width or y + box_height >= height
            ):
                continue
            short_side = min(box_width, box_height)
            long_side = max(box_width, box_height)
            if long_side <= 0 or short_side / long_side < min_aspect_ratio:
                continue

            aspect_ratio = short_side / long_side
            if aspect_ratio >= ellipse_threshold:
                (center_x, center_y), radius = cv2.minEnclosingCircle(hull)
                bbox = circle_bbox(float(center_x), float(center_y), float(radius), width, height)
            else:
                if len(hull) >= 5:
                    (center_x, center_y), (axis_a, axis_b), ellipse_angle = cv2.fitEllipse(hull)
                else:
                    (center_x, center_y), (axis_a, axis_b), ellipse_angle = cv2.minAreaRect(hull)
                radius = math.sqrt(max(axis_a, 1.0) * max(axis_b, 1.0)) / 2.0
                bbox = (
                    float(max(0, x)),
                    float(max(0, y)),
                    float(min(box_width, width - x)),
                    float(min(box_height, height - y)),
                )
            detections.append(
                Detection(
                    method=self.method_name,
                    center_x=float(center_x),
                    center_y=float(center_y),
                    radius=float(radius),
                    bbox=bbox,
                    score=float(solidity),
                    ellipse_axes=(float(axis_a / 2.0), float(axis_b / 2.0)) if aspect_ratio < ellipse_threshold else None,
                    ellipse_angle=float(ellipse_angle) if aspect_ratio < ellipse_threshold else None,
                )
            )
        return sorted(detections, key=lambda item: (item.center_y, item.center_x))


def _odd_kernel_size(value: int) -> int:
    if value < 1:
        raise ValueError("Kernel size must be positive.")
    return value if value % 2 == 1 else value + 1
