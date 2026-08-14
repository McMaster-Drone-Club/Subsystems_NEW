from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.target_detection.detectors import BlobDetector, HoughCircleDetector, HybridDetector
from perception.target_detection.detectors.base import Detection, DetectorResult, load_yaml_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def make_image(circles: list[tuple[int, int, int]], size: tuple[int, int] = (240, 320)) -> np.ndarray:
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for center_x, center_y, radius in circles:
        cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), -1)
    return image


def assert_has_detection_near(
    result: DetectorResult,
    expected_x: float,
    expected_y: float,
    expected_radius: float,
    center_tol: float = 8.0,
    radius_tol: float = 10.0,
) -> None:
    for detection in result.detections:
        center_error = np.hypot(detection.center_x - expected_x, detection.center_y - expected_y)
        radius_error = abs((detection.radius or 0.0) - expected_radius)
        if center_error <= center_tol and radius_error <= radius_tol:
            return
    raise AssertionError(f"No detection near ({expected_x}, {expected_y}, r={expected_radius}) in {result}")


@pytest.mark.parametrize("detector_cls,config_name", [
    (HoughCircleDetector, "hough.yaml"),
    (BlobDetector, "blob.yaml"),
])
def test_common_output_structure(detector_cls, config_name: str) -> None:
    detector = detector_cls.from_config(CONFIG_DIR / config_name)
    result = detector.detect(make_image([(120, 100, 35)]))

    assert isinstance(result, DetectorResult)
    assert isinstance(result.detections, list)
    assert result.method in {"hough", "blob"}
    assert result.timing.preprocessing_ms >= 0.0
    assert result.timing.detection_ms >= 0.0
    assert result.timing.total_ms >= 0.0
    if result.detections:
        detection = result.detections[0]
        assert isinstance(detection, Detection)
        assert len(detection.bbox) == 4


@pytest.mark.parametrize("detector_cls,config_name", [
    (HoughCircleDetector, "hough.yaml"),
    (BlobDetector, "blob.yaml"),
])
def test_one_clear_circle(detector_cls, config_name: str) -> None:
    detector = detector_cls.from_config(CONFIG_DIR / config_name)
    result = detector.detect(make_image([(160, 120, 40)]))

    assert_has_detection_near(result, 160, 120, 40)


@pytest.mark.parametrize("detector_cls,config_name", [
    (HoughCircleDetector, "hough.yaml"),
    (BlobDetector, "blob.yaml"),
])
def test_multiple_circles(detector_cls, config_name: str) -> None:
    detector = detector_cls.from_config(CONFIG_DIR / config_name)
    result = detector.detect(make_image([(85, 90, 28), (230, 145, 34)]))

    assert len(result.detections) >= 2
    assert_has_detection_near(result, 85, 90, 28)
    assert_has_detection_near(result, 230, 145, 34)


@pytest.mark.parametrize("detector_cls,config_name", [
    (HoughCircleDetector, "hough.yaml"),
    (BlobDetector, "blob.yaml"),
])
def test_no_circle_returns_empty_list(detector_cls, config_name: str) -> None:
    detector = detector_cls.from_config(CONFIG_DIR / config_name)
    result = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))

    assert isinstance(result.detections, list)
    assert result.detections == []


@pytest.mark.parametrize("detector_cls,config_name", [
    (HoughCircleDetector, "hough.yaml"),
    (BlobDetector, "blob.yaml"),
])
def test_small_circle(detector_cls, config_name: str) -> None:
    detector = detector_cls.from_config(CONFIG_DIR / config_name)
    result = detector.detect(make_image([(90, 80, 12)], size=(160, 180)))

    assert_has_detection_near(result, 90, 80, 12, center_tol=8.0, radius_tol=8.0)


@pytest.mark.parametrize("bad_image", [
    None,
    np.array([], dtype=np.uint8),
    np.zeros((20, 20), dtype=np.uint8),
    np.zeros((20, 20, 4), dtype=np.uint8),
    np.zeros((20, 20, 3), dtype=np.float32),
])
def test_invalid_or_empty_input(bad_image) -> None:
    detector = HoughCircleDetector.from_config(CONFIG_DIR / "hough.yaml")
    with pytest.raises((TypeError, ValueError)):
        detector.detect(bad_image)


def test_configuration_loading() -> None:
    config = load_yaml_config(CONFIG_DIR / "hough.yaml")
    assert config["method"] == "hough"
    assert "hough" in config


def test_partially_visible_circle_does_not_crash() -> None:
    detector = HoughCircleDetector.from_config(CONFIG_DIR / "hough.yaml")
    image = make_image([(-4, 80, 34)], size=(160, 180))
    result = detector.detect(image)

    assert isinstance(result.detections, list)


def test_blob_detects_oval() -> None:
    detector = BlobDetector.from_config(CONFIG_DIR / "blob.yaml")
    image = np.zeros((260, 320, 3), dtype=np.uint8)
    cv2.ellipse(image, (160, 130), (45, 85), 0, 0, 360, (255, 255, 255), -1)

    result = detector.detect(image)

    assert any(np.hypot(item.center_x - 160, item.center_y - 130) <= 10 for item in result.detections)


def test_blob_detects_partially_occluded_oval() -> None:
    detector = BlobDetector.from_config(CONFIG_DIR / "blob.yaml")
    image = np.zeros((260, 320, 3), dtype=np.uint8)
    cv2.ellipse(image, (160, 130), (55, 85), 0, 0, 360, (255, 255, 255), -1)
    cv2.rectangle(image, (155, 45), (205, 145), (0, 0, 0), -1)

    result = detector.detect(image)

    assert any(np.hypot(item.center_x - 145, item.center_y - 145) <= 35 for item in result.detections)


def test_hybrid_detects_colored_oval() -> None:
    detector = HybridDetector.from_config(CONFIG_DIR / "hybrid.yaml")
    hsv_image = np.zeros((260, 320, 3), dtype=np.uint8)
    cv2.ellipse(hsv_image, (160, 130), (45, 80), 0, 0, 360, (90, 220, 220), -1)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    result = detector.detect(image)

    assert any(np.hypot(item.center_x - 160, item.center_y - 130) <= 10 for item in result.detections)


def test_hybrid_white_blob_fallback() -> None:
    detector = HybridDetector.from_config(CONFIG_DIR / "hybrid.yaml")
    image = np.zeros((260, 320, 3), dtype=np.uint8)
    cv2.ellipse(image, (160, 130), (40, 20), 0, 0, 360, (230, 230, 230), -1)

    result = detector.detect(image)

    assert any(np.hypot(item.center_x - 160, item.center_y - 130) <= 10 for item in result.detections)
