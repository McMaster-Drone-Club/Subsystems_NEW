"""Run a target detector on one image or a directory of images."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.target_detection.detectors import BlobDetector, HoughCircleDetector, HSVDetector, HybridDetector
from perception.target_detection.detectors.base import Detection, DetectorResult

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_CONFIGS = {
    "hough": REPO_ROOT / "perception" / "target_detection" / "configs" / "hough.yaml",
    "blob": REPO_ROOT / "perception" / "target_detection" / "configs" / "blob.yaml",
    "hsv": REPO_ROOT / "perception" / "target_detection" / "configs" / "hsv.yaml",
    "hybrid": REPO_ROOT / "perception" / "target_detection" / "configs" / "hybrid.yaml",
}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "perception" / "target_detection" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hough or blob target detection.")
    parser.add_argument("--detector", choices=("hough", "blob", "hsv", "hybrid"), required=True)
    parser.add_argument("--input", required=True, help="Image file or directory of images.")
    parser.add_argument("--config", help="YAML detector config path. Defaults to detector config.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON lines instead of readable text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 2

    config_path = Path(args.config) if args.config else DEFAULT_CONFIGS[args.detector]
    if not config_path.exists():
        print(f"Config path does not exist: {config_path}", file=sys.stderr)
        return 2

    detector = build_detector(args.detector, config_path)
    image_paths = collect_image_paths(input_path)
    if not image_paths:
        print(f"No readable image files found at: {input_path}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    if args.save_annotated:
        output_dir.mkdir(parents=True, exist_ok=True)

    failed: list[Path] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failed.append(image_path)
            print(f"Could not read image: {image_path}", file=sys.stderr)
            continue

        result = detector.detect(image)
        if args.save_annotated:
            annotated = annotate_image(image, result)
            output_path = output_dir / f"{image_path.stem}_{args.detector}{image_path.suffix}"
            cv2.imwrite(str(output_path), annotated)
        print_result(image_path, result, args.json)

    if failed:
        print(f"Failed to read {len(failed)} file(s).", file=sys.stderr)
        return 1
    return 0


def build_detector(detector_name: str, config_path: Path):
    if detector_name == "hough":
        return HoughCircleDetector.from_config(config_path)
    if detector_name == "blob":
        return BlobDetector.from_config(config_path)
    if detector_name == "hsv":
        return HSVDetector.from_config(config_path)
    if detector_name == "hybrid":
        return HybridDetector.from_config(config_path)
    raise ValueError(f"Unsupported detector: {detector_name}")


def collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(
        [path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: natural_image_key(path),
    )


def natural_image_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def annotate_image(image, result: DetectorResult):
    annotated = image.copy()
    for detection in result.detections:
        draw_detection(annotated, detection)
    cv2.putText(
        annotated,
        f"{result.method}: {len(result.detections)} detections",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"{result.method}: {len(result.detections)} detections",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_detection(image, detection: Detection) -> None:
    center = (int(round(detection.center_x)), int(round(detection.center_y)))
    radius = int(round(detection.radius or max(detection.bbox[2], detection.bbox[3]) / 2.0))
    x, y, width, height = detection.bbox
    cv2.circle(image, center, 4, (0, 0, 255), -1)
    if detection.ellipse_axes is not None:
        axes = (
            max(1, int(round(detection.ellipse_axes[0]))),
            max(1, int(round(detection.ellipse_axes[1]))),
        )
        cv2.ellipse(
            image,
            center,
            axes,
            float(detection.ellipse_angle or 0.0),
            0,
            360,
            (0, 255, 0),
            2,
        )
    else:
        cv2.circle(image, center, max(1, radius), (0, 255, 0), 2)
    cv2.rectangle(
        image,
        (int(round(x)), int(round(y))),
        (int(round(x + width)), int(round(y + height))),
        (255, 0, 0),
        2,
    )
    label = detection.method
    if detection.score is not None:
        label = f"{label} {detection.score:.2f}"
    cv2.putText(
        image,
        label,
        (center[0] + 8, max(16, center[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (center[0] + 8, max(16, center[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def print_result(image_path: Path, result: DetectorResult, as_json: bool) -> None:
    if as_json:
        payload = {
            "image": str(image_path),
            "method": result.method,
            "detections": [detection.as_dict() for detection in result.detections],
            "timing": asdict(result.timing),
            "image_shape": result.image_shape,
        }
        print(json.dumps(payload))
        return

    timing = result.timing
    print(
        f"{image_path}: {len(result.detections)} detection(s); "
        f"pre={timing.preprocessing_ms:.2f} ms, "
        f"detect={timing.detection_ms:.2f} ms, total={timing.total_ms:.2f} ms"
    )
    for index, detection in enumerate(result.detections, start=1):
        x, y, width, height = detection.bbox
        score = "n/a" if detection.score is None else f"{detection.score:.3f}"
        radius = "n/a" if detection.radius is None else f"{detection.radius:.1f}"
        print(
            f"  {index}. center=({detection.center_x:.1f}, {detection.center_y:.1f}), "
            f"radius={radius}, bbox=({x:.1f}, {y:.1f}, {width:.1f}, {height:.1f}), "
            f"score={score}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
