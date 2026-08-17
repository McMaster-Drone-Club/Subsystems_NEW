"""Benchmark Hough, blob, and HSV target detectors on one labeled dataset.

Rerun with:

    python perception/target_detection/scripts/benchmark.py

which uses the default annotations, image directory, and per-method config
files already checked into the repo. All paths can be overridden; run with the
--help flag

Ground truth format
--------------------
Reads a Roboflow COCO export where each target is labeled with two
annotations sharing an image: one `Circle` category bbox (the target's
extent) and one `Center` category point (the true center, marked
independently of the bbox so it stays accurate under partial occlusion).
Each image's Circle boxes are paired with its Center points by nearest
distance. Images may contain zero, one, or several targets.

Detection-to-ground-truth matching
------------------------------------
For each image and method, detections are greedily matched to ground-truth
targets by ascending center distance. A detection may match a target only if
their centers are within `max(target_radius * radius_multiplier,
MIN_MATCH_RADIUS_PX)` pixels and the detection's bbox overlaps the
target's bbox by at least MIN_MATCH_IOU. The IoU floor exists so a tiny,
wrongly-sized detection that merely happens to fall near a large target's
center doesn't get credited as a correct find; each detection/target is used
at most once. Unmatched detections are false positives; unmatched targets
are missed detections (false negatives).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.target_detection.detectors import (
    BlobDetector,
    HoughCircleDetector,
    HSVDetector,
    HybridDetector,
)
from perception.target_detection.detectors.base import BBox, Detection, DetectorResult, bbox_iou

# Reuse the single-image drawing helpers so annotated outputs look the same
# whether produced by run_detector.py or this script.
from run_detector import annotate_image  # noqa: E402  (path set up above)

TRAIN_ANNOTATIONS = REPO_ROOT / "image-annotations" / "training-annotations.coco.json"
TRAIN_IMAGES_DIR = REPO_ROOT / "training-images"
TRAIN_OUTPUT_DIR = REPO_ROOT / "perception" / "target_detection" / "outputs" / "benchmark" / "train"

TEST_ANNOTATIONS = REPO_ROOT / "image-annotations" / "test-annotations.coco.json"
TEST_IMAGES_DIR = REPO_ROOT / "test-images"
TEST_OUTPUT_DIR = REPO_ROOT / "perception" / "target_detection" / "outputs" / "benchmark" / "test"

# (label, annotations_path, images_dir, output_dir) for each pass, run in order.
DATASETS: list[tuple[str, Path, Path, Path]] = [
    ("train", TRAIN_ANNOTATIONS, TRAIN_IMAGES_DIR, TRAIN_OUTPUT_DIR),
    ("test", TEST_ANNOTATIONS, TEST_IMAGES_DIR, TEST_OUTPUT_DIR),
]

DEFAULT_CONFIGS = {
    "hough": REPO_ROOT / "perception" / "target_detection" / "configs" / "hough.yaml",
    "blob": REPO_ROOT / "perception" / "target_detection" / "configs" / "blob.yaml",
    "hsv": REPO_ROOT / "perception" / "target_detection" / "configs" / "hsv.yaml",
    "hybrid": REPO_ROOT / "perception" / "target_detection" / "configs" / "hybrid.yaml",
}
DETECTOR_CLASSES = {
    "hough": HoughCircleDetector,
    "blob": BlobDetector,
    "hsv": HSVDetector,
    "hybrid": HybridDetector,
}

MIN_MATCH_RADIUS_PX = 8.0
''' A detection matches a ground-truth target only if BOTH:
       1. their centers fall within max(target_radius * MATCH_RADIUS_MULTIPLIER,
       MIN_MATCH_RADIUS_PX) pixels, and
       2. their bboxes overlap by at least MIN_MATCH_IOU.
       The IoU floor stops a small, wrongly-sized detection from being credited
       as a correct find just because it lands near a much larger target's center. '''
MATCH_RADIUS_MULTIPLIER = 1.0
MIN_MATCH_IOU = 0.5
GT_CIRCLE_COLOR = (255, 0, 255)  # magenta
GT_CENTER_COLOR = (0, 255, 255)  # yellow


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruthTarget:
    """One labeled target: a Circle bbox paired with its Center point."""

    center_x: float
    center_y: float
    bbox: BBox
    radius: float  # equal-area radius derived from the bbox, handles elliptical bboxes


@dataclass(frozen=True)
class GroundTruthImage:
    targets: list[GroundTruthTarget]


def load_ground_truth(annotations_path: Path) -> dict[str, GroundTruthImage]:
    with annotations_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    category_id_by_name = {category["name"].lower(): category["id"] for category in data["categories"]}
    circle_id = _find_category_id(category_id_by_name, "circle")
    center_id = _find_category_id(category_id_by_name, "center")

    image_id_to_name = {
        image["id"]: (image.get("extra") or {}).get("name", image["file_name"]) for image in data["images"]
    }

    circle_anns: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_id_to_name}
    center_anns: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_id_to_name}
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if annotation["category_id"] == circle_id:
            circle_anns[image_id].append(annotation)
        elif annotation["category_id"] == center_id:
            center_anns[image_id].append(annotation)

    ground_truth: dict[str, GroundTruthImage] = {}
    for image_id, file_name in image_id_to_name.items():
        targets = _pair_circles_with_centers(circle_anns[image_id], center_anns[image_id], file_name)
        ground_truth[file_name] = GroundTruthImage(targets=targets)
    return ground_truth


def _find_category_id(category_id_by_name: dict[str, int], name: str) -> int:
    if name not in category_id_by_name:
        raise ValueError(
            f"Annotation file has no '{name}' category. Found: {sorted(category_id_by_name)}"
        )
    return category_id_by_name[name]


def _pair_circles_with_centers(
    circles: list[dict[str, Any]], centers: list[dict[str, Any]], file_name: str
) -> list[GroundTruthTarget]:
    if len(circles) != len(centers):
        raise ValueError(
            f"{file_name}: {len(circles)} Circle annotation(s) but {len(centers)} Center "
            "annotation(s); every target needs exactly one of each."
        )
    if not circles:
        return []

    circle_centers = [(box["bbox"][0] + box["bbox"][2] / 2.0, box["bbox"][1] + box["bbox"][3] / 2.0) for box in circles]
    remaining_centers = list(range(len(centers)))

    targets: list[GroundTruthTarget] = []
    for circle, (approx_x, approx_y) in zip(circles, circle_centers):
        best_index = min(
            remaining_centers,
            key=lambda i: (centers[i]["bbox"][0] - approx_x) ** 2 + (centers[i]["bbox"][1] - approx_y) ** 2,
        )
        remaining_centers.remove(best_index)
        center_ann = centers[best_index]
        cx = center_ann["bbox"][0] + center_ann["bbox"][2] / 2.0
        cy = center_ann["bbox"][1] + center_ann["bbox"][3] / 2.0

        x, y, w, h = circle["bbox"]
        radius = ((w * h) ** 0.5) / 2.0  # equal-area radius; fair for elliptical/tilted targets
        targets.append(
            GroundTruthTarget(center_x=cx, center_y=cy, bbox=(x, y, w, h), radius=radius)
        )
    return targets


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageEvaluation:
    image: str
    num_gt: int
    num_det: int
    false_positives: int
    false_negatives: int
    center_errors_px: list[float]  # matched pairs only
    ious: list[float]  # matched pairs only
    timing: DetectorResult


def match_detections(
    detections: list[Detection], targets: list[GroundTruthTarget], radius_multiplier: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """one-to-one matching by ascending center distance.

    A candidate pair must satisfy both the center-distance threshold and the
    MIN_MATCH_IOU floor (see the module-level constants) before it's eligible
    to match at all.

    Returns (matches, unmatched_detection_indices, unmatched_target_indices) where
    matches is a list of (detection_index, target_index) pairs.
    """
    candidates: list[tuple[float, int, int]] = []
    for t_idx, target in enumerate(targets):
        threshold = max(target.radius * radius_multiplier, MIN_MATCH_RADIUS_PX)
        for d_idx, detection in enumerate(detections):
            distance = ((detection.center_x - target.center_x) ** 2 + (detection.center_y - target.center_y) ** 2) ** 0.5
            if distance <= threshold and bbox_iou(detection.bbox, target.bbox) >= MIN_MATCH_IOU:
                candidates.append((distance, d_idx, t_idx))
    candidates.sort(key=lambda item: item[0])

    matched_detections: set[int] = set()
    matched_targets: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, d_idx, t_idx in candidates:
        if d_idx in matched_detections or t_idx in matched_targets:
            continue
        matched_detections.add(d_idx)
        matched_targets.add(t_idx)
        matches.append((d_idx, t_idx))

    unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
    unmatched_targets = [i for i in range(len(targets)) if i not in matched_targets]
    return matches, unmatched_detections, unmatched_targets


def evaluate_image(
    result: DetectorResult, ground_truth: GroundTruthImage, radius_multiplier: float, image_name: str
) -> ImageEvaluation:
    matches, unmatched_detections, unmatched_targets = match_detections(
        result.detections, ground_truth.targets, radius_multiplier
    )

    center_errors: list[float] = []
    ious: list[float] = []
    for d_idx, t_idx in matches:
        detection = result.detections[d_idx]
        target = ground_truth.targets[t_idx]
        center_errors.append(((detection.center_x - target.center_x) ** 2 + (detection.center_y - target.center_y) ** 2) ** 0.5)
        ious.append(bbox_iou(detection.bbox, target.bbox))

    return ImageEvaluation(
        image=image_name,
        num_gt=len(ground_truth.targets),
        num_det=len(result.detections),
        false_positives=len(unmatched_detections),
        false_negatives=len(unmatched_targets),
        center_errors_px=center_errors,
        ious=ious,
        timing=result,
    )


# --------------------------------------------------------------------------
# Annotation drawing (detections + ground truth overlay)
# --------------------------------------------------------------------------


def draw_ground_truth(image: np.ndarray, target: GroundTruthTarget) -> None:
    x, y, w, h = target.bbox
    cv2.ellipse(
        image,
        (int(round(x + w / 2)), int(round(y + h / 2))),
        (max(1, int(round(w / 2))), max(1, int(round(h / 2)))),
        0,
        0,
        360,
        GT_CIRCLE_COLOR,
        2,
    )
    cv2.drawMarker(
        image,
        (int(round(target.center_x)), int(round(target.center_y))),
        GT_CENTER_COLOR,
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
    )


def build_annotated_image(image: np.ndarray, result: DetectorResult, ground_truth: GroundTruthImage) -> np.ndarray:
    annotated = annotate_image(image, result)
    for target in ground_truth.targets:
        draw_ground_truth(annotated, target)
    legend = f"magenta/yellow = ground truth ({len(ground_truth.targets)})"
    cv2.putText(annotated, legend, (12, annotated.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(annotated, legend, (12, annotated.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def summarize_method(evaluations: list[ImageEvaluation]) -> dict[str, Any]:
    all_center_errors = [e for ev in evaluations for e in ev.center_errors_px]
    all_ious = [i for ev in evaluations for i in ev.ious]
    all_total_ms = [ev.timing.timing.total_ms for ev in evaluations]
    all_detect_ms = [ev.timing.timing.detection_ms for ev in evaluations]
    all_preprocess_ms = [ev.timing.timing.preprocessing_ms for ev in evaluations]
    true_positives = sum(ev.num_gt - ev.false_negatives for ev in evaluations)
    false_positives = sum(ev.false_positives for ev in evaluations)
    false_negatives = sum(ev.false_negatives for ev in evaluations)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "center_error_px": _stats(all_center_errors),
        "iou": _stats(all_ious),
        "total_latency_ms": _stats(all_total_ms),
        "detection_latency_ms": _stats(all_detect_ms),
        "preprocessing_latency_ms": _stats(all_preprocess_ms),
    }


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark classical target detectors against labeled ground truth. "
            "Run with no arguments to evaluate all three detectors, using their "
            "checked-in configs, against the repo's labeled dataset."
        )
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=("hough", "blob", "hsv", "hybrid"),
        default=("hough", "blob", "hsv", "hybrid"),
        help="Subset of detectors to run, e.g. --detectors hsv hybrid. Defaults to all four.",
    )
    parser.add_argument(
        "--dataset",
        choices=("train", "test"),
        default=None,
        help="Run only the train or only the test pass. Defaults to running both, train then test.",
    )
    return parser.parse_args()


def _natural_sort_key(file_name: str) -> tuple[int, str]:
    """Sort '0.png', '1.png', ..., '10.png' numerically instead of lexicographically.

    Falls back to a pure string sort (after the numeric ones) for any filename
    that isn't just digits, so this doesn't blow up on unexpected names.
    """
    stem = Path(file_name).stem
    if stem.isdigit():
        return (0, f"{int(stem):010d}")
    return (1, file_name)


def run_dataset(
    label: str,
    annotations_path: Path,
    images_dir: Path,
    output_dir: Path,
    detectors: dict[str, Any],
) -> None:
    """Run every detector over one labeled image set and print its report."""
    ground_truth = load_ground_truth(annotations_path)

    missing_images = [name for name in ground_truth if not (images_dir / name).exists()]
    if missing_images:
        print(f"Warning: {len(missing_images)} labeled image(s) not found in {images_dir}: {missing_images}", file=sys.stderr)

    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    evaluations_by_method: dict[str, list[ImageEvaluation]] = {name: [] for name in detectors}

    image_names = sorted(
        (name for name in ground_truth if (images_dir / name).exists()),
        key=_natural_sort_key,
    )
    for image_name in image_names:
        gt_image = ground_truth[image_name]
        image_path = images_dir / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: could not read {image_path}", file=sys.stderr)
            continue

        for method, detector in detectors.items():
            result = detector.detect(image)
            evaluation = evaluate_image(result, gt_image, MATCH_RADIUS_MULTIPLIER, image_name)
            evaluations_by_method[method].append(evaluation)

            annotated = build_annotated_image(image, result, gt_image)
            out_path = annotated_dir / f"{image_path.stem}_{method}{image_path.suffix}"
            cv2.imwrite(str(out_path), annotated)

    summaries = {method: summarize_method(evaluations) for method, evaluations in evaluations_by_method.items()}

    banner = f" {label.upper()} SET ".center(70, "=")
    print(f"\n{banner}")
    print_accuracy_table(summaries, len(image_names))
    print_latency_table(summaries)
    print_failure_cases(evaluations_by_method)


def main() -> int:
    args = parse_args()

    detectors = {name: DETECTOR_CLASSES[name].from_config(DEFAULT_CONFIGS[name]) for name in args.detectors}

    datasets = DATASETS if args.dataset is None else [d for d in DATASETS if d[0] == args.dataset]

    for label, annotations_path, images_dir, output_dir in datasets:
        run_dataset(label, annotations_path, images_dir, output_dir, detectors)
    return 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_accuracy_table(summaries: dict[str, dict[str, Any]], num_images: int) -> None:
    print(f"\nEvaluated {num_images} image(s).\n")
    print("Accuracy")
    header = (
        f"{'method':<8} {'TP':>4} {'FP':>5} {'FN':>4} "
        f"{'precision':>10} {'recall':>8} {'F1':>8} "
        f"{'center_err_px':>14} {'iou':>8} {'total_latency_ms':>18}"
    )
    print(header)
    print("-" * len(header))
    for method, summary in summaries.items():
        print(
            f"{method:<8} "
            f"{summary['true_positives']:>4} "
            f"{summary['false_positives']:>5} "
            f"{summary['false_negatives']:>4} "
            f"{_fmt(summary['precision']):>10} "
            f"{_fmt(summary['recall']):>8} "
            f"{_fmt(summary['f1']):>8} "
            f"{_fmt(summary['center_error_px']['mean']):>14} "
            f"{_fmt(summary['iou']['mean']):>8} "
            f"{_fmt(summary['total_latency_ms']['mean']):>18}"
        )
    print()


def print_latency_table(summaries: dict[str, dict[str, Any]]) -> None:
    print("Latency by step (ms)")
    header = f"{'method':<8} {'step':<14} {'mean':>9} {'median':>9} {'min':>9} {'max':>9}"
    print(header)
    print("-" * len(header))
    steps = (
        ("preprocessing", "preprocessing_latency_ms"),
        ("detection", "detection_latency_ms"),
        ("total", "total_latency_ms"),
    )
    for method, summary in summaries.items():
        for step_label, key in steps:
            stats = summary[key]
            print(
                f"{method:<8} "
                f"{step_label:<14} "
                f"{_fmt(stats['mean']):>9} "
                f"{_fmt(stats['median']):>9} "
                f"{_fmt(stats['min']):>9} "
                f"{_fmt(stats['max']):>9}"
            )
    print()


def print_failure_cases(evaluations_by_method: dict[str, list[ImageEvaluation]]) -> None:
    print("Failure cases (false positives or missed detections)")
    for method, evaluations in evaluations_by_method.items():
        failures = [e for e in evaluations if e.false_positives or e.false_negatives]
        print(f"\n{method}:")
        if not failures:
            print("  none")
            continue
        header = f"  {'image':<10} {'gt_targets':>10}    {'detections':>10} {'false_positives':>16} {'missed':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for e in failures:
            print(f"  {e.image:<10} {e.num_gt:>10}    {e.num_det:>10} {e.false_positives:>16} {e.false_negatives:>7}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())