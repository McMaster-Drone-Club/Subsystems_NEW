# Target Detection Experiments

## Purpose and scope

This package contains Hough-circle, blob, and HSV colored-target detectors plus a shared labeled benchmark for McMaster-Drone-Club/VM_Scripts_NEW issue #4.

## Folder structure

```text
perception/
  target_detection/
    detectors/
      base.py
      hough_detector.py
      blob_detector.py
      hsv_detector.py
      hybrid_detector.py
    configs/
      hough.yaml
      blob.yaml
      hsv.yaml
      hybrid.yaml
    scripts/
      run_detector.py
      benchmark.py
    tests/
      test_detectors.py
    requirements.txt
    README.md
    TUNING_NOTES.md
```

Generated annotated images are written to `perception\target_detection\outputs` by default. That directory is gitignored.

## Windows setup

From Windows PowerShell:

```powershell
cd C:\Users\User\Desktop\Subsystems_NEW
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r perception\target_detection\requirements.txt
```

If activation is blocked for the current PowerShell process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Tests

```powershell
cd C:\Users\User\Desktop\Subsystems_NEW
python -m pytest perception\target_detection\tests
```

The unit tests use deterministic synthetic OpenCV/NumPy images with known circle geometry. The uploaded `test-images` files are used for smoke testing only until annotations exist.

## Runner commands

Run the reproducible benchmark for all detectors:

```powershell
python perception\target_detection\scripts\benchmark.py
```

Run only the tuned Hough and blob detectors:

```powershell
python perception\target_detection\scripts\benchmark.py --detectors hough blob
```

Compare the teammate-owned HSV detector with the HSV-gated blob hybrid:

```powershell
python perception\target_detection\scripts\benchmark.py --detectors hsv hybrid
```

The benchmark uses images from `test-images`, COCO ground truth from `test-image-annotations\annotations.coco.json`, and the checked-in global configurations in `perception\target_detection\configs`. Annotated overlays are generated in `perception\target_detection\outputs\benchmark\annotated`.

Run Hough on one image and save an annotation:

```powershell
python perception\target_detection\scripts\run_detector.py --detector hough --input test-images\test_0.jpg --config perception\target_detection\configs\hough.yaml --output-dir perception\target_detection\outputs --save-annotated
```

Run blob detection on one image:

```powershell
python perception\target_detection\scripts\run_detector.py --detector blob --input test-images\test_0.jpg --config perception\target_detection\configs\blob.yaml --output-dir perception\target_detection\outputs --save-annotated
```

Run Hough on the complete image directory:

```powershell
python perception\target_detection\scripts\run_detector.py --detector hough --input test-images --config perception\target_detection\configs\hough.yaml --output-dir perception\target_detection\outputs --save-annotated
```

Run blob detection on the complete image directory:

```powershell
python perception\target_detection\scripts\run_detector.py --detector blob --input test-images --config perception\target_detection\configs\blob.yaml --output-dir perception\target_detection\outputs --save-annotated
```

Add `--json` to print one JSON object per processed image.

## Detector output format

Each detector accepts one already-loaded OpenCV BGR image as a NumPy array. File loading is intentionally outside the reusable detector logic.

```python
from perception.target_detection.detectors import BlobDetector, HoughCircleDetector

hough = HoughCircleDetector.from_config("perception/target_detection/configs/hough.yaml")
blob = BlobDetector.from_config("perception/target_detection/configs/blob.yaml")
result = hough.detect(bgr_image)
```

The returned `DetectorResult` contains:

- `method`: detector name, currently `hough` or `blob`
- `detections`: zero or more `Detection` dataclasses
- `timing`: `TimingInfo` in milliseconds
- `image_shape`: original BGR image shape

Each `Detection` contains:

- `method`: detector name
- `center_x`, `center_y`: center in pixel coordinates
- `radius`: circle radius or apparent radius
- `bbox`: `(x, y, width, height)` in pixel coordinates
- `score`: detector confidence when meaningful; `None` for Hough and usually `None` for SimpleBlobDetector

## Timing format

Timing uses `time.perf_counter()` and reports:

- `preprocessing_ms`: grayscale conversion, optional histogram equalization, and blur
- `detection_ms`: OpenCV detector execution and conversion into dataclasses
- `total_ms`: preprocessing plus detector work inside `detect()`

Image loading, annotation drawing, and output writing are excluded from detector latency.

## Configuration

`configs\hough.yaml` controls grayscale preprocessing, blur, Hough `dp`, minimum center distance, Canny high threshold `param1`, accumulator threshold `param2`, radius bounds, and duplicate suppression.

`configs\blob.yaml` controls grayscale preprocessing, blur, threshold range, area range, circularity, convexity, inertia, minimum blob spacing, and optional blob color filtering. The blob detector uses a permissive large-shape pass for oval or partially occluded targets and a stricter small-shape pass for small circular targets, then suppresses duplicates between the two passes. A global boundary-gradient check rejects soft circular highlights and lens flare while retaining blobs with a physical target-like edge. SimpleBlobDetector keypoint size is interpreted as blob diameter, so reported radius is `keypoint.size / 2`.

The checked-in parameters are one global configuration per detector; no filename, image-specific value, or ground-truth coordinate is used by detector logic. Hough and blob were tuned with controlled changes to smoothing, Hough voting/radius/spacing, and blob threshold/area/shape/spacing filters. The measured trials and remaining limitations are summarized in `TUNING_NOTES.md`.

## Benchmark metrics

The shared benchmark reports TP, FP, FN, precision, recall, F1, matched center error, matched bounding-box IoU, and preprocessing/detection/total latency. All detectors are evaluated against the same annotations and matching rules.

The hybrid reuses `configs\hsv.yaml` for its primary color mask. It applies connected-region and adaptive circle/ellipse geometry, uses the tuned blob detector as a tightly gated fallback for low-saturation white targets, and invokes a color-gated Hough fallback only when the faster stages find nothing. Hybrid-specific settings live in `configs\hybrid.yaml`.

## Known failure cases

- Large circular background objects can be accepted as Hough targets.
- Non-circular distractors may still create circle-like edges or bright blobs.
- Small or distant targets may be filtered by radius or area thresholds.
- Partial target visibility can reduce Hough votes or blob circularity.
- Lighting and contrast changes can alter grayscale threshold behavior.
- Hough can return duplicate circles around the same target despite suppression.
- Multiple overlapping targets can merge into one blob or confuse circle fitting.
- Blob filters can reject real targets if circularity, area, color, convexity, or inertia assumptions are too strict.
- Blob detection can accept similarly colored or similarly shaped clutter.
