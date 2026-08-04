# Target Detection Experiments

## Purpose and scope

This package contains the Hough-circle and blob-detector portion of the colored-target benchmark work for McMaster-Drone-Club/VM_Scripts_NEW issue #4. It does not implement HSV/color segmentation, final benchmark metrics, annotations, or the shared batch benchmark script.

## Folder structure

```text
perception/
  target_detection/
    detectors/
      base.py
      hough_detector.py
      blob_detector.py
    configs/
      hough.yaml
      blob.yaml
    scripts/
      run_detector.py
    tests/
      test_detectors.py
    requirements.txt
    README.md
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

Run Hough on one image and save an annotation:

```powershell
python perception\target_detection\scripts\run_detector.py --detector hough --input test-images\0.png --config perception\target_detection\configs\hough.yaml --output-dir perception\target_detection\outputs --save-annotated
```

Run blob detection on one image:

```powershell
python perception\target_detection\scripts\run_detector.py --detector blob --input test-images\0.png --config perception\target_detection\configs\blob.yaml --output-dir perception\target_detection\outputs --save-annotated
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

`configs\blob.yaml` controls grayscale preprocessing, blur, threshold range, area range, circularity, convexity, inertia, minimum blob spacing, and optional blob color filtering. SimpleBlobDetector keypoint size is interpreted as blob diameter, so reported radius is `keypoint.size / 2`.

The current configs are baseline shared parameter sets for the complete `test-images` folder. They are not tuned per image.

## Integration with teammate benchmark

The future benchmark script can keep its own image loading, annotations, metric calculations, and reporting. It can import either detector, instantiate from YAML, and call `.detect(bgr_image)`. If a different benchmark contract is introduced later, `detectors\base.py` is intentionally isolated so only the adapter layer should need changes.

No completed annotation files were present when this implementation was created, so accuracy, false-positive rate, center error, and size error are not calculated here.

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