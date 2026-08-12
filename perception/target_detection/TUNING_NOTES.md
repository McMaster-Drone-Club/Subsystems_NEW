# Hough and Blob Tuning Notes

All trials used one global configuration per detector on the same 11-image, 17-target benchmark. Because parameters were selected on this dataset, these numbers are tuning-set results rather than held-out performance.

## Hough

Starting configuration: median blur 5; `dp=1.2`; `min_dist=90`; `param1=120`; `param2=28`; radius 10–80 px. Baseline: TP 10, FP 941, FN 7, precision 0.0105, recall 0.5882, F1 0.0207, mean center error 33.942 px, mean IoU 0.400, mean total latency 1068.495 ms.

Important trials expanded the radius range to cover labeled radii (approximately 11–102 px), raised `param2` to reject weak forest-texture circles, increased median smoothing, and tested larger center spacing. Strong smoothing and higher vote thresholds reduced clutter but either missed more benchmark targets or failed deterministic synthetic-circle tests; expanding the radius range also worsened benchmark TP/FP. Even `param2=30`, which modestly reduced benchmark false positives without changing benchmark recall, failed the small-circle regression test. No tested change improved the benchmark while preserving required detector behavior, so the starting configuration was retained.

Final configuration (retained): median blur 5; `dp=1.2`; `min_dist=90`; `param1=120`; `param2=28`; radius 10–80 px; original duplicate suppression retained.

Final metrics (Hough/blob-only validation run): TP 10, FP 941, FN 7, precision 0.0105, recall 0.5882, F1 0.0207, mean center error 33.942 px, mean IoU 0.400; mean preprocessing 3.159 ms, detection 1045.761 ms, and total 1048.923 ms. These are effectively the baseline accuracy results because the validated starting configuration remained strongest.

## Blob

Starting configuration: Gaussian blur 3; thresholds 10–240 step 10; spacing 20; bright-blob filter enabled; area 120–25,000; circularity 0.65; convexity/inertia disabled. Baseline: TP 2, FP 253, FN 15, precision 0.0078, recall 0.1176, F1 0.0147, mean center error 31.591 px, mean IoU 0.524, mean total latency 67.000 ms.

Important trials tested bright, dark, and polarity-neutral blobs; area and circularity ranges; convexity/inertia; threshold step; spacing; and stronger blur. Fixed brightness polarity cannot cover disks of widely varying colors. Disabling it improved recall; minimum area 300, circularity 0.55, and spacing 50 reduced texture detections while retaining the four measured true detections. Higher circularity/inertia/spacing or threshold step 20 removed valid targets.

Final configuration: Gaussian blur 3; thresholds 10–240 step 10; spacing 50; color filter disabled; area 300–40,000; circularity 0.55; convexity/inertia disabled.

Final metrics (Hough/blob-only validation run): TP 4, FP 148, FN 13, precision 0.0263, recall 0.2353, F1 0.0473, mean center error 16.302 px, mean IoU 0.628; mean preprocessing 1.145 ms, detection 62.214 ms, and total 63.363 ms.

## Remaining failure modes

Forest texture still produces many circle/blob candidates, especially in `6.png` for blob and the zero-target `10.png` for Hough. Blob correctly detects the small target in `2.png` but adds two clutter blobs; it detects targets in `7.png` and two of six in `9.png`. Hough detects four of six targets in `9.png` but produces 88 false positives there. Small/distant targets (`2.png` for Hough and several in `9.png`), the partial/non-circular target (`6.png`), shadows over disks (`5.png`), and variable grayscale brightness remain difficult. Hough still produces many clutter circles and inaccurate sizes/centers; blob still misses most colored disks because their grayscale polarity is inconsistent. No obvious duplicate match was counted as a true positive because benchmark matching is one-to-one, but many nearby clutter detections remain visually overlapping.
