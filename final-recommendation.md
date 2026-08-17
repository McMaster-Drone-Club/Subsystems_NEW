# Detection Method Recommendation
**Recommendation:** HSV Detector


The tables below show each detection method's performance on the training set and the test set.

## Accuracy Data

**Training set (20 images):**

| Metric | Hough | Blob | HSV | Hybrid |
|---|---|---|---|---|
| Precision | 0.002 | 0.067 | 0.824 | 0.941 |
| Recall | 0.118 | 0.176 | 0.824 | 0.941 |
| F1 | 0.004 | 0.097 | 0.824 | 0.941 |
| False positives | 949 | 42 | 3 | 1 |

**Test set (5 images):**

| Metric | Hough | Blob | HSV | Hybrid |
|---|---|---|---|---|
| Precision | 0.002 | 0.062 | 0.273 | 0.286 |
| Recall | 0.167 | 0.167 | 0.500 | 0.333 |
| F1 | 0.004 | 0.091 | 0.353 | 0.308 |
| False positives | 509 | 15 | 8 | 5 |

### Latency

| Method | Train mean | Test mean |
|---|---|---|
| Hough | 1162.5 ms | 1615.7 ms |
| Blob | 389.1 ms | 472.6 ms |
| HSV | 31.9 ms | 34.6 ms |
| Hybrid | 623.2 ms | 880.2 ms |

## Unviable candidates

The Blob detection method has poor accuracy, measured through its precision, recall and F1 scores, and generates a fair number of false positives as well. This is evident in both the training and test sets. Despite considerable effort in tuning, this detection method's performance did not improve. Thus, we will not be considering this method as a viable option.

The Hough method performs worse than the Blob method across all categories, and generates false positives on the order of 100s per image. In addition to the poor performance in identifying targets, this method is also incredibly computation intensive and slow, making it impossible to use in a real-time setting.

## Viable candidates

The two potential viable candidates for detecting circles are the HSV detector and the Hybrid detector. On the training data the hybrid method performs near perfect, but this performance significantly diminishes on the test set. Although the HSV detector's performance also decreases on the test set, it is a less drastic decrease in performance compared to the Hybrid detector. While the test set is small, the consistent direction of the gap between HSV and Hybrid is indicative of each detector's performance in real-world scenarios.

Although the HSV detector's performance is still not ideal, we believe it can still be improved by providing a narrower range of colours for the expected targets. This can improve detection quality in two ways. Firstly, this will reduce false positives currently picked up due to the wide colour range currently accepted by the thresholding masks. Secondly, a tighter color range lets us calibrate the colour thresholds more precisely, so we can afford to be more permissive with each of the target colours without picking up unrelated background colors, thereby reducing false negatives on legitimate targets.

In terms of latency, the HSV method is ~20 to 25 times faster than the Hybrid method, making it a highly realistic choice for our time-constrained applications.

## Tuning complexity

The HSV method has a small number of parameters to tune compared to the Hybrid method. For the HSV, the parameters include colour ranges, image blur parameters, contrast improvement parameters and morphology parameters. In comparison the Hybrid method requires all parameters needed for the HSV method, as well as all parameters used to tune the Blob and Hough methods. Therefore, the Hybrid method is much harder to tune than the HSV method.

## Jetson compatibility

The HSV method's ~35 ms max leaves enough computational time and resources for other tasks. The Hybrid method's worst-case ~2.4 s spikes are scene-triggered and unpredictable, making them unsafe for a real-time loop.

## Conclusion

The Hybrid method's performance on the training set does not remain the same on test data, while the HSV method's performance somewhat remains stable. In addition, the HSV method's main weakness (false positives and negatives due to a wide colour range) is a known and addressable tuning problem. We recommend the HSV method as the primary detection method.