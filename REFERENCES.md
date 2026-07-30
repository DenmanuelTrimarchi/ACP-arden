# Code attribution and reference register

This register identifies every external source this artefact depends on, and
states precisely what the dependency is: an **artefact** consumed (a pretrained
model, a benchmark dataset), an **API** called, or a **published method** whose
standard definition is implemented here.

## Outcome of the review

Every line of `ACP_arden.py` and of the test suite was reviewed for externally
adapted code. **No file contains code copied or materially adapted from an
external source.** Calling a library's documented public API is use of that
library, not adaptation of its source; implementing a standard statistical
definition from its published description is not adaptation of any particular
implementation.

Attribution headers are nonetheless recorded in the source at each point where
an external model, dataset, API or published method enters the pipeline, so a
reader can trace the provenance of every component without consulting this file.
They use the four-field form:

```python
##############
# Title: Short title of the adapted code or method
# Author: Author name, organisation or project
# Date: Year or full date where known
# Availability: Stable URL, DOI or repository location
##############
```

## Reviewed areas

The table lists only areas a reviewer could reasonably question on provenance
grounds. Every one was confirmed to be original project code.

| Code area | Section | External source | Adaptation status | Licence checked |
|---|---|---|---|---|
| `YuNetDetector` wrapper around `cv2.FaceDetectorYN` | 6 | None — OpenCV's documented public API only; the YuNet weights are an external artefact | Original | MIT (weights) |
| `SFaceEmbedder` wrapper around `cv2.FaceRecognizerSF` | 7 | None — OpenCV's documented public API only; the SFace weights are an external artefact | Original | Apache-2.0 (weights) |
| `l2_normalize`, `cosine_similarity` | 8 | None — standard vector algebra written independently | Original | Not applicable |
| LFW and CPLFW pair-file parsing | 9 | None — the *file formats* are defined by the dataset authors; the parsers are original | Original | Not applicable |
| `roc_auc` rank-based implementation | 10 | None — the ROC-AUC/Wilcoxon–Mann–Whitney rank identity is a standard statistical result, implemented here without reference to another codebase | Original | Not applicable |
| `equal_error_rate`, `roc_points`, `confusion_matrix` | 10 | None — standard definitions implemented independently, deliberately avoiding a scikit-learn dependency | Original | Not applicable |
| `percentile` | 10 | None — linear-interpolation percentile, matching NumPy's documented default method | Original | Not applicable |
| Two-stage calibration and the frozen-threshold guard | 11 | None — this project's own methodological design | Original | Not applicable |
| Bounded image loading and EXIF orientation handling | 5 | None — uses Pillow's documented `ImageOps.exif_transpose`; the bounds and failure taxonomy are project-specific | Original | Not applicable |
| Deterministic 1:N gallery construction | 13 | None — this project's own experimental design | Original | Not applicable |
| Streamlit review page | 16 | None — standard Streamlit widgets composed for this project | Original | Not applicable |

## Models (external artefacts, not code)

Neither file is committed to this repository. Each is verified against a pinned
SHA-256 digest before it is loaded, and refused rather than used on a mismatch.

**YuNet — face detection**

```text
Title: YuNet: A Tiny Millisecond-level Face Detector
Author: Wu, W., Peng, H. and Yu, S., Machine Intelligence Research, 20(5), pp. 656-665
Date: 2023
Availability: https://doi.org/10.1007/s11633-023-1423-y
File: face_detection_yunet_2023mar.onnx
Licence: MIT
SHA-256: 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
```

**SFace — face embedding**

```text
Title: SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition
Author: Zhong, Y., Deng, W., Hu, J., Zhao, D., Li, X. and Wen, D., IEEE Transactions on Image Processing, 30, pp. 2587-2598
Date: 2021
Availability: https://doi.org/10.1109/TIP.2020.3048632
File: face_recognition_sface_2021dec.onnx
Licence: Apache-2.0
SHA-256: 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
```

Both weight files are distributed through the OpenCV Zoo model repository
(https://github.com/opencv/opencv_zoo), under `models/face_detection_yunet/`
and `models/face_recognition_sface/` respectively.

## Datasets (external inputs, not code)

Neither dataset is redistributed here. Both are used under their authors'
published terms, and the archive checksums recorded in section 2 identify the
exact copies this project evaluated.

```text
Title: Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments
Author: Huang, G.B., Ramesh, M., Berg, T. and Learned-Miller, E., University of Massachusetts, Amherst, Technical Report 07-49
Date: October 2007
Availability: http://vis-www.cs.umass.edu/lfw/lfw.pdf
```

```text
Title: Unsupervised Joint Alignment of Complex Images
Author: Huang, G.B., Jain, V. and Learned-Miller, E., Proceedings of the IEEE International Conference on Computer Vision (ICCV)
Date: 2007
Availability: http://vis-www.cs.umass.edu/lfw/
```

The second reference describes the "funnelled" alignment procedure that
produced the `lfw_funneled` image set evaluated here, which is a distinct image
set from the original unaligned LFW distribution.

```text
Title: Cross-Pose LFW: A Database for Studying Cross-Pose Face Recognition in Unconstrained Environments
Author: Zheng, T. and Deng, W., Beijing University of Posts and Telecommunications, Technical Report 18-01
Date: February 2018
Availability: http://www.whdeng.cn/cplfw/
```

CPLFW ships two non-interchangeable image sets in one archive. The published
results here use the authors' raw, unconstrained images (`images.rar`), never
the separately pre-cropped copy (`cp-aligned.zip`), and every CPLFW artefact
records which variant it scored.

## Published methods implemented independently

```text
Title: The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve
Author: Hanley, J.A. and McNeil, B.J., Radiology, 143(1), pp. 29-36
Date: 1982
Availability: https://doi.org/10.1148/radiology.143.1.7063747
```

Establishes that the area under the ROC curve equals the probability that a
randomly chosen positive is ranked above a randomly chosen negative — the
quantity estimated by the Wilcoxon rank-sum statistic. `roc_auc` in section 10
computes the area through that rank identity rather than by integrating the
curve, which is exact under ties and needs no additional dependency.

## Libraries

Used through their documented public APIs. No library's source is adapted into
this file.

```text
Title: The OpenCV Library
Author: Bradski, G., Dr. Dobb's Journal of Software Tools, 25(11), pp. 120-125
Date: 2000
Availability: https://opencv.org/
```

```text
Title: Array programming with NumPy
Author: Harris, C.R., Millman, K.J., van der Walt, S.J. et al., Nature, 585(7825), pp. 357-362
Date: 2020
Availability: https://doi.org/10.1038/s41586-020-2649-2
```

```text
Title: Pillow (the friendly PIL fork)
Author: Clark, A. and contributors
Date: 2010 onwards
Availability: https://python-pillow.github.io/
```

```text
Title: Streamlit
Author: Snowflake Inc. and contributors
Date: 2019 onwards
Availability: https://streamlit.io/
```

## A note on spelling

Prose and comments throughout this project use British English. Cited titles
are reproduced verbatim, including where the original uses American spelling —
most visibly *Labeled Faces in the Wild*, which is the dataset's published
title and is not silently Britishised here. Citation fidelity takes precedence
over house style: altering a title would misquote its authors.

## Maintenance rule

If code is later adapted from an external source, add a four-field header
immediately above the adapted block and a matching row to the table above. Do
not add a header where the origin cannot be verified: record the uncertainty
instead and raise it for review. A header is a claim about provenance, and an
unverifiable claim is worse than an acknowledged gap.
