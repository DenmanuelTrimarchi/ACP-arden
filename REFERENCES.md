# References

Sources for the pretrained models, benchmark datasets, libraries and evaluation
methods used in this project. Model checksums identify the evaluated weight
files; the source and model-weight licences are recorded separately.

## Pretrained models

The YuNet and SFace weights are stored outside the repository. The loader
checks each file against its pinned SHA-256 digest and refuses a mismatch.

**YuNet — face detection**

```text
Title: YuNet: A Tiny Millisecond-level Face Detector
Author: Wu, W., Peng, H. and Yu, S., Machine Intelligence Research, 20(5), pp. 656-665
Date: 2023
Availability: https://doi.org/10.1007/s11633-023-1423-y
GitHub: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
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
GitHub: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
File: face_recognition_sface_2021dec.onnx
Licence: Apache-2.0
SHA-256: 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
```

Both weight files are distributed through the OpenCV Zoo model repository
(https://github.com/opencv/opencv_zoo), under `models/face_detection_yunet/`
and `models/face_recognition_sface/` respectively.

## Datasets

Dataset images are not redistributed. The experiments use the authors'
published data under the applicable terms. Archive checksums in section 2 of
`ACP_arden.py` identify the evaluated copies.

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

BFW is an optional external benchmark used by the supplementary open-set
experiment. It is not redistributed here and is obtained by the researcher from
the official project under its own terms.

```text
Title: Face Recognition: Too Bias, or Not Too Bias?
Author: Robinson, J.P., Livitz, G., Henon, Y., Qin, C., Fu, Y. and Timoner, S., Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
Date: 2020
Availability: https://doi.org/10.1109/CVPRW50498.2020.00008
```

```text
Title: Balanced Faces in the Wild (BFW), source data, metadata table and benchmarks
Author: Robinson, J.P. and contributors, visionjo/facerec-bias-bfw
Date: 2020
Availability: https://github.com/visionjo/facerec-bias-bfw
```

BFW is published with verification and bias-analysis protocols. It does **not**
publish an open-set identification protocol. The identity-disjoint open-set
partition used by Experiment 6 is constructed by this project from the official
data and metadata, and is identified in every artefact by its own protocol
version string so it can never be mistaken for an upstream standard.

## Evaluation methods

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

The open-set identification metrics follow the definitions used by NIST's 1:N
evaluation: FPIR as the proportion of non-mated searches returning at least one
candidate above threshold, and FNIR at rank *k* as the proportion of mated
searches that fail to return the correct mate within rank *k* above threshold.
Section 20 implements these metric definitions.

```text
Title: Face Recognition Technology Evaluation (FRTE) 1:N Identification
Author: National Institute of Standards and Technology (NIST)
Date: 2025
Availability: https://pages.nist.gov/frvt/html/frvt1N.html
```

The reporting framework — separating the decision policy from the error rates,
quoting every rate with the coverage it was measured over, and declaring
operating points before testing — follows the principles of the international
biometric performance-testing standard.

```text
Title: ISO/IEC 19795-1:2021 Information technology — Biometric performance testing and reporting — Part 1: Principles and framework
Author: International Organization for Standardization
Date: 2021
Availability: https://www.iso.org/standard/73515.html
```

The cluster bootstrap used for the confidence intervals resamples identities
rather than images, because multiple probes can belong to one identity and
resampling images independently would understate the intervals.

```text
Title: Bootstrap Methods: Another Look at the Jackknife
Author: Efron, B., The Annals of Statistics, 7(1), pp. 1-26
Date: 1979
Availability: https://doi.org/10.1214/aos/1176344552
```

The review classifier uses scikit-learn's logistic-regression API with
project-specific search features, identity-disjoint splitting, threshold
freezing and held-out evaluation.

```text
Title: The Regression Analysis of Binary Sequences
Author: Cox, D.R., Journal of the Royal Statistical Society, Series B, 20(2), pp. 215-242
Date: 1958
Availability: https://doi.org/10.1111/j.2517-6161.1958.tb00292.x
```

```text
Title: Scikit-learn: Machine Learning in Python (LogisticRegression and StandardScaler APIs)
Author: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E., Journal of Machine Learning Research, 12, pp. 2825-2830
Date: 2011
Availability: https://jmlr.org/papers/v12/pedregosa11a.html
```

Figures are drawn with matplotlib through its documented public API.

```text
Title: Matplotlib: A 2D Graphics Environment
Author: Hunter, J.D., Computing in Science and Engineering, 9(3), pp. 90-95
Date: 2007
Availability: https://doi.org/10.1109/MCSE.2007.55
```

### InsightFace components

**Project:** InsightFace (deepinsight/insightface). **Model package:**
`buffalo_l`. **Components evaluated:** SCRFD detection (`det_10g.onnx`) and
ArcFace recognition (`w600k_r50.onnx`), 512-dimensional embeddings.

**Source-code licence:** the InsightFace source code is published under the MIT
licence. **The pretrained weights are governed by separate terms** permitting
non-commercial research use; the MIT licence covering the source code does not
automatically extend to every pretrained weight file, and this project does not
treat it as doing so.

**Purpose in this project:** local, non-commercial benchmark evaluation only, as
part of an MSc academic research artefact. There is no commercial deployment,
paid service, commercial decision-making or real-user exposure.

The project uses the documented public API and pretrained artefacts. The
models were created and trained externally by the InsightFace project and
remain frozen during evaluation.

**Redistribution:** none. Weight files are stored in private local research
storage, are excluded from Git and from any release, and are never downloaded
automatically by the programme.

The SCRFD + ArcFace comparison is recorded in
`results/aggregate/pipeline_comparison_metrics.json`, including the evaluated
models, frozen thresholds, software versions and held-out outcomes.

```text
Title: InsightFace: 2D and 3D Face Analysis Project
Author: InsightFace contributors (deepinsight/insightface)
Date: 2021 onwards; buffalo_l model pack released 2021
Availability: https://github.com/deepinsight/insightface
```

```text
Title: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
Author: Deng, J., Guo, J., Xue, N. and Zafeiriou, S., Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
Date: 2019
Availability: https://doi.org/10.1109/CVPR.2019.00482
GitHub: https://github.com/deepinsight/insightface
```

```text
Title: Sample and Computation Redistribution for Efficient Face Detection (SCRFD)
Author: Guo, J., Deng, J., Lattas, A. and Zafeiriou, S., International Conference on Learning Representations (ICLR)
Date: 2022
Availability: https://arxiv.org/abs/2105.04714
GitHub: https://github.com/deepinsight/insightface
```

## Libraries

Used through their documented public APIs.

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

Keyed opaque identifiers use HMAC as specified in RFC 2104, through Python's
standard-library API.

```text
Title: HMAC: Keyed-Hashing for Message Authentication (RFC 2104)
Author: Krawczyk, H., Bellare, M. and Canetti, R., Internet Engineering Task Force
Date: February 1997
Availability: https://www.rfc-editor.org/rfc/rfc2104
```

```text
Title: hmac — Keyed-Hashing for Message Authentication (Python standard library documentation)
Author: Python Software Foundation
Date: 2025
Availability: https://docs.python.org/3/library/hmac.html
```

## Benchmark and identification references

These sources describe the verification, identification and subgroup evaluation
methods used for context. Project measurements are recorded separately in
`results/aggregate/`.

- NIST, *Face Recognition Technology Evaluation (FRTE), 1:N Identification*,
  [evaluation definitions and resources](https://pages.nist.gov/frvt/html/frvt1N.html).
- Robinson, J.P., Livitz, G., Henon, Y., Qin, C., Fu, Y. and Timoner, S. (2020),
  *Face Recognition: Too Bias, or Not Too Bias?*, CVPR Workshops,
  [primary paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Robinson_Face_Recognition_Too_Bias_or_Not_Too_Bias_CVPRW_2020_paper.pdf).
- Huang, G.B., Ramesh, M., Berg, T. and Learned-Miller, E. (2007),
  *Labeled Faces in the Wild: A Database for Studying Face Recognition in
  Unconstrained Environments*, Technical Report 07-49,
  [author-hosted copy](https://people.cs.umass.edu/~elm/papers/lfw.pdf).
- Zheng, T. and Deng, W. (2018), *Cross-Pose LFW: A Database for Studying
  Cross-Pose Face Recognition in Unconstrained Environments*,
  [author-hosted paper](https://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf).
- Deng, J., Guo, J., Xue, N. and Zafeiriou, S. (2019), *ArcFace: Additive
  Angular Margin Loss for Deep Face Recognition*, CVPR, pp. 4690–4699,
  [published paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html).

## Comparative analysis methods

The final LFW evaluation implements the official ten-fold pair protocol,
fitting an accuracy-maximising threshold on nine folds and evaluating the
remaining fold. See the LFW technical report by Huang et al. (2007),
[Labelled Faces in the Wild](https://people.cs.umass.edu/~elm/papers/lfw.pdf).
The implementation preserves protocol positions when extraction fails and
reports both coverage and conditional performance. The local overlap audit
explains why the development files cannot define an independent final LFW test.

Paired comparisons use the project's existing identity-cluster percentile
bootstrap, now sharing the same identity draws between methods. Resampling is
stratified by probe role and BFW subgroup. Common-success comparisons and
full-protocol end-to-end comparisons estimate different quantities.

The supplementary zero-event bound inverts the binomial zero-event probability:
`P(no identity with a false referral) = (1-p)^n`, giving a one-sided 95% upper
bound `1 - 0.05**(1/n)`. Here trials are independent, exchangeable identity
clusters; dependence among photographs within an identity is unrestricted.
The bound concerns any referral within an identity's intended probe set, not
the conditional scored-probe FPIR. Gallery, population and calibration
uncertainty are outside its scope.
