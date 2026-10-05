# Evidence addressing supervisor feedback

**Research objective:** A deployment-oriented evaluation methodology for duplicate face detection that incorporates unprocessed images, human-review workload and computational cost into the comparison of detector–recogniser pipelines.

This response is generated from the saved results. The main research report contains the experimental results and their limitations; the sections below make each feedback point explicit.

## 1. Threshold calibration and validation across datasets

```text
HOW EVERY THRESHOLD WAS CALIBRATED AND VALIDATED

Each combination has its own threshold, because SFace and ArcFace give
similarity scores on different scales. Every threshold was fitted on
development data, frozen, and only then applied to test data. No threshold is
shared between combinations or between tasks, with one deliberate exception:
BFW layers 1 and 2 and the LFW gallery reuse the YuNet + SFace one to one
threshold as a control, to show why a borrowed threshold fails.

  Task and dataset                        Calibration data                                   Selection rule                                 Validation data               Frozen threshold
  --------------------------------------  -------------------------------------------------  ---------------------------------------------  ----------------------------  ----------------------------------
  LFW one to one, YuNet + SFace           9 folds of LFW pairs.txt                           highest accuracy on those 9 folds              each held out fold in turn    0.318371 to 0.335979 over 10 folds
  LFW one to one, SCRFD + ArcFace         9 folds of LFW pairs.txt                           highest accuracy on those 9 folds              each held out fold in turn    0.272602 to 0.274292 over 10 folds
  CPLFW one to one, YuNet + SFace         LFW pairsDevTrain.txt                              highest balanced accuracy on pairsDevTest.txt  CPLFW pairs_CPLFW.txt         0.363012
  CPLFW one to one, SCRFD + ArcFace       LFW pairsDevTrain.txt                              highest balanced accuracy on pairsDevTest.txt  CPLFW pairs_CPLFW.txt         0.329049
  BFW gallery search, YuNet + SFace       BFW development identities                         highest detection with FPIR at most 0.30%      BFW held out test identities  0.477118
  BFW gallery search, SCRFD + SFace       BFW development identities                         highest detection with FPIR at most 0.30%      BFW held out test identities  0.471432
  BFW gallery search, YuNet + ArcFace     BFW development identities                         highest detection with FPIR at most 0.30%      BFW held out test identities  0.391290
  BFW gallery search, SCRFD + ArcFace     BFW development identities                         highest detection with FPIR at most 0.30%      BFW held out test identities  0.393958
  BFW review classifier, YuNet + SFace    BFW development identities kept back from fitting  highest detection with FPIR at most 0.30%      BFW held out test identities  0.571467 (probability)
  BFW review classifier, SCRFD + ArcFace  BFW development identities kept back from fitting  highest detection with FPIR at most 0.30%      BFW held out test identities  0.845732 (probability)

Development FPIR against test FPIR: YuNet + SFace 0.28% against 0.52%;
SCRFD + SFace 0.27% against 0.64%; YuNet + ArcFace 0.10% against 0.17%;
SCRFD + ArcFace 0.13% against 0.13%.

Test FPIR rose above the 0.30% development target for YuNet + SFace and
SCRFD + SFace. The target guides the choice of threshold, but it does not
guarantee the rate on new identities.
```

## 2. Dataset limitations and generalisability

```text
WHAT THESE DATASETS CAN AND CANNOT SHOW

LFW was collected from news photographs of public figures. CPLFW keeps the LFW
identities, so both verification datasets share one population. BFW also shows
public figures, and its gallery search split was designed by this project
rather than published with the dataset. None of the datasets was validated
here as representative of dating uploads, including selfies, filters, group
photographs and edited images. Some identities may appear in the web
collections used to train the pretrained models, which could make the results
look better than they would be for unseen people.

The gallery holds only 200 identities, so a service with many more profiles
would offer more chances of a false match.

So these results show how the combinations compare under the same conditions.
They cannot show how any combination would perform on a real dating service.
```

## 3. Accuracy, latency, storage and review workload

```text
WHAT EACH COMBINATION COSTS AND WHAT IT BUYS

  Combination      End to end detection (95% CI)  False reviews per 1,000 (95% CI)  Detection and embedding, ms  Model storage, MB
  ---------------  -----------------------------  --------------------------------  ---------------------------  -----------------
  YuNet + SFace    87.20% (84.30 to 90.00)        5.25 (1.77 to 9.11)               20.79                        37.1
  SCRFD + SFace    94.90% (93.30 to 96.30)        6.36 (2.68 to 11.04)              44.32 (estimated)            53.0
  YuNet + ArcFace  91.60% (89.40 to 93.60)        1.75 (0.35 to 3.49)               55.68 (estimated)            166.5
  SCRFD + ArcFace  96.70% (95.40 to 97.90)        1.34 (0.00 to 3.69)               79.21                        182.4

End to end detection counts every intended known duplicate, including
photographs that could not be processed. False reviews are counted per 1,000
new profile photographs that were processed. Times are the mean detection plus
the mean embedding per image from the pipeline comparison run. The two crossed
combinations were not timed in that run, so their time adds one pipeline's
detection to the other's embedding and is marked (estimated). Storage is the
size of the two weight files. The saved MB fields use binary megabytes (MiB);
gallery templates and runtime memory are not included.

Starting from YuNet + SFace, swapping YuNet for SCRFD adds 23.52 ms per image
and 15.9 MB of storage, and raises end to end detection from 87.20% to 94.90%.

Swapping SFace for ArcFace adds 34.89 ms per image and 129.4 MB of storage,
and cuts false reviews from 5.25 to 1.75 per 1,000 new profiles.

YuNet + SFace is the cheapest combination and SCRFD + ArcFace detects the most
duplicates. For 100,000 uploads, YuNet + SFace would need about 36 minutes of
processing, and SCRFD + ArcFace about 2 hours 15 minutes. These use the
complete time per image measured on the machine used for these runs, which
also counts loading the photograph. For every 100,000 new profiles processed,
SCRFD + ArcFace would raise roughly 390 fewer false reviews. Of every 1,000
known duplicates, it would find about 95 more. Reviewer cost and the harm of a
missed duplicate were not measured, so these figures cannot say which
combination is worth its cost.
```

## 4. Detector settings and processing failures

```text
HOW DETECTOR SETTINGS SHAPE THE FAILURES

  Detector  Confidence threshold  Input size                  Dominant failure and counts
  --------  --------------------  --------------------------  ----------------------------------------------------------------------
  YuNet     0.9                   each photograph's own size  zero faces: CPLFW 2,321 pairs, BFW 189 photographs
  SCRFD     0.5                   320 by 320 pixels           multiple faces: LFW 1,794 pairs, CPLFW 1,008 pairs, BFW 12 photographs

On LFW, YuNet lost 540 pairs to multiple faces and SCRFD lost 1,794. On CPLFW,
YuNet lost 2,321 pairs to zero faces and SCRFD lost 57.

The one face rule directly explains why zero or multiple detections produce no
comparison. Within a fixed detector, changing the confidence cutoff changes
which candidates can pass; lowering it can admit both difficult faces and
unwanted detections. YuNet and SCRFD scores are not calibrated on a common
scale, so 0.9 versus 0.5 alone cannot explain their different failure counts.
SCRFD resizes while preserving aspect ratio and pads to a 320 by 320 pixel
canvas. The extra LFW detections were not checked by hand, so they may be
background faces or false detections.

Controlled settings evidence

The experiment changes one setting within a detector on identical images, with fixed weights, NMS and face acceptance rule. The preset comparisons below connect processing failures to unresolved workload and detector time. All variants and transition counts appear in the separate detector settings report; none was chosen to replace a main pipeline.

  Cohort           Setting               Coverage change, points (95% CI)  Unprocessed per 1,000  Detector mean ms
  ---------------  --------------------  --------------------------------  ---------------------  ----------------
  BFW development  YuNet confidence 0.8  +3.91 (+1.95 to +6.25)            39.1 to 0.0            2.50 to 2.50
  BFW development  YuNet canvas 320      +3.12 (+1.17 to +5.08)            39.1 to 7.8            2.50 to 5.00
  BFW development  SCRFD confidence 0.7  -0.39 (-1.56 to +0.78)            7.8 to 11.7            26.16 to 26.20
  BFW development  SCRFD canvas 640      -98.44 (-100.00 to -96.48)        7.8 to 992.2           26.16 to 93.35
  BFW test         YuNet confidence 0.8  +3.52 (+1.56 to +5.86)            39.1 to 3.9            2.50 to 2.48
  BFW test         YuNet canvas 320      +2.73 (+0.78 to +5.08)            39.1 to 11.7           2.50 to 5.06
  BFW test         SCRFD confidence 0.7  -2.34 (-4.69 to -0.39)            0.0 to 23.4            25.50 to 25.32
  BFW test         SCRFD canvas 640      -94.53 (-97.27 to -91.80)         0.0 to 945.3           25.50 to 91.96
  CPLFW raw        YuNet confidence 0.8  +21.48 (+15.62 to +26.95)         281.2 to 66.4          3.29 to 3.32
  CPLFW raw        YuNet canvas 320      +8.20 (+4.30 to +12.11)           281.2 to 199.2         3.29 to 5.11
  CPLFW raw        SCRFD confidence 0.7  -16.80 (-22.27 to -10.94)         82.0 to 250.0          25.90 to 25.87
  CPLFW raw        SCRFD canvas 640      -1.17 (-3.52 to +0.78)            82.0 to 93.8           25.90 to 93.28
  LFW development  YuNet confidence 0.8  -8.98 (-12.89 to -5.47)           54.7 to 144.5          3.21 to 3.24
  LFW development  YuNet canvas 320      -0.39 (-1.56 to +0.78)            54.7 to 58.6           3.21 to 4.95
  LFW development  SCRFD confidence 0.7  +8.98 (+5.46 to +12.89)           207.0 to 117.2         25.43 to 25.43
  LFW development  SCRFD canvas 640      -0.39 (-3.12 to +2.34)            207.0 to 210.9         25.43 to 91.27

Cohorts use a preset equal number of images per sampled identity. BFW development and test use disjoint identities. The interventions establish processing effects on these sampled images; they do not prove the correctness of detections or the cause of the entire original dataset gap. Intervals use paired identity resampling and are exploratory, without adjustment for multiple comparisons. These are image counts, not failed pair counts.

One detected face is eligibility for recognition, not identification accuracy. Unprocessed images are unresolved cases, not false duplicate referrals. No annotation establishes whether extra detections are real background faces. Model storage is fixed within each detector; only the configured processing and detector runtime change. Recognition must be recalibrated and evaluated before any alternative setting could replace the frozen main pipeline.
```

## 5. Statistical support for the main conclusion

```text
IS THE BEST COMBINATION REALLY BETTER?

SCRFD + ArcFace has the highest end to end detection, 96.70%, so it is
compared below with each other combination. Both sides of each comparison use
the same identities, and each 95% interval comes from resampling those
identities 2,000 times. Differences are in percentage points. A difference is
supported when its whole interval lies on one side of zero, and not
distinguishable otherwise.

  Compared with    Gain in end to end detection (95% CI)  Verdict    Reduction in FPIR (95% CI)  Verdict
  ---------------  -------------------------------------  ---------  --------------------------  -------------------
  YuNet + SFace    9.50 (7.30 to 12.10)                   supported  0.39 (0.11 to 0.71)         supported
  SCRFD + SFace    1.80 (1.00 to 2.80)                    supported  0.50 (0.23 to 0.84)         supported
  YuNet + ArcFace  5.10 (3.60 to 6.90)                    supported  0.04 (-0.13 to 0.21)        not distinguishable

The gain in end to end detection is supported against YuNet + SFace,
SCRFD + SFace and YuNet + ArcFace.

The reduction in FPIR is supported against YuNet + SFace and SCRFD + SFace,
and not distinguishable against YuNet + ArcFace.

These intervals are exploratory and are not adjusted for making several
comparisons. A zero event interval, from a run with no false reviews, cannot
bound the population FPIR. They condition on fitted models, frozen thresholds
and fixed galleries, excluding fitting, calibration, gallery selection and
population shift uncertainty.
```

The detector sensitivity study measures controlled processing interventions on fixed benchmark images. Its configurations do not replace the main frozen recognition policies. Higher one-face coverage is not evidence of better duplicate identification by itself. All comparisons remain limited to the evaluated benchmark populations and protocols.

Sources: [main research report](RESEARCH_REPORT.md), [paired pipeline statistics](COMPARATIVE_STATISTICS_REPORT.md), [full detector sensitivity study](detector_settings/DETECTOR_SETTINGS_REPORT.md).
