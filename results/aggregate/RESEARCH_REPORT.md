# COM7014 Advanced Computing Project — research report

Auto-generated from the published artefacts. Ordered to show what each layer was intended to improve, and where it did not.

**Research objective.** Assess how detector and recogniser choice, enrolment images and threshold calibration affect duplicate detection, false-review burden, subgroup performance and computational cost. Controlled comparisons separate these factors where the protocol permits. The face networks remain pretrained and frozen; only the review classifier is trained here.

**Academic contribution.** A reproducible empirical comparison of four configured detector–recogniser pipelines, combining processing coverage, review referrals, calibration transfer and computational cost. These are established evaluation concepts, not new metrics. The contribution is the controlled application and the resulting evidence about these combinations.

[NIST FRTE](https://pages.nist.gov/frvt/html/frvt1N.html) already evaluates identification errors, human-review scenarios and resource use. [Robinson et al. (2020)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Robinson_Face_Recognition_Too_Bias_or_Not_Too_Bias_CVPRW_2020_paper.pdf) study BFW subgroup verification and threshold differences. This project applies those established concerns to its own identity-disjoint gallery protocol, crossed components and explicit failed-processing denominators. It does not claim that deployment-oriented evaluation was absent from prior work.

See the [report-writing guide](../../REPORT_WRITING_GUIDE.md) for the requirement-to-evidence map, literature positioning and limits on interpretation.

## 1. LFW 1:1 verification

LFW mean fold accuracy 99.28%, pooled FMR 0.33%, FNMR 1.11%, EER 0.78%, extraction failure 10.02%. This is a 1:1 pair task and its FMR is not comparable with the 1:N FPIR figures below. The official ten-fold protocol fits a threshold using the other nine folds for each held-out fold; the separate development-frozen transfer threshold is not used here.

## 2. CPLFW cross-pose transfer

Conditional accuracy 90.24% over 3,515 scored pairs, with 41.42% of the protocol never reaching comparison. Cross-pose *detection*, not comparison, is the dominant finding.

## 3. BFW single-image open-set control

FPIR 8.95%, TPIR@1 93.46%, 89.5 false reviews per 1,000. Reusing a 1:1 threshold for 1:N search refers a large share of genuinely new identities.

## 4. BFW three-image template, same threshold

FPIR 15.22%, TPIR@1 96.71%. Averaging three images changes both genuine and impostor score distributions. These renormalised templates do not necessarily become closer to every face. The separately calibrated one-versus-three-image comparison appears in the supplementary diagnostics.

## 5. BFW gallery-specific calibration

FPIR 0.52%, TPIR@1 92.57%, 5.2 false reviews per 1,000. Comparing this row with the preceding three-image row isolates the threshold change on the same templates.

## 6. Logistic-regression review classifier

FPIR 0.70% against the threshold method's 0.52%; TPIR@1 94.27% against 92.57%; 7.00 false reviews per 1,000 against 5.25.

The primary hypothesis was that the classifier would reduce false review referrals while retaining detection. That criterion is **not achieved**. The classifier raises identification while referring more innocent registrations, which is a trade-off rather than an improvement.

## 7. Female subgroup analysis

Pooled over identity outcomes, not by averaging subgroup percentages.

| Pipeline | Identities | FPIR | TPIR@1 | TPIR@5 | Mated scored/intended | Mated failures | Mated coverage | Non-mated scored/intended | Non-mated failures | Non-mated coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 200 | 0.27% [0.00%–0.74%] | 97.40% [95.60%–99.20%] | 97.40% [95.60%–99.20%] | 500/500 | 0 | 100.00% [100.00%–100.00%] | 1496/1500 | 4 | 99.73% [99.47%–99.93%] |
| opencv-sface-2021dec-yunet-2023mar | 200 | 0.90% [0.28%–1.66%] | 90.64% [86.64%–94.38%] | 90.64% [86.64%–94.38%] | 481/500 | 19 | 96.20% [94.40%–97.80%] | 1442/1500 | 58 | 96.13% [94.67%–97.47%] |

## 8. Male subgroup analysis

Pooled over identity outcomes, not by averaging subgroup percentages.

| Pipeline | Identities | FPIR | TPIR@1 | TPIR@5 | Mated scored/intended | Mated failures | Mated coverage | Non-mated scored/intended | Non-mated failures | Non-mated coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 200 | 0.00% [0.00%–0.00%] | 96.19% [94.20%–97.99%] | 96.19% [94.20%–97.99%] | 499/500 | 1 | 99.80% [99.40%–100.00%] | 1491/1500 | 9 | 99.40% [99.00%–99.73%] |
| opencv-sface-2021dec-yunet-2023mar | 200 | 0.14% [0.00%–0.35%] | 94.58% [92.41%–96.72%] | 94.58% [92.41%–96.72%] | 461/500 | 39 | 92.20% [88.60%–95.20%] | 1417/1500 | 83 | 94.47% [93.00%–95.73%] |

## 9. Profile-photo identity consistency

Exploratory threshold reuse. The operating threshold was frozen for open-set duplicate-profile screening and is applied here unchanged; no threshold was calibrated for profile-photo consistency and none of these figures has been separately validated. This is not a validated identity-authentication system and must not be reported as one.

| Pipeline | Consistency (cond.) | Consistency (end-to-end) | Open-set control detection (cond.) | Wrong-template detection (cond.) | Wrong-template false-consistency (cond.) | Same-person coverage |
| --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 96.80% | 96.70% | 99.87% | 100.00% | 0.00% | 99.90% |
| opencv-sface-2021dec-yunet-2023mar | 92.57% | 87.20% | 99.48% | 100.00% | 0.00% | 94.20% |

The outcomes are not equivalent. A consistent photograph opens no case. An inconsistent one — a *low* similarity to the profile's own template — opens a consistency review. An extraction failure resolves nothing and is an unresolved outcome rather than a decision. Duplicate screening runs in the opposite direction: there a *high* similarity to another enrolled identity opens the review.

The two controls are also different questions. The open-set control searches an absent person against the whole gallery, which is the stricter test. The wrong-template control is the direct one-photograph-to-one-profile comparison and is supplementary.

A non-match indicates that the photograph is inconsistent with the enrolled facial template under the evaluated model and threshold. It does not prove that the photograph belongs to another person or that fraud occurred. Pose, lighting, occlusion, image quality, age difference, face-detection failure and model error can all produce the same result. An inconsistent photograph opens a human-review case; a consistent one does not, and an extraction failure resolves nothing.

## 10. YuNet + SFace against SCRFD + ArcFace

| Pipeline | Threshold | FPIR | TPIR@1 | Reviews/1,000 | Coverage |
| --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 0.393958 | 0.13% | 96.80% | 1.34 | 100.00% |
| opencv-sface-2021dec-yunet-2023mar | 0.477118 | 0.52% | 92.57% | 5.25 | 99.00% |

| Pipeline | End-to-end (95% CI) | Zero-face | Multiple-face | Embed mean | Complete mean | Model size |
| --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 96.70% [95.40%–97.90%] | 2 | 12 | 52.78 ms | 80.73 ms | 182.4 MB |
| opencv-sface-2021dec-yunet-2023mar | 87.20% [84.30%–89.90%] | 189 | 0 | 17.89 ms | 21.72 ms | 37.1 MB |

Each pipeline was calibrated on its own development scores; the SFace threshold is never applied to ArcFace. This is a complete-pipeline comparison — detection, alignment, preprocessing, embedding width and runtime all differ — so no difference is attributable to the embedding model alone.

## 10a. The same two pipelines on one-to-one verification

Section 10 compared the pipelines on gallery search alone, so the conclusion rested on a single task. Experiments 9 and 10 put the comparison pipeline through the same one-to-one protocols. LFW uses official ten-fold cross-validation, fitting each threshold on the other nine folds. CPLFW uses each pipeline's separate frozen LFW development threshold. No threshold is shared between pipelines.

| Metric | LFW YuNet+SFace | LFW SCRFD+ArcFace | CPLFW YuNet+SFace | CPLFW SCRFD+ArcFace |
| --- | --- | --- | --- | --- |
| Correct among scored pairs | 99.28% | 99.79% | 90.24% | 93.13% |
| Reached comparison | 89.98% | 70.10% | 58.58% | 82.25% |
| Zero-face failures | 61 | 0 | 2,321 | 57 |
| Multiple-face failures | 540 | 1,794 | 164 | 1,008 |

Conditional accuracy must be read alongside extraction coverage: the pipelines score different subsets of pairs. The exactly-one-face rule rejects both zero-face and multiple-face detections. The breakdown records these outcomes but cannot establish whether additional detections are bystanders or false positives without manual annotation. These results compare complete pipelines and do not isolate the recogniser.


## 10b. The review classifier on both pipelines

Section 6 fitted the classifier on the baseline pipeline and section 10 compared the pipelines without it, so the framework's most elaborate addition and its strongest components were never combined. Experiment 11 runs the identical method over the comparison pipeline, under the same seed and therefore the same identity groups.

| Review burden per 1,000 new profiles | Threshold alone | With the classifier |
| --- | --- | --- |
| YuNet + SFace | 5.2 | 7.0 |
| SCRFD + ArcFace | 1.3 | 0.0 |

The classifier moves the burden in **opposite directions** on the two pipelines. Its effect is therefore a property of the components it runs on rather than of the classifier alone. The negative result in section 6 stands for the baseline pipeline, but it cannot be stated as a general finding about the method.

The zero on the second row is an observation over 2,987 scored new profiles, not a demonstration that the population rate is zero; the empirical zero-event bootstrap interval cannot bound population FPIR. A supplementary identity-level upper bound is reported with its assumptions. Detection fell from 96.80% to 95.90% in exchange.


## 10c. Detector, recogniser and coverage contributions

Same intended identities and pipeline-specific development thresholds. End-to-end detection retains failed extraction and failed enrolment in its denominator.

| Pipeline | Conditional TPIR@1 | End-to-end detection | Mated coverage | False reviews / 1,000 scored new probes | Common-success TPIR@1 |
| --- | --- | --- | --- | --- | --- |
| YuNet + SFace | 92.57% | 87.20% | 94.20% | 5.25 | 92.57% |
| SCRFD + SFace | 94.99% | 94.90% | 99.90% | 6.36 | 95.44% |
| YuNet + ArcFace | 97.24% | 91.60% | 94.20% | 1.75 | 97.24% |
| SCRFD + ArcFace | 96.80% | 96.70% | 99.90% | 1.34 | 97.24% |

Common-success subset: 3,789 of 4,000 intended probes. Conditional comparisons across different surviving subsets cannot isolate a detector effect.

Paired changes below are right minus left in percentage points. Every bootstrap replicate uses the same identity draws across all methods.

| Left → right | End-to-end change (95% CI) | FPIR change (95% CI) |
| --- | --- | --- |
| YuNet + SFace → SCRFD + SFace | +7.70 [+5.80, +9.90] | +0.11 [-0.19, +0.44] |
| YuNet + SFace → YuNet + ArcFace | +4.40 [+2.80, +6.20] | -0.35 [-0.70, -0.07] |
| YuNet + SFace → SCRFD + ArcFace | +9.50 [+7.30, +12.10] | -0.39 [-0.71, -0.11] |
| SCRFD + SFace → YuNet + ArcFace | -3.30 [-5.30, -1.50] | -0.46 [-0.86, -0.13] |
| SCRFD + SFace → SCRFD + ArcFace | +1.80 [+1.00, +2.80] | -0.50 [-0.84, -0.23] |
| YuNet + ArcFace → SCRFD + ArcFace | +5.10 [+3.60, +6.90] | -0.04 [-0.21, +0.13] |

Detector-by-embedder interaction in end-to-end detection: -2.60 [-3.90, -1.50] percentage points. Contrast: (SCRFD+ArcFace - YuNet+ArcFace) - (SCRFD+SFace - YuNet+SFace).

Exploratory unadjusted 95% intervals. Fixed galleries and frozen policies; uncertainty excludes model fitting, threshold estimation, gallery selection and population shift. Common-success analysis describes a selected subset only. Zero-event bootstrap intervals cannot bound population FPIR.

An interval containing zero does not establish equivalence or absence of a contribution. The effects describe the complete configured detector, alignment, recogniser and calibrated-policy combinations.

## 11. Performance against cost

The following summary reads the saved measurements. Mixed-pipeline times are estimates assembled from measured stages, not direct timings. Storage covers weight files only; gallery templates, runtime memory and reviewer minutes were not measured. Scaling examples assume the same benchmark rates and sequential processing on the measured machine, not observed deployment throughput or total moderation demand.

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

## 11a. Additional controlled diagnostics

The separate COMPARISON_DIAGNOSTICS_REPORT.md reports calibrated one- versus three-image enrolment, gallery-size sensitivity, classifier feature ablations and error analysis. These extensions were designed after the benchmark test results were inspected and are exploratory, not independent confirmation.

## 12. Limitations and policy

This is a benchmark-validated, human-review-only academic face-comparison study. It evaluates duplicate-profile screening and profile-photo facial consistency using frozen pretrained face-recognition pipelines and an identity-disjoint logistic-regression review classifier.

The two tasks refer in opposite directions, and a single threshold statement would misdescribe one of them:

- **Duplicate-profile screening** — a *high* similarity to some other enrolled identity opens a duplicate-profile review.
- **Profile-photo consistency** — a *low* similarity to the profile's own enrolled template opens an inconsistency review.

Neither is proof of fraud, ownership or identity, and an extraction failure resolves nothing in either direction.

No face-detection or face-recognition network is trained or fine-tuned. Experiment 7 trains a small logistic-regression review classifier on identity-disjoint BFW development data and evaluates it on untouched held-out identities.

- This remains a proof of concept. No result here proves fraud, misuse or misrepresentation by any person.
- No automatic sanction is applied. Every outcome opens a case for human review and nothing else. In this gallery-screening experiment the referral is triggered by a high similarity to another enrolled identity; profile-photo consistency refers a low similarity to the profile's own template instead, and an extraction failure makes no decision at all.
- The BFW open-set evaluation uses a protocol defined by this project. BFW publishes verification and bias-analysis protocols, not an open-set identification protocol.
- Development and test identities are completely disjoint, and the operating threshold was frozen before the held-out test partition was scored.
- Extraction failures are counted as coverage failures, never as genuine no-match decisions.
- Confidence intervals describe sampling uncertainty over these benchmark identities only. They do not extend to any other population.
- Benchmark demographics do not represent any real deployed user population, so subgroup figures must not be read as deployment estimates.
- The review classifier was fitted separately on each pipeline and moved the review burden in opposite directions on the two. Its contribution therefore depends on the components underneath it, and neither result should be read as a general property of the method.
- Coverage differences between the two detectors are shaped by this protocol's requirement that exactly one face be found. Extra detections may be bystanders or false positives; they were not manually annotated. A coverage loss is not on its own evidence of weaker detection.

### Dataset scope

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

## 13. Threshold calibration and validation

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

## 14. Detector settings and failure interpretation

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

Full controlled settings, paired transitions, timings and the frozen experiment plan: [detector settings report](detector_settings/DETECTOR_SETTINGS_REPORT.md). This sensitivity study preserves all original recognition policies and main results.
