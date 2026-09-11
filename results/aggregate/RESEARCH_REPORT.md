# COM7014 Advanced Computing Project — research report

Auto-generated from the published artefacts. Ordered to show what each layer was intended to improve, and where it did not.

**Research objective.** To establish whether a framework combining several existing models achieves better results than any one of them used alone. Each layer below adds one component to the previous combination, so the difference between consecutive layers measures what that component contributes. No face-detection or face-recognition network is trained or fine-tuned; the contribution under test is the composition, not the models themselves.

## 1. LFW 1:1 verification

Accuracy 99.09%, FMR 0.11%, FNMR 1.71%, EER 0.78%, extraction failure 10.02%. This is a 1:1 pair task and its FMR is not comparable with the 1:N FPIR figures below.

## 2. CPLFW cross-pose transfer

Conditional accuracy 90.24% over 3,515 scored pairs, with 41.42% of the protocol never reaching comparison. Cross-pose *detection*, not comparison, is the dominant finding.

## 3. BFW single-image open-set control

FPIR 8.95%, TPIR@1 93.46%, 89.5 false reviews per 1,000. Reusing a 1:1 threshold for 1:N search refers a large share of genuinely new identities.

## 4. BFW three-image template, same threshold

FPIR 15.22%, TPIR@1 96.71%. Averaging three images raises identification but **raises** FPIR at a fixed threshold: a mean template sits nearer the centre of the embedding space and is closer to everyone. Multi-image enrolment alone did not reduce false reviews.

## 5. BFW gallery-specific calibration

FPIR 0.52%, TPIR@1 92.57%, 5.2 false reviews per 1,000. The reduction is attributable to calibration, not to the representation.

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
| insightface-scrfd-arcface-buffalo_l | 96.70% [95.40%–97.90%] | 2 | 12 | 51.58 ms | 78.49 ms | 182.4 MB |
| opencv-sface-2021dec-yunet-2023mar | 87.20% [84.30%–89.90%] | 189 | 0 | 17.89 ms | 21.72 ms | 37.1 MB |

Each pipeline was calibrated on its own development scores; the SFace threshold is never applied to ArcFace. This is a complete-pipeline comparison — detection, alignment, preprocessing, embedding width and runtime all differ — so no difference is attributable to the embedding model alone.

## 10a. The same two pipelines on one-to-one verification

Section 10 compared the pipelines on gallery search alone, so the conclusion rested on a single task. Experiments 9 and 10 put the comparison pipeline through the same one-to-one chain, each with a threshold calibrated on LFW development pairs and frozen before either evaluation.

| Metric | LFW YuNet+SFace | LFW SCRFD+ArcFace | CPLFW YuNet+SFace | CPLFW SCRFD+ArcFace |
| --- | --- | --- | --- | --- |
| Correct among scored pairs | 99.09% | 99.69% | 90.24% | 93.13% |
| Reached comparison | 89.98% | 70.10% | 58.58% | 82.25% |
| Zero-face failures | 61 | 0 | 2,321 | 57 |
| Multiple-face failures | 540 | 1,794 | 164 | 1,008 |

SCRFD + ArcFace is more accurate on both datasets, but it reaches comparison on **fewer** LFW pairs than YuNet + SFace and on far more CPLFW pairs. The failure breakdown explains the reversal, and it is not a detection weakness: SCRFD recorded almost no zero-face failures on either dataset, where YuNet failed to find a face in 2,321 CPLFW images. Every one of SCRFD's losses comes from finding more than one face.

LFW images are press photographs that frequently contain bystanders. The protocol requires exactly one detected face, so a more sensitive detector converts additional true detections into rejections. The LFW coverage gap is therefore an artefact of that constraint rather than evidence about the detector, and coverage should not be compared across pipelines on this dataset without stating it. On CPLFW, where the difficulty is pose rather than bystanders, the comparison is unambiguous.


## 10b. The review classifier on both pipelines

Section 6 fitted the classifier on the baseline pipeline and section 10 compared the pipelines without it, so the framework's most elaborate addition and its strongest components were never combined. Experiment 11 runs the identical method over the comparison pipeline, under the same seed and therefore the same identity groups.

| Review burden per 1,000 new profiles | Threshold alone | With the classifier |
| --- | --- | --- |
| YuNet + SFace | 5.2 | 7.0 |
| SCRFD + ArcFace | 1.3 | 0.0 |

The classifier moves the burden in **opposite directions** on the two pipelines. Its effect is therefore a property of the components it runs on rather than of the classifier alone. The negative result in section 6 stands for the baseline pipeline, but it cannot be stated as a general finding about the method.

The zero on the second row is an observation over 2,987 scored new profiles, not a demonstration that the population rate is zero; the interval around a zero-event rate remains wide. Detection fell from 96.80% to 95.90% in exchange.


## 10c. Which component carries the gain

Experiments 6, 8 and 11 each change the detector and the embedder together, so none of them can say which component earned the difference. Experiment 12 runs the two crossings on the same held-out identities, each at a threshold frozen on the development identities by the same rule.

| Pipeline | Duplicates detected (TPIR@1) | 95% interval | Reviews per 1,000 |
| --- | --- | --- | --- |
| YuNet + SFace | 92.57% | 90.15–94.89% | 5.2 |
| SCRFD + SFace | 94.99% | 93.39–96.50% | 6.4 |
| YuNet + ArcFace | 97.24% | 95.98–98.47% | 1.7 |
| SCRFD + ArcFace | 96.80% | 95.50–98.00% | 1.3 |

Changing the embedder alone moves detection by +4.67 percentage points; changing the detector alone moves it by +2.43. The gain belongs almost entirely to the embedder.

Which of those differences the intervals actually support is stated rather than assumed. Comparing independent intervals is conservative: separation is evidence of a difference, but overlap on its own does not establish that there is none.

The intervals are disjoint for swapping the embedder at the YuNet detector, so that difference is supported.
They overlap for swapping the embedder at the SCRFD detector, swapping the detector at the SFace embedder and swapping the detector at the ArcFace embedder, which this benchmark cannot separate.

The two changes are not additive. Making both moves detection by +4.23 points, less than the +7.10 the separate gains would predict, and no better than the embedder alone. The stronger detector adds nothing once the stronger embedder is in place.

This qualifies the project's own objective. Combining components did produce the best result, but not because the combination was greater than its parts: one component carried the improvement and the other contributed within sampling noise. A study that swapped both at once, as sections 8 and 10 do, would have credited the pairing for a gain that one component produced alone.

The review burden orders the pipelines differently. The two ArcFace cells refer far fewer profiles than the two SFace cells, but within each embedder the stronger detector refers slightly more, having scored more of the harder photographs rather than failing to extract them. Detection and burden are therefore not improved by the same choice. Every interval on these burden figures overlaps every other, so that ordering is the direction the point estimates take rather than a difference this benchmark establishes.


## 11. Performance against cost

A stronger pipeline is not free. Where it improves extraction and identification it also costs disk and latency, and the trade-off is shown in `implementation_layers_performance_latency` rather than omitted.

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
- Coverage differences between the two detectors are shaped by this protocol's requirement that exactly one face be found. A detector that recovers faint faces also recovers bystanders, so a coverage loss is not on its own evidence of weaker detection.
