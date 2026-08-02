# ACP-arden research report

Auto-generated from the published artefacts. Ordered to show what each layer was intended to improve, and where it did not.

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

| Pipeline | Identities | FPIR | TPIR@1 | TPIR@5 | Mated coverage | Non-mated coverage |
| --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 200 | 0.27% [0.00%–0.74%] | 97.40% [95.60%–99.20%] | 97.40% [95.60%–99.20%] | 100.00% [100.00%–100.00%] | 99.73% [99.47%–99.93%] |
| opencv-sface-2021dec-yunet-2023mar | 200 | 0.90% [0.28%–1.66%] | 90.64% [86.64%–94.38%] | 90.64% [86.64%–94.38%] | 96.20% [94.40%–97.80%] | 96.13% [94.67%–97.47%] |

## 8. Male subgroup analysis

Pooled over identity outcomes, not by averaging subgroup percentages.

| Pipeline | Identities | FPIR | TPIR@1 | TPIR@5 | Mated coverage | Non-mated coverage |
| --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 200 | 0.00% [0.00%–0.00%] | 96.19% [94.20%–97.99%] | 96.19% [94.20%–97.99%] | 99.80% [99.40%–100.00%] | 99.40% [99.00%–99.73%] |
| opencv-sface-2021dec-yunet-2023mar | 200 | 0.14% [0.00%–0.35%] | 94.58% [92.41%–96.72%] | 94.58% [92.41%–96.72%] | 92.20% [88.60%–95.20%] | 94.47% [93.00%–95.73%] |

## 9. Profile-photo identity consistency

| Pipeline | Consistency (cond.) | Consistency (end-to-end) | Mismatch detection (cond.) | Mismatch detection (end-to-end) | False-consistency | Same-person coverage |
| --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 96.80% | 96.70% | 99.87% | 99.43% | 0.13% | 99.90% |
| opencv-sface-2021dec-yunet-2023mar | 92.57% | 87.20% | 99.48% | 94.80% | 0.52% | 94.20% |

The four outcomes are not equivalent. A consistent photograph opens no case. An inconsistent one opens a consistency review. A mismatched control is correctly identified when it falls below threshold and false-consistent when it does not. An extraction failure resolves nothing and is an unresolved outcome rather than a decision.

A non-match indicates that the photograph is inconsistent with the enrolled facial template under the evaluated model and threshold. It does not prove that the photograph belongs to another person or that fraud occurred. Pose, lighting, occlusion, image quality, age difference, face-detection failure and model error can all produce the same result. An inconsistent photograph opens a human-review case; a consistent one does not, and an extraction failure resolves nothing.

## 10. YuNet + SFace against SCRFD + ArcFace

| Pipeline | Threshold | FPIR | TPIR@1 | Reviews/1,000 | Coverage |
| --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 0.393958 | 0.13% | 96.80% | 1.34 | 100.00% |
| opencv-sface-2021dec-yunet-2023mar | 0.477118 | 0.52% | 92.57% | 5.25 | 99.00% |

| Pipeline | End-to-end (95% CI) | Zero-face | Multiple-face | Embed mean | Complete mean | Model size |
| --- | --- | --- | --- | --- | --- | --- |
| insightface-scrfd-arcface-buffalo_l | 96.70% [95.40%–97.90%] | 2 | 12 | 63.59 ms | 96.19 ms | 182.4 MB |
| opencv-sface-2021dec-yunet-2023mar | 87.20% [84.30%–89.90%] | 189 | 0 | 17.71 ms | 21.42 ms | 37.1 MB |

Each pipeline was calibrated on its own development scores; the SFace threshold is never applied to ArcFace. This is a complete-pipeline comparison — detection, alignment, preprocessing, embedding width and runtime all differ — so no difference is attributable to the embedding model alone.

## 11. Performance against cost

A stronger pipeline is not free. Where it improves extraction and identification it also costs disk and latency, and the trade-off is shown in `implementation_layers_performance_latency` rather than omitted.

## 12. Limitations and policy

ACP-arden is a benchmark-validated, human-review-only academic face-comparison proof of concept. It evaluates duplicate-profile screening and profile-photo facial consistency using frozen pretrained face-recognition pipelines and an identity-disjoint logistic-regression review classifier.

The two tasks refer in opposite directions, and a single threshold statement would misdescribe one of them:

- **Duplicate-profile screening** — a *high* similarity to some other enrolled identity opens a duplicate-profile review.
- **Profile-photo consistency** — a *low* similarity to the profile's own enrolled template opens an inconsistency review.

Neither is proof of fraud, ownership or identity, and an extraction failure resolves nothing in either direction.

No face-detection or face-recognition network is trained or fine-tuned. Experiment 7 trains a small logistic-regression review classifier on identity-disjoint BFW development data and evaluates it on untouched held-out identities.

- This remains a proof of concept. No result here proves fraud, misuse or misrepresentation by any person.
- No automatic sanction is applied. A score above threshold opens a case for human review and nothing else.
- The BFW open-set evaluation uses a protocol defined by this project. BFW publishes verification and bias-analysis protocols, not an open-set identification protocol.
- Development and test identities are completely disjoint, and the operating threshold was frozen before the held-out test partition was scored.
- Extraction failures are counted as coverage failures, never as genuine no-match decisions.
- Confidence intervals describe sampling uncertainty over these benchmark identities only. They do not extend to any other population.
- Benchmark demographics do not represent a real dating-application user population, so subgroup figures must not be read as deployment estimates.
