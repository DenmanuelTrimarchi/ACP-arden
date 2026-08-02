# Figure captions

Generated from the published artefacts by `ACP_arden.py`. Every figure is drawn from machine-readable results; no value is typed by hand.

Denominators for the BFW held-out test partition: 198 of 200 gallery identities enrolled, 942 scored mated probes, 2859 scored non-mated probes.

## Figure 1 — false_reviews_per_1000_by_method

False human-review referrals per 1,000 non-mated searches, over 2859 scored non-mated probes. Lower is better. Bars are counts scaled to a common denominator, not rates over different bases.

## Figure 2 — duplicate_detection_by_method

Conditional TPIR@1 over 942 scored mated probes, end-to-end duplicate detection over every intended mated probe, and gallery enrolment coverage over 200 intended identities. The three use different denominators and are labelled separately for that reason; they must not be read as one series.

## Figure 3 — open_set_operating_curve

TPIR@1 against FPIR at the three pre-declared operating points, on a logarithmic FPIR axis. Development and held-out test are drawn separately. The threshold was selected on the development curve only; the test curve is shown for reporting and was never used to choose an operating point.

## Figure 4 — subgroup_fpir_tpir_with_confidence_intervals

Per-subgroup FPIR and TPIR@1 with 95% identity-cluster bootstrap intervals. Subgroup sample counts are in `ml_review_subgroup_metrics.csv` and are small once the partition is divided eight ways, which is why several intervals are wide. Overlapping intervals are not evidence of equality.

## Figure 5 — ml_review_classifier_coefficients

Standardised logistic-regression coefficients. Positive values increase the probability of opening a human-review case; negative values reduce it. These describe association within this classifier on these benchmark identities. They are not causal and do not transfer to another population.

## Figure 6 — pipeline_coverage_and_latency

Extraction coverage and search latency for the primary pipeline. Coverage denominators are 200 intended gallery identities, 942 mated and 2859 non-mated probes. Latency is shown rather than omitted so the cost of a stronger pipeline stays visible in any future comparison.

## Limitations common to every figure

- Each rate is conditional on the coverage reported beside it.
- These are benchmark identities, not a user population.
- A referral opens human review only; nothing here proves duplication or misuse.

## Note on Figure 1

The classifier and the calibrated threshold referred the same 20 non-mated searches in error, so their bars are equal by measurement rather than by rounding.

## Note on the absent comparison

No stronger-pipeline bar appears in Figure 1 and no comparison panel appears in Figure 6: the comparison did not run (`not_run_model_files_not_configured`). Nothing is estimated in its place.
