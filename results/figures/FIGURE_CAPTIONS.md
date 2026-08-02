# Figure captions

Generated from the published JSON and CSV artefacts by `ACP_arden.py`. No value is typed by hand. Every figure is PNG at 300 dpi plus SVG, with PNG text metadata stripped, and passes the project privacy scan.

**Denominators (BFW held-out test):** 198 of 200 gallery identities enrolled, 942 scored mated probes, 2863 scored non-mated probes.

**Metric definitions.** FPIR is the proportion of non-mated searches returning at least one candidate above threshold — a 1:N quantity that compounds with gallery size, and never interchangeable with the 1:1 false-match rate. TPIR@k is the proportion of mated searches placing the correct identity within rank k *and* above threshold; a referral to another identity is a referral, not an identification. End-to-end detection divides by every intended mated probe, so extraction failures reduce it; conditional rates divide by those actually scored.

**Confidence intervals.** All intervals are 95% percentile bounds from an identity-cluster bootstrap: identities are resampled with replacement, carrying their complete protocol outcomes, with subgroup stratification preserved. Images are never resampled independently, which would treat correlated probes of one person as independent observations and understate the intervals.

**Coefficients describe association inside the fitted classifier on these benchmark identities. They are not causal and do not transfer to another population.**

**Experiment 8 status:** `evaluated_non_commercial_academic_research`.

## Implementation layers (Figures A-D)

The five layers are measured on the same BFW open-set protocol and are therefore directly comparable, in the order the project developed them:

1. Single-image gallery, transferred 1:1 threshold
2. Three-image gallery, transferred 1:1 threshold
3. Three-image gallery, BFW development calibration
4. Logistic-regression review classifier
5. SCRFD + ArcFace, its own BFW development calibration

LFW and CPLFW are 1:1 verification and are deliberately absent from this series: mixing an FMR into an FPIR axis would compare different quantities.

- **implementation_layers_fpir** — false review referrals per 1,000 non-mated searches over 2863 scored probes. Lower is better.
- **implementation_layers_duplicate_detection** — TPIR@1, TPIR@5 and end-to-end detection, kept as separate bars because they use different denominators. Higher is better.
- **implementation_layers_coverage** — gallery, mated and non-mated coverage. The remainder in each bar is extraction failure, which is shown rather than hidden.
- **implementation_layers_performance_latency** — end-to-end detection against mean search latency; point size is false reviews per 1,000, so the speed cost of a stronger pipeline stays visible.

## Same-person and profile-photo figures (Figures E-F)

- **mated_non_mated_similarity_distributions** — aggregate histograms only. No individual score, identifier or path is published.
- **profile_photo_consistency_outcomes** — over 1000 photographs: 871 consistent, 71 review candidates, 48 extraction failures. An inconsistent result is **not** proof of photo theft or fraud: pose, lighting, occlusion, image quality, age difference, detection failure and model error all produce it. Every outcome opens human review only.

## Sex-separated figures (Figures G-I)

Sex is an evaluation dimension only and is never a classifier feature or threshold input. The female panel covers asian, black, indian and white females; the male panel covers the same four categories. Both use identical axes, units and ordering so they compare fairly.

- **female_subgroup_pipeline_comparison** / **male_subgroup_pipeline_comparison** — FPIR and TPIR@1 with 95% identity-cluster intervals.
- **female_male_aggregate_comparison** — pooled from underlying identity outcomes, not by averaging subgroup percentages, which would weight a small subgroup as heavily as a large one.

These benchmark categories do not represent every identity or any real dating-application population.

## Pipeline figures

- **pipeline_coverage_and_latency** — coverage beside latency, so a stronger pipeline's cost is not omitted.

## Limitations common to every figure

- Each rate is conditional on the coverage reported beside it.
- Subgroup intervals are wide once the partition is divided eight ways; overlapping intervals are not evidence of equality.
- These are benchmark identities, not a user population.
- A referral opens human review only. Nothing here proves duplication, fraud, ownership or identity.

## Note on the classifier

The classifier referred 20 non-mated searches in error against the calibrated threshold's 15. Its primary hypothesis — fewer false referrals — is not achieved.
