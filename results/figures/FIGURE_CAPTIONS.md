# Figure captions

Generated from the published JSON and CSV artefacts by `ACP_arden.py`. No value is typed by hand. Every figure is PNG at 300 dpi plus SVG, with PNG text metadata stripped, and passes the project privacy scan.

**Denominators (BFW held-out test):** 198 of 200 gallery identities enrolled, 942 scored mated probes, 2863 scored non-mated probes.

**Metric definitions.** FPIR is the proportion of non-mated searches returning at least one candidate above threshold — a 1:N quantity that compounds with gallery size, and never interchangeable with the 1:1 false-match rate. TPIR@k is the proportion of mated searches placing the correct identity within rank k *and* above threshold; a referral to another identity is a referral, not an identification. End-to-end detection divides by every intended mated probe, so extraction failures reduce it; conditional rates divide by those actually scored.

**Confidence intervals.** All intervals are 95% percentile bounds from an identity-cluster bootstrap: identities are resampled with replacement, carrying their complete protocol outcomes, with subgroup stratification preserved. Images are never resampled independently, which would treat correlated probes of one person as independent observations and understate the intervals.

**Coefficients describe association inside the fitted classifier on these benchmark identities. They are not causal and do not transfer to another population.**

**Experiment 8 status:** `evaluated_non_commercial_academic_research`.

## Result order

Captions follow the order in which the study developed, so each layer's intent is visible before the pretrained comparison.

### 1. LFW 1:1 verification

Pairwise verification with the frozen threshold, reported as accuracy, FMR and FNMR over scored pairs. A 1:1 quantity that never appears on an FPIR axis: one comparison, no competing candidates, no ranking. Conditional on scored pairs.

### 2. CPLFW cross-pose evaluation

The same frozen threshold on raw cross-pose images. Conditional accuracy only, always quoted with its extraction-failure rate; cross-pose detection rather than comparison is the dominant effect.

## Implementation layers (results 3-6)

The five layers share the BFW open-set protocol and are directly comparable, in the order the project developed them:

3. Single-image gallery, transferred 1:1 threshold
4. Three-image gallery, transferred 1:1 threshold — higher TPIR but **higher FPIR**; a mean template sits nearer the centre of the embedding space and is closer to everyone, so this layer is not an improvement
5. Three-image gallery, BFW development calibration — the reduction in false reviews comes from calibration, not from the representation
6. Logistic-regression review classifier, frozen probability threshold
7. SCRFD + ArcFace, its own frozen BFW development calibration

LFW and CPLFW are 1:1 verification and are deliberately absent from this series: mixing an FMR into an FPIR axis would compare different quantities.

- **implementation_layers_fpir** — false review referrals per 1,000 non-mated searches over 2863 scored probes. Lower is better.
- **implementation_layers_duplicate_detection** — TPIR@1, TPIR@5 and end-to-end detection, kept as separate bars because they use different denominators. Higher is better.
- **implementation_layers_coverage** — gallery, mated and non-mated coverage. The remainder in each bar is extraction failure, which is shown rather than hidden.
- **implementation_layers_performance_latency** — end-to-end detection against mean search latency; point size is false reviews per 1,000, so the speed cost of a stronger pipeline stays visible.

## Same-person and profile-photo figures (Figures E-F)

- **mated_non_mated_similarity_distributions** — aggregate histograms only. No individual score, identifier or path is published.
- **profile_photo_consistency_outcomes** — over 1000 photographs: 871 consistent, 71 review candidates, 48 extraction failures. An inconsistent result is **not** proof of photo theft or fraud: pose, lighting, occlusion, image quality, age difference, detection failure and model error all produce it. Every outcome opens human review only.

## 7-8. Female and male subgroup evaluation

Sex is an evaluation dimension only: never a classifier feature, threshold input, calibration variable, or reason to apply a different decision policy. The female panel covers asian, black, indian and white females; the male panel covers the same four categories. Both use identical metric order, axis limits (0-100%), units and interval format so they compare fairly.

- **female_subgroup_pipeline_comparison** / **male_subgroup_pipeline_comparison** — FPIR (lower better), TPIR@1 and TPIR@5 (higher better), mated coverage and non-mated coverage, each with 95% identity-cluster bounds.
- **female_male_aggregate_comparison** — pooled from underlying identity outcomes, not by averaging four subgroup percentages, which would weight a small subgroup as heavily as a large one.

These are binary dataset categories. They do not represent the full range of gender identities, every identity, or any real dating-application population.

## 9. Profile-photo consistency analysis

A same-identity probe stands for a photograph belonging to the enrolled person; a non-mated probe is the mismatched control, where referral is the correct outcome.

> A non-match indicates that the photograph is inconsistent with the enrolled facial template under the evaluated model and threshold. It does not prove that the photograph belongs to another person or that fraud occurred.

## 10-11. Pipeline comparison and the latency trade-off

- **pipeline_coverage_and_latency** — both pipelines once Experiment 8 is evaluated, each with its own frozen development threshold; the SFace threshold is never applied to ArcFace. A complete-pipeline comparison: detection, alignment, preprocessing and embedding width all differ, so no difference is attributable to the embedding model alone.
- **implementation_layers_performance_latency** — end-to-end detection against latency, point size being false reviews per 1,000. Latency excludes one-time model loading. A stronger pipeline is not free and its cost is shown, not omitted.

## 12. Limitations

## Limitations common to every figure

- Each rate is conditional on the coverage reported beside it.
- Subgroup intervals are wide once the partition is divided eight ways; overlapping intervals are not evidence of equality.
- These are benchmark identities, not a user population.
- A referral opens human review only. Nothing here proves duplication, fraud, ownership or identity.

## Note on the classifier

The classifier referred 20 non-mated searches in error against the calibrated threshold's 15. Its primary hypothesis — fewer false referrals — is not achieved.
