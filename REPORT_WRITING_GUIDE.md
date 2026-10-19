# Report evidence and response to supervisor feedback

This guide addresses the feedback dated **19 October 2026**, retaining the
earlier requirements from 29 September. It interprets the saved main evaluation
artefacts from 3 October 2026 and the additional controlled detector-settings
experiment completed on 5 October 2026. The October code and documentation
revision adds interpretation and exposes existing sensitivity evidence; it
does not represent a new benchmark run. Use the generated
[research report](results/aggregate/RESEARCH_REPORT.md) for results and the
[reference register](REFERENCES.md) for component attribution.

## 1. Academic contribution and relationship to previous research

**Proposed contribution statement.** This project contributes a reproducible,
deployment-oriented empirical evaluation of duplicate-profile screening. It
compares four configured detector–recogniser combinations under a shared,
identity-disjoint gallery protocol, retaining failed processing in end-to-end
detection and relating detection to false-review referrals, latency and model
storage. Controlled comparisons examine threshold transfer, enrolment and
component changes; paired uncertainty estimates delimit the conclusions.

The contribution is an experimental design, reproducible artefact and set of
empirical findings. Neither duplicate-face lookup, open-set identification,
coverage measurement nor cost-aware evaluation is claimed as a new invention.

| Prior research | Established contribution | What this project investigates in addition |
| --- | --- | --- |
| Huang et al. (2007), [LFW](https://people.cs.umass.edu/~elm/papers/lfw.pdf) | Unconstrained face-pair verification and evaluation protocols. | Whether a separately frozen pairwise policy transfers to gallery search; how processing failures change the interpretation of conditional accuracy. |
| Zheng and Deng (2018), [CPLFW](https://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf) | A more difficult verification benchmark with pose variation and identities shared with LFW. | Transfer of each pipeline's frozen LFW development threshold to raw CPLFW images, with explicit zero-face and multiple-face failure counts. This transfer experiment is distinct from CPLFW fold-wise recalibration. |
| Deng et al. (2019), [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html) | An angular-margin training loss and recognition benchmark evaluation. | Frozen pretrained recognisers paired with alternative configured detectors; the study does not retrain ArcFace or compare loss functions. |
| Robinson et al. (2020), [BFW](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Robinson_Face_Recognition_Too_Bias_or_Not_Too_Bias_CVPRW_2020_paper.pdf) | Demographic differences in verification and subgroup thresholds. | A project-defined open-set gallery protocol, one global policy per pipeline, coverage and false-referral reporting, and component comparisons. It is not an official BFW identification benchmark. |
| NIST, [FRTE 1:N identification](https://pages.nist.gov/frvt/html/frvt1N.html) | Identification error rates, review scenarios, timing and storage evaluation already exist. | A locally reproducible comparison of four accessible component combinations, paired outcomes and controlled extensions on the selected academic datasets. This study neither replaces NIST nor establishes a first deployment-oriented methodology. |

These sources establish the context, not an exhaustive proof of novelty. The
research question left to this study is which configured combination provides
the most useful balance **under its particular protocol**, and how that answer
changes when failed processing and review referrals are retained. The four-way
comparison, interaction estimate and opposite classifier outcomes supply the
project-specific evidence. Academic acceptance of that contribution remains a
judgement for the supervisor and assessment process.

### Critical synthesis rather than a catalogue of models

The research argument should compare what different approaches measure, what
they leave unresolved and how that changes this study's design:

1. **Verification scores and complete processing answer different questions.**
   LFW offers a reproducible pairwise reference, but its images have already
   passed a detection-based collection process. A high score on that sample
   cannot establish coverage on arbitrary uploads. CPLFW challenges optimistic
   verification results through pose and harder negative pairs. Applying a
   frozen LFW policy to its raw images tests a complete transfer configuration;
   differences from aligned-image or fold-recalibrated studies cannot be
   attributed solely to recognition. Shared identities also restrict
   independent-population claims. See [LFW](https://people.cs.umass.edu/~elm/papers/lfw.pdf)
   and [CPLFW](https://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf).
2. **Representation learning and pipeline selection are different experiments.**
   ArcFace establishes an angular-margin approach to learning embeddings.
   This artefact compares frozen published weights, with different training
   histories, capacities and alignment. It cannot identify a causal advantage
   of the loss function. Similarly, the exactly-one-face rule means that more
   detections can reduce usable coverage. The relevant empirical question is
   which configured system produces useful review referrals with its failures
   retained. See [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf).
3. **Threshold objectives compete.** BFW's verification study shows that
   subgroup score distributions and threshold choices matter. An aggregate
   global-policy audit does not test whether subgroup-specific policies would
   be acceptable or effective. Here the policy remains global; small subgroup
   samples and unmeasured reviewer outcomes leave fairness unresolved. See
   [Robinson et al.](https://arxiv.org/abs/2002.06483).
4. **Identification errors and resource costs are established concerns.**
   NIST already evaluates search errors and computational resources. The gap
   addressed here is which of these four accessible configurations balances
   intended-probe detection, false referrals and measured cost under this
   fixed protocol. The executable comparison and its findings supply the
   contribution; combining existing metrics alone does not establish a new
   general methodology. See [NIST FRTE](https://pages.nist.gov/frvt/html/frvt1N.html).

These are interpretations of the cited sources in relation to this artefact,
not a claim that the literature has never considered coverage or workload.
The contribution must remain bounded by the selected models, data and policies.

## 2. Threshold calibration and validation

| Task | Calibration and selection | Evaluation and interpretation |
| --- | --- | --- |
| LFW final verification | Fit a threshold on nine official folds, selecting maximum training-fold accuracy. Repeat for each held-out fold. | Score the tenth fold; report the ten-fold result and processing coverage. No single development threshold is used for this final result. |
| CPLFW transfer | Generate candidates on LFW `pairsDevTrain.txt`; select on `pairsDevTest.txt` by maximum balanced accuracy, then lower false-match rate and candidate name. Freeze separately for each pipeline. | Apply unchanged to raw CPLFW. Thresholds: YuNet + SFace 0.363012; SCRFD + ArcFace 0.329049. This tests transfer across image conditions, not an independent population. |
| BFW gallery search | Use development identities disjoint from test identities. At the primary 0.3% FPIR target, maximise development TPIR@1 among admissible thresholds; break ties by lower FPIR, higher threshold and candidate name. | Apply the frozen policy to held-out identities. Thresholds: YuNet + SFace 0.477118; SCRFD + SFace 0.471432; YuNet + ArcFace 0.391290; SCRFD + ArcFace 0.393958. |
| Review classifier | Fit on 70% of the BFW development identities; calibrate its referral-score cutoff on the remaining 30%, preserving subgroup stratification. | Evaluate on held-out test identities. Score cutoffs are pipeline-specific and are not calibrated real-world probabilities of duplication. |

The borrowed LFW threshold in the gallery controls is deliberate: it measures
transfer failure. A pairwise false-match rate is not a gallery FPIR. Also,
meeting a development target does not guarantee a test rate: YuNet + SFace
achieves about 0.28% development FPIR and 0.52% test FPIR; SCRFD + SFace rises
from about 0.27% to 0.64%. Report these deviations rather than saying the 0.3%
target was satisfied on every held-out pipeline.

Evidence: [baseline calibration and transfer](results/aggregate/FINAL_EVALUATION_REPORT.md),
[frozen gallery policy](results/aggregate/bfw_open_set_threshold.json), and
section 13 of the [generated report](results/aggregate/RESEARCH_REPORT.md).
Profile-photo consistency reuses the gallery threshold experimentally and has
not received separate task-specific calibration or validation.

### Moderation rationale and sensitivity

The 0.1%, 0.3% and 1% development FPIR targets represent illustrative budgets
of one, three and ten false referrals per 1,000 scored non-mated searches.
The primary choice is a research operating point, not a requirement validated
with moderators or measured service costs. A realistic policy would also need
duplicate prevalence, missed-duplicate costs, unresolved extractions, review
capacity, reviewer decisions and appeals. False-referral counts alone do not
measure total workload or reviewer minutes.

For YuNet + SFace, the existing frozen targets yield held-out conditional
TPIR@1 of 42.99%, 92.57% and 94.69%, with 0, 5.25 and 17.84 false referrals
per 1,000 scored non-mated probes. The conclusions are materially sensitive
to policy: tightening the target reduces detection sharply, and the primary
and looser development budgets do not hold on test. These are descriptive
within-protocol comparisons, without paired intervals for the budget changes.

The generated response puts the existing development and held-out points
beside one another for each pipeline. The original SCRFD + ArcFace comparison
saved held-out outcomes only at the primary point. Its non-primary sensitivity
remains unavailable in those results; development rates are not test rates.
Updated code saves held-out outcomes for all pre-frozen points in future
comparison runs. No claim of a stable four-pipeline ranking across all budgets
is supported by the present files, and test outcomes never select the policy.

## 3. Dataset limitations and generalisability

LFW and CPLFW do not constitute independent population replications: CPLFW
uses LFW identities while changing the image and pair conditions. Their
public-figure imagery does not establish performance on ordinary service
users. BFW's balanced demographic sampling supports comparisons within the
benchmark, but its balance and group definitions do not estimate a deployment
population's composition or eliminate bias.

### Why these datasets and why a project-defined gallery protocol

LFW supplies a documented verification reference and development files.
CPLFW changes image difficulty while retaining identities, enabling a bounded
transfer test. BFW supplies repeated images and official subgroup metadata,
making identity-disjoint calibration, gallery search and subgroup reporting
feasible within the project. The choices answer complementary questions; they
were not selected to stand in for a representative dating-service population.

The BFW authors' verification task does not supply the required gallery roles.
This study therefore constructs a separate protocol with explicit enrolled
and absent identities. It must be described as a project-defined evaluation,
and its rates cannot be compared directly with published BFW verification or
NIST identification numbers. The official images and metadata are the source
data, not an endorsement of this particular protocol.

The reproducible recipe is in the [README](README.md#exact-bfw-construction-and-its-consequences)
and `build_open_set_protocol`. Within each subgroup, sorted identities undergo
seeded 50/50 development/test and enrolled/non-mated splits. Images are sorted
lexically: the first three enrol, the next five are mated probes; the first
fifteen from absent identities are non-mated probes. The fixed seed is
20260727 and the partition is independent of the identifier secret. Eligibility
and odd-count rules are explicit in code. Future protocol summaries also
record the selection recipe; older summaries retain counts and fingerprints.

Each evaluated partition has 400 identities: 200 intended gallery identities,
600 enrolment images, 1,000 mated probes and 3,000 non-mated probes. The protocol
uses 9,200 of the corpus's 20,000 images. The balance preserves separate
development and test identities; multi-image enrolment and repeated probes
permit a feasible local comparison and finer empirical probe-rate resolution.
These choices are design compromises, not demonstrated optimal settings.
The remaining images are unused, and the 1:3 probe ratio is not prevalence.

Repeated probes introduce correlation: 3,000 non-mated searches represent 200
identities, with just 25 non-mated identities per subgroup. Identity-cluster
resampling addresses within-identity dependence for the stated estimates;
it does not remedy lexical-order bias, a single identity split, fixed galleries,
pretraining overlap or an unrepresentative population. Gallery-seed diagnostics
do not independently replicate the development/test split.

The primary BFW test gallery has 200 intended identities. Supplementary
gallery-size runs cover 25, 50, 100 and 200 identities; three gallery seeds do
not provide three independent development/test populations. None of these
results establishes million-profile search performance. The study has not
validated representative frequencies of selfies, filters, editing, ageing,
occlusion or adversarial submissions. Identity overlap with external
pretraining collections has not been audited, so benchmark familiarity remains
a possible limitation rather than a demonstrated source of contamination.

Confidence intervals quantify sampling variation within the benchmark and
protocol. They do not cover population shift. Further validation would require
consented, representative data, a newly frozen policy and independent test
identities. No additional dataset is represented as evaluated here.

## 4. Accuracy, latency, storage and review workload

The following snapshot is from the [pipeline comparison](results/aggregate/PRETRAINED_PIPELINE_COMPARISON_REPORT.md).

| Measure | YuNet + SFace | SCRFD + ArcFace |
| --- | --- | --- |
| Correct duplicate identification / all 1,000 intended mated probes | 87.20% | 96.70% |
| Conditional TPIR@1 / scored mated probes | 92.57% | 96.80% |
| False referrals / 1,000 scored non-mated probes | 5.25 | 1.34 |
| Mean complete processing per image | 21.72 ms | 80.73 ms |
| Complete processing p95 | 27.10 ms | 85.06 ms |
| Detector and recogniser weight files, approximately | 37.1 MiB | 182.4 MiB |

The improved end-to-end rate costs approximately 3.72 times the processing
time and 4.92 times the weight-file storage. Under unchanged benchmark rates,
100,000 successfully processed new identities imply roughly 525 versus 134
false-review cases. This is a workload proxy, not measured reviewer time or
the total review queue: true referrals and unresolved extraction failures may
also require work. No optimal economic choice follows without review costs,
duplicate prevalence and the cost of missed duplicates.

Processing time includes loading, detection and embedding; gallery search is
reported separately. Timings are specific to the recorded hardware/software
environment. Mixed-pipeline times in the generated cost summary are sums of
stages measured in the two original pipelines and are explicitly estimates.
Storage figures cover weight files only, not gallery templates, the review
database, runtime memory or library binaries. Existing JSON fields labelled
`megabytes` divide bytes by 1,048,576 and thus represent MiB.

## 5. Detector settings and failure rates

YuNet uses confidence 0.9 at the image's native dimensions. SCRFD uses
confidence 0.5 and aspect-preserving resizing with padding to a 320 × 320
canvas. The confidence values come from different models and are not on a
shared calibrated scale; comparing 0.9 with 0.5 does not establish which
detector is intrinsically stricter.

The **directly established mechanism** is the exactly-one-face policy: zero
detections or multiple detections stop processing before recognition. On LFW,
SCRFD loses 1,794 pairs to multiple detections, versus 540 for YuNet. On CPLFW,
YuNet loses 2,321 pairs to zero detections, versus 57 for SCRFD. On BFW, zero
face counts are 189 versus 2, while multiple face counts are 0 versus 12.
Pair counts and probe-image counts are different denominators; do not pool
them. A failed pair receives the first terminal error in left-to-right
processing, not a count of every defective image in the pair.

The [controlled detector-settings study](results/aggregate/detector_settings/DETECTOR_SETTINGS_REPORT.md)
now tests the effects of confidence and input canvas within each detector on
identical images, holding weights, NMS and the exactly-one-face rule fixed.
It evaluates 14 preset configurations across four cohorts, each containing
256 images from 128 identities: LFW development, raw CPLFW, BFW development
and BFW test. BFW development and test identities are disjoint. Settings and
cohort fingerprints were saved before measurement; no configuration was selected
to replace the main pipeline. Independent detector instances with identical
baseline settings provide repeat controls.

The interventions show why the processing trade-off depends on the images:

- On LFW development images, lowering YuNet confidence from 0.9 to 0.8 increased
  multiple detections from 14 to 37 and reduced one-face coverage by 8.98
  percentage points (95% CI −12.89 to −5.47).
- On raw CPLFW images, the same confidence change reduced zero detections from
  70 to 9 and increased one-face coverage by 21.48 points (15.62–26.95).
- On the BFW test cohort, changing SCRFD's canvas from 320 to 640 pixels reduced
  one-face coverage from 256/256 to 14/256, a change of −94.53 points
  (−97.27 to −91.80). This directly measures sensitivity to the canvas setting
  on these images; it does not establish a universal disadvantage of 640 pixels.

The study reports all preset variants, paired image transitions, unresolved
images per 1,000 intended images, detector mean/p95 latency and 2,000-replicate
paired identity-bootstrap intervals. Model bytes stay fixed within a detector.
These are controlled processing outcomes, not recognition accuracy or false
duplicate referrals. Extra detections remain unannotated, so they cannot be
called confirmed bystanders or confirmed false positives. Intervals are
exploratory and unadjusted, and the cohorts are samples rather than full pair
protocols or a new deployment population.

The crossed experiment supports comparisons of **configured components**:
with SFace held fixed, replacing YuNet with SCRFD raises end-to-end detection
by 7.70 percentage points (95% CI 5.80–9.90). With YuNet held fixed, replacing
SFace with ArcFace raises it by 4.40 points (2.80–6.20). Alignment and separately
calibrated recognition thresholds remain part of those configurations. The
effects are not additive; the interaction is −2.60 points (−3.90 to −1.50).

This is partial control of configured components. Across detectors, score
scales, accepted detections, input canvas and landmarks differ; across
recognisers, alignment, preprocessing, dimensionality, capacity and external
training data differ. Separate calibration is necessary for useful policies
but also forms part of the comparison. Equal numerical detector cutoffs would
not make their scores equivalent. Neither the crossed comparison nor the
detector study establishes an isolated architecture or loss-function effect.
The latter holds other settings fixed within a detector, and its outcome is
processing eligibility rather than recognition performance.

Future work concerns annotation and adoption: manually label ambiguous
detections, choose any replacement setting using development data only, then
recalibrate recognition and evaluate duplicate identification, false referrals
and total computational cost on new held-out identities. Increased one-face
coverage alone does not justify changing the main recognition pipeline.

## 6. Statistical support and report conclusion

The [paired comparison](results/aggregate/COMPARATIVE_STATISTICS_REPORT.md)
uses 2,000 identity-cluster bootstrap replicates with the same identity draws
for every pipeline and subgroup stratification. End-to-end detection retains
all intended mated probes, including extraction and enrolment failures.

SCRFD + ArcFace versus YuNet + SFace improves end-to-end detection by **9.50
percentage points (95% CI 7.30–12.10)**. The FPIR difference is **−0.39 points
(−0.71 to −0.11)**. Both paired intervals exclude zero. The FPIR difference
against YuNet + ArcFace is −0.04 points (−0.21 to +0.13), so a lower FPIR than
that alternative is not established. Compare paired differences directly;
overlap between separate marginal intervals is not a significance test.

These are exploratory, unadjusted comparisons designed after inspecting
benchmark results. Intervals condition on fitted models, frozen thresholds
and fixed galleries; they omit fitting, calibration, gallery-selection and
population-shift uncertainty. A zero-event bootstrap interval cannot establish
zero population risk. Standalone and paired reports can give slightly different
marginal bounds because their resampling procedures differ; retain each estimate
with its own source and use the paired report for comparative claims.

An interval excluding zero supplies exploratory statistical support under
the stated conditioning. An interval including zero leaves a difference
unresolved; it does not prove equality or equivalence. The best configuration
was identified from the evaluated results, rather than being a prespecified
universal winner. Threshold-sensitivity, gallery-size and quality trends must
be identified as descriptive unless their own comparative uncertainty is
reported. Subgroup estimates describe disparities in this benchmark, not
fairness certification or evidence for a demographic policy change.

**Suggested conclusion.** Under this project's BFW protocol, SCRFD + ArcFace
provided the highest observed end-to-end duplicate identification among the
four evaluated configurations. Relative to YuNet + SFace, its improvement and
lower false-referral rate were supported by exploratory paired intervals, at
substantially greater processing and model-storage cost. Crossed comparisons
indicated contributions from both configured detector and recogniser changes.
The study demonstrates why conditional recognition accuracy alone is
insufficient for selecting a duplicate-profile screening pipeline; it does
not establish universal superiority, an economically optimal deployment,
or proof of fraudulent activity. The separate controlled settings study
establishes processing effects on fixed benchmark images; it does not establish
that those settings improve duplicate identification in a deployment population.

## 7. Remaining thesis revisions from 19 October

The supplied draft PDF contains 694 pages. Pasted source begins on actual PDF
page 52, although the contents list gives different appendix labels and page
numbers. Remove the source listing and use the repository entry point and a
specific revision as the implementation reference. Reduce code screenshots on
pages 18–23 where prose can explain the methodological decision. Then rebuild
the contents, appendix labels, figure numbers and cross-references. Keep only
technical detail that supports the research argument.

The cover states 11,290 main-body words. The module handbook and its chapter
expectations were not supplied, so neither the declared count nor compliance
with chapter limits has been verified. Count the revised chapters using the
handbook's inclusion rules and compare them with the actual requirements;
do not invent chapter quotas. Removing code does not by itself verify the
remaining word count.

Page 24 contains a verification note about private-file tests and fresh
benchmark reproduction; page 46 contains a note about the ethics approval
record. Resolve these using dated test output and the actual authorised
approval scope. Passing dataset-free software tests cannot establish fresh
benchmark reproduction or ethics approval. The repository revision updates
code, documentation and generated interpretation; the supplied thesis PDF is
unchanged and these submission tasks remain outstanding.

In the final evaluation, keep three claim levels explicit. Measured results
describe saved benchmark outcomes; paired intervals support only the stated
exploratory comparisons. Budget, gallery and subgroup patterns remain
descriptive where comparative uncertainty is absent. General deployment,
fairness, reviewer effectiveness, fraud prevention and an economic optimum
require further investigation.

## Requirement-to-evidence map

| Supervisor request | Where it is addressed | Limit retained in the report |
| --- | --- | --- |
| Justify the contribution beyond face lookup | Section 1; generated report introduction | Applied empirical contribution; no unsupported first-of-its-kind claim. |
| Clarify threshold calibration and validation | Section 2; generated report section 13; saved threshold JSON | Development targets can be missed on test; transfer differs from recalibration. |
| Discuss the three datasets' generalisability | Section 3; generated report dataset scope | No independent deployment-population validation. |
| Analyse accuracy, latency and storage | Section 4; generated report section 11 | Estimated crossed timings, weight-only storage and referral-based workload. |
| Explain detector failures in relation to settings | Section 5; generated report section 14; controlled detector-settings report | Within-detector processing interventions measured; extra detections unannotated and replacement recognition policies unvalidated. |
| Support the conclusion with uncertainty | Section 6; paired statistical report | Exploratory paired intervals with stated conditioning and no multiplicity adjustment. |
| Critically synthesise competing approaches | Section 1, critical synthesis | Specific empirical gap; no new general method, training loss or fairness claim. |
| Justify dataset selection and BFW construction | Section 3; README protocol recipe; generated response | Project-defined identification task, correlated probes, lexical selection and one identity split. |
| Explain limits of component control | Section 5; generated response | Settings, alignment and separately calibrated policies remain part of each configuration. |
| Relate thresholds to moderation and sensitivity | Section 2; generated sensitivity summary | Illustrative budgets, actual test overruns and unavailable non-primary comparator results. |
| Bound subgroup and final-evaluation claims | Sections 3 and 6; generated claim boundaries | Small identity samples, descriptive disparities and no deployment fairness inference. |
| Remove source listings and check chapter lengths | Section 7 | Thesis editing and handbook-based counts remain outstanding. |
| Confirm tests and approval evidence | Section 7 | No claim of fresh inference reproduction or approval without records. |

Primary-source context was reviewed for this revision on 19 October 2026.
The linked FRTE page is updated over time; its role here is methodological
context, not a source of this project's performance figures. This guide
interprets the cited work alongside the saved local evidence; it is not an
exhaustive literature review or a claim of priority.
