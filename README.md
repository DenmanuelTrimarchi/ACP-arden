# ACP-arden

A single-file, locally runnable research artefact that measures how well a
fixed, pretrained face-verification pipeline can decide whether two
unconstrained facial images show the same person, and whether the same
similarity signal can surface duplicate profiles in a 1:N gallery under a
human-review decision policy.

Everything lives in one executable Python file, [ACP_arden.py](ACP_arden.py).

## Research question

> To what extent can gallery-specific threshold calibration and multi-image
> profile enrolment reduce false duplicate-profile reviews while retaining
> duplicate-detection performance in an open-set face-verification proof of
> concept evaluated on real public benchmark datasets?

The five baseline experiments below answer the original, narrower question and
are retained unchanged as the **baseline study**:

> How effectively can a pretrained face-embedding model verify whether two
> unconstrained facial images belong to the same person and identify potential
> duplicate profiles under a human-review decision policy?

The baseline study establishes the problem the revised question addresses: a
threshold calibrated for 1:1 verification transfers badly to 1:N search,
referring a large share of genuinely new identities for review. The
supplementary experiment evaluates whether gallery-specific calibration and
multi-image enrolment can reduce false human-review referrals while retaining
sensitivity to same-identity duplicate profiles. It does **not** claim that
duplicate-profile detection is solved.

This is not a dating application, not a fraud detector, and not a new
face-recognition model. **No face-detection or face-recognition network is
trained or fine-tuned**; those models are pretrained and frozen. Experiment 7
trains a small logistic-regression review classifier using BFW development
identities only, and no test identity is used during training. No website is
scraped, and no account is ever banned, rejected, accused or classified as a
scam. A similarity above the operating threshold opens a case for a human
reviewer and nothing more.

## What it does

Five experiments, run in a fixed order:

1. **Threshold candidates** from LFW `pairsDevTrain.txt` only. Six candidate
   strategies are produced; none is selected.
2. **Development selection and freezing** on LFW `pairsDevTest.txt`. Exactly
   one candidate is selected by a fixed deterministic rule and the artefact is
   marked `frozen`.
3. **Final LFW evaluation** on the untouched `pairs.txt`, using the frozen
   threshold with no recalibration.
4. **Raw CPLFW cross-pose generalisation** using that same frozen threshold.
5. **1:N duplicate-profile gallery** built deterministically from real LFW
   images, with calibration images excluded and opaque identifiers throughout.

A final evaluation refuses to run against a threshold whose status is not
`frozen`. That refusal is the held-out boundary, enforced in code rather than
only described in prose.

## Quick start

Open this folder in VS Code, select `.venv/bin/python` as the interpreter,
open `ACP_arden.py`, and press the **Run Python File** play button. A menu
appears; nothing long-running starts until you choose an option.

```text
ACP-arden — Face Verification Research Artefact

1. Check local environment
2. Verify models and benchmark datasets
3. Run the complete five-experiment evaluation
4. Show the existing results summary
5. Launch the local human-review interface
6. Run synthetic self-tests
7. Exit
8. Run BFW open-set development and held-out evaluation
9. Show open-set results summary
10. Train and evaluate the machine-learning review classifier
11. Show review-classifier summary
12. Compare pretrained pipelines
13. Run both extension experiments and regenerate figures

Select an option:
```

### Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # tests and type checking
python ACP_arden.py
```

Any Python 3.11 or later works; 3.13 is the version the published results were
produced on.

### Command line

No argument is ever mandatory. Every menu option has a non-interactive
equivalent:

```bash
python ACP_arden.py --mode menu       # the default
python ACP_arden.py --mode check      # interpreter, pinned dependencies, configuration
python ACP_arden.py --mode verify     # model digests, LFW and raw CPLFW protocols
python ACP_arden.py --mode full       # the complete five-experiment evaluation
python ACP_arden.py --mode summary    # headline figures from the existing results
python ACP_arden.py --mode review     # the local human-review interface
python ACP_arden.py --mode self-test  # deterministic synthetic tests, no data needed

# Supplementary Experiment 6 (needs the official BFW dataset)
python ACP_arden.py --mode open-set          # development, freezing, held-out test
python ACP_arden.py --mode open-set-summary  # headline open-set figures

# Extension experiments 7 and 8
python ACP_arden.py --mode ml-review                 # train, freeze, evaluate the classifier
python ACP_arden.py --mode ml-review-summary         # headline classifier figures
python ACP_arden.py --mode pipeline-compare          # pretrained pipeline comparison
python ACP_arden.py --mode pipeline-compare-summary  # its status
python ACP_arden.py --mode extensions                # both, then regenerate figures
```

`--mode full` continues to mean exactly the original five-experiment
evaluation. The open-set experiment is separate and never alters it.

`--mode self-test` and the pytest suite need no dataset, no model file and no
network, so the code can be checked before any biometric data is touched.

## Datasets and models

Neither the benchmark images nor the two model binaries are stored in this
repository, and nothing here downloads them automatically. Their locations are
read from a git-ignored `.env` beside `ACP_arden.py`:

```bash
cp .env.example .env   # then fill in your own storage paths
```

| Variable | Contents |
|---|---|
| `FACE_DATA_ROOT` | Extracted dataset image directories, including `lfw_funneled/` |
| `FACE_PROTOCOL_ROOT` | `pairsDevTrain.txt`, `pairsDevTest.txt`, `pairs.txt`, `pairs_CPLFW.txt` |
| `FACE_CPLFW_RAW_ROOT` | Flat directory of the authors' raw CPLFW images |
| `FACE_MODEL_ROOT` | The two pinned `.onnx` model files |
| `FACE_ID_HMAC_KEY` | **Required.** Secret key for the opaque public identifiers |
| `FACE_CACHE_ROOT` | Optional embedding cache; leave unset by default |
| `FACE_BFW_ROOT` | Optional. Extracted BFW tree, for Experiment 6 |
| `FACE_BFW_METADATA_ROOT` | Optional. Where the official BFW datatable lives; defaults to `FACE_BFW_ROOT` |
| `FACE_ARCFACE_MODEL_ROOT` | Optional. InsightFace weights; the comparison is reported as not run |

**Datasets.** LFW (Labeled Faces in the Wild) is the primary dataset; CPLFW
(Cross-Pose LFW) is the secondary one. CPLFW ships two non-interchangeable
image sets — the authors' raw, unconstrained images (`images.rar`) and a
separately pre-cropped and aligned copy (`cp-aligned.zip`). Every result
records which variant it scored; the published figures use `raw`.

**Models.** Two pinned OpenCV Zoo files, verified by SHA-256 before either is
loaded, and refused rather than used if a digest does not match:

```text
face_detection_yunet_2023mar.onnx    MIT
  sha256 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
face_recognition_sface_2021dec.onnx  Apache-2.0
  sha256 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
```

**Ethics.** Face images and embeddings are biometric data. Public availability
of a dataset does not by itself satisfy an ethics or data-protection
requirement. Confirm your own approval position before downloading, opening
with a face model, or embedding any real image.

## Results

Produced by a complete local run against LFW and the raw CPLFW image set. Every
figure below is read from `results/aggregate/*.json` by `--mode summary`; none
is transcribed by hand.

```text
Final LFW accuracy: 99.09%
Final LFW false-match rate: 0.11%
Final LFW false-non-match rate: 1.71%
Final LFW EER: 0.78%
Final LFW extraction-failure rate: 10.02%

Raw CPLFW conditional accuracy: 90.24%
Raw CPLFW scored pairs: 3,515 / 6,000
Raw CPLFW failed pairs: 2,485
Raw CPLFW extraction-failure rate: 41.42%
Raw CPLFW false-match rate: 1.95%
Raw CPLFW false-non-match rate: 17.46%
Raw CPLFW EER: 9.77%

Intended gallery entries: 1,047, enrolled 986
Gallery enrolment-failure rate: 5.83% (61 references never enrolled)
Duplicate detection rate (conditional): 99.36%
Duplicate detection rate (end-to-end): 89.40%
Mated probes with no enrolled reference: 61
Rank-1 identification rate: 97.98%
False duplicate-review rate: 52.56%
```

Two figures must never be quoted alone:

- The **CPLFW accuracy is conditional** on the 3,515 pairs that yielded exactly
  one detectable face on both sides. The other 41.42% of the protocol never
  reached the comparison stage. An extraction failure is not a verification
  error — the pipeline produced no score at all, so those pairs can be neither
  correct nor incorrect — but omitting the rate would make the accuracy look
  better than it is. Cross-pose *detection*, not comparison, is the dominant
  finding, and it is present in the authors' raw images rather than being an
  artefact of pre-cropping.
- The **gallery detection rate is meaningless without its denominator**. 61 of
  1,047 gallery references never enrolled — 52 images yielded multiple faces
  and 9 yielded none — so the mated probes pointing at them had nothing to
  match against. Counting those as detection misses (as the pre-revision
  artefact did) understates the conditional rate; ignoring them overstates the
  real one. Both are now published: **99.36% conditional** over probes that
  were actually scored, **89.40% end-to-end** over every intended probe. The
  code refuses to print either alone.
- The **gallery detection rate is also inseparable from the 52.56% false
  duplicate-review rate**. A 0.11% single-comparison false-match rate compounds
  across 986 gallery comparisons per probe. That is direct, quantified evidence
  that a threshold calibrated for 1:1 verification is not fit for 1:N search at
  this scale without its own calibration — and the evidence base for this
  project's human-review-only policy.

Full write-up: [results/aggregate/FINAL_EVALUATION_REPORT.md](results/aggregate/FINAL_EVALUATION_REPORT.md).

### Published outputs

```text
results/aggregate/calibrated_threshold.json
results/aggregate/lfw_development_metrics.json
results/aggregate/lfw_final_metrics.json
results/aggregate/cplfw_metrics.json
results/aggregate/duplicate_gallery_metrics.json
results/aggregate/run_manifest.json
results/aggregate/metrics_summary.csv
results/aggregate/confusion_matrices.csv
results/aggregate/roc_points.csv
results/aggregate/FINAL_EVALUATION_REPORT.md
```

The corrected gallery accounting and Experiment 6 add, without altering any of
the above:

```text
results/aggregate/duplicate_gallery_metrics_v2.json
results/aggregate/bfw_open_set_protocol_summary.json
results/aggregate/bfw_open_set_threshold.json
results/aggregate/bfw_open_set_development_metrics.json
results/aggregate/bfw_open_set_test_metrics.json
results/aggregate/bfw_subgroup_metrics.csv
results/aggregate/open_set_confidence_intervals.json
results/aggregate/open_set_method_comparison.csv
results/aggregate/pretrained_pipeline_comparison.csv
results/aggregate/OPEN_SET_EVALUATION_REPORT.md
```

`duplicate_gallery_metrics.json` is retained unchanged for provenance. It
predates the corrected accounting, so its detection rate is conditional only;
the summary says so rather than letting that figure stand unqualified.

Each JSON artefact embeds its own provenance: the software environment, both
model digests, the protocol digest, a digest of exactly which images were
evaluated, and the dataset archive checksum. `run_manifest.json` records the
environment-variable *names* and a SHA-256 of every other output — never an
absolute path. A run refuses to finish if any published output contains a
personal or absolute filesystem path.

## Local human-review interface

```bash
python ACP_arden.py --mode review
```

Re-runs this same file under Streamlit, bound to `127.0.0.1`, reading
`results/raw/review.sqlite` (git-ignored). The page shows only opaque one-way
identifiers, a similarity score and the threshold that opened the case — never
a name, a file path or an embedding. It states plainly that similarity does not
prove misuse, and it applies no sanction of any kind.

A complete run opens roughly 2,500 cases, so the page is filterable by status
and shows a bounded slice — the highest-similarity cases first, 25 by default.
Each row is a prompt for a human decision, not a finding.

## Testing

```bash
python -m py_compile ACP_arden.py
python ACP_arden.py --mode self-test
pytest -v
pyright                 # configuration is pinned in pyrightconfig.json
```

The self-tests cover cosine similarity, L2 normalisation, confusion-matrix
accounting, the derived rates, ROC-AUC, EER, candidate generation,
deterministic selection, rejection of non-frozen thresholds, failure
accounting, gallery role uniqueness, opaque-ID stability, path-leak detection
and deterministic gallery sampling. The pytest suite in
[tests/](tests/) additionally pins the methodology contract: model digests,
detector settings, embedding dimensionality, the random seed and the exact
selection rule.

## Layout

```text
ACP_arden.py        the entire programme
CONVERSION_MAP.md   component-to-section map
REFERENCES.md       code attribution and reference register
requirements.txt        pinned runtime dependencies
requirements-dev.txt    pinned test and type-checking dependencies
requirements-comparison.txt       optional comparison; install via the script below
requirements-comparison-deps.txt  exact InsightFace dependency pins
scripts/install_comparison_environment.sh  verified installer
pyrightconfig.json  type-checker configuration
.env.example        template for local storage paths
.vscode/            run configuration for the play button
results/aggregate/  published, privacy-scanned results
results/raw/        local, git-ignored working files
tests/              contract tests, no dataset or model required
```

## Supplementary experiment 6 — BFW open-set duplicate-profile evaluation

The baseline gallery experiment reuses the LFW 1:1 threshold for 1:N search.
That is a *control*, and it fails in an informative way: a threshold chosen to
compare exactly two images refers a large share of people with no gallery match
for human review, because one non-mated search performs as many comparisons as
there are enrolled identities. Experiment 6 asks whether calibrating for the
gallery, and enrolling more than one image per profile, fixes that.

### What is new

1. **Corrected gallery accounting.** A reference image that fails to enrol is
   recorded, not dropped. Mated probes whose reference never enrolled are
   reported as `gallery_reference_unavailable`, never as similarity misses.
   Both a `conditional` and an `end_to_end` rate are always published.
2. **Identity-disjoint open-set protocol** built from official BFW data,
   stratified by the eight demographic subgroups, seeded at 20260727.
3. **Two enrolment methods** over the same identity partition.
4. **Open-set calibration** at target FPIRs, frozen before the held-out test.
5. **Cluster-bootstrap confidence intervals** and subgroup reporting.

### Metric definitions, and why they are not the pairwise ones

| Open-set (1:N) | Pairwise (1:1) | Difference |
| --- | --- | --- |
| **FPIR** — a search against the whole gallery returns at least one candidate above threshold when the person is *not* enrolled | **FMR** — one comparison between two images exceeds threshold | FPIR aggregates over every gallery identity, so it compounds with gallery size; FMR does not |
| **FNIR@k** — a mated search fails to place the correct identity within rank *k* above threshold | **FNMR** — one genuine comparison falls below threshold | FNIR involves ranking against competing candidates; FNMR has no competitors |
| **TPIR@k** (= DIR) — `1 − FNIR@k` | **TMR** — `1 − FNMR` | — |

An FPIR is never comparable with an FMR, and the artefacts keep them in
separate files to make substitution difficult.

### The two methods

| | Method A (control) | Method B (proposed) |
| --- | --- | --- |
| Name | `single_image_pairwise_threshold` | `three_image_open_set_calibrated` |
| Gallery images per identity | 1 | up to 3, minimum 2 |
| Representation | one normalised embedding | mean of L2-normalised embeddings, re-normalised |
| Threshold | the LFW 1:1 frozen threshold, unchanged | calibrated on the BFW development partition at a target FPIR |

Method A exists to quantify threshold transfer failure. Its threshold is a
**control only and is never described as a valid open-set operating point.**

### Operating points

Candidate thresholds are every distinct development top score plus two
bracketing sentinels. For each target FPIR (0.001, **0.003 primary**, 0.01) the
rule is: keep candidates whose development FPIR is at or below the target, then
take the highest development TPIR@1; ties break by lower development FPIR, then
higher threshold, then candidate name.

The policy is written with status `open_set_frozen`, and
`require_frozen_open_set_policy` refuses to score the held-out test partition
with anything else.

### Confidence intervals

Several probes can belong to one identity, so images are **not** resampled
independently — that would treat correlated observations as independent and
produce intervals that are far too narrow. Identities are resampled with
replacement instead (cluster bootstrap), 2,000 replicates at seed 20260727,
subgroup stratification preserved, reported as 2.5/97.5 percentiles. A
replicate in which a metric is undefined is excluded and counted; it is never
replaced by zero.

### Subgroup analysis

Per-subgroup FPIR, FNIR@1, TPIR@1 and coverage, using the dataset's own
aggregate annotations. No attribute is inferred with another model, and no
subgroup label is published beside anything identifying a person. One global
threshold is used for the primary result; subgroup-specific thresholds are not
applied to the held-out test. The max/min FPIR ratio is reported only when the
denominator is non-zero — otherwise the absolute range is given instead.

### Setup

```bash
FACE_BFW_ROOT=/path/to/bfw-images            # extracted <subgroup>/<identity>/<image>.jpg
FACE_BFW_METADATA_ROOT=/path/to/bfw-metadata # optional; defaults to FACE_BFW_ROOT
FACE_ID_HMAC_KEY=...                         # required; see below
```

BFW must be obtained from the [official project](https://github.com/visionjo/facerec-bias-bfw)
under its own terms. Nothing here downloads it, and no mirror is used. The
adapter pins the official datatable schema and stops with an explicit message
rather than guessing at a variant.

**This project defines its own open-set protocol from the official BFW data.
BFW publishes verification and bias-analysis protocols; it does not publish an
open-set identification protocol, and none is implied.**

### Optional extensions

#### AgeDB cross-dataset transfer — withdrawn, not pursued

An AgeDB transfer test was implemented and then **deliberately removed**. It is
not part of this artefact: there is no adapter, no configuration variable, no
execution mode and no result file, and contract tests assert that none returns.

It is recorded here only so the git history is interpretable. No AgeDB claim
appears in the research conclusions.

#### Complete-pipeline comparison (`FACE_ARCFACE_MODEL_ROOT`)

Experiment 8 compares two **complete pipelines** — detection, alignment,
preprocessing and embedding — over the identical BFW protocol:

| | Pipeline A | Pipeline B |
| --- | --- | --- |
| Detector | OpenCV YuNet 2023mar | InsightFace SCRFD `det_10g` |
| Recognition | OpenCV SFace 2021dec | InsightFace ArcFace `w600k_r50` |
| Embedding width | 128 | 512 |

Each receives **its own development-only threshold**, frozen before the held-out
identities are scored. Similarity scores from different embedding models are not
interchangeable, so the SFace threshold is never applied to ArcFace. Neither
network is trained or fine-tuned. No difference may be attributed to the
embedding model alone, because the detector and preprocessing differ too.

**Licensing.** InsightFace publishes its pretrained models for non-commercial
research use, and this artefact is non-commercial MSc academic research, so that
use falls within those terms. The models were created and trained externally;
the MIT licence covering InsightFace source code does not automatically extend
to the weight files, and no ownership of the models, their training data or
their weights is claimed.

**Setup.** Obtain the official `buffalo_l` pack, store it in local research
storage outside version control, point `FACE_ARCFACE_MODEL_ROOT` at the
directory *directly* containing `det_10g.onnx` and `w600k_r50.onnx`, pin both
SHA-256 digests in source, and install `requirements-comparison.txt`.

Nothing is downloaded automatically. Each ONNX file is loaded by its exact
verified path through `insightface.model_zoo`; `FaceAnalysis` is deliberately
not used, because it resolves models through a cache directory and fetches the
pack over the network when that directory is empty — which would both download
automatically and evaluate files other than the pinned ones.

One preprocessing note: SCRFD rescales input to a fixed square, and InsightFace's
default of 640 detects nothing on BFW's ~100-pixel crops. The input size is
pinned to 320; the detection threshold stays at the published default, so
coverage is not inflated by lowering the decision bar.

#### Held-out outcome — extraction, not ranking

Status **`evaluated_non_commercial_academic_research`**. Both pipelines scored once on the same held-out
identities, each under its own frozen development threshold:

| | YuNet + SFace | SCRFD + ArcFace |
| --- | --- | --- |
| Frozen threshold | 0.477118 | 0.393958 |
| FPIR | 0.52% | **0.13%** |
| TPIR@1 | 92.57% | **96.80%** |
| False reviews / 1,000 | 5.25 | **1.34** |
| End-to-end detection | 87.20% | **96.70%** |
| Gallery coverage | 99.00% | **100.00%** |
| Complete-pipeline latency | **22.0 ms** | 96.0 ms |

**The difference is extraction, not ranking.** YuNet failed to detect a face in
189 images; SCRFD failed in
2. CMC rank-1 is near-identical
(98.09% against 98.40%), so both
models rank about equally well *given* an embedding. ArcFace's advantage comes
overwhelmingly from succeeding on BFW's small, awkward crops — bought with
roughly 4.4× the per-image cost and five times the disk.

It also fails differently: 12
multiple-face detections where YuNet had
0. Neither network is trained or
fine-tuned, and no difference is attributable to the embedding model alone.

### Pre-declared success criteria

Declared in source before the held-out test was run, and reported as achieved,
not achieved, or not measurable:

| Criterion | Target |
| --- | --- |
| Held-out FPIR | ≤ 0.01, target 0.003 |
| TPIR@1 | ≥ 0.90 |
| TPIR@5 | ≥ 0.95 |
| Gallery enrolment coverage | ≥ 0.90 |
| Probe extraction coverage | ≥ 0.90 |

These are research targets, not results, and are not revised after seeing test
outcomes.

## Extension experiments 7 and 8

### Experiment 7 — interpretable review classifier

> Can an interpretable machine-learning review classifier trained on BFW
> development identities reduce false duplicate-profile review referrals while
> retaining duplicate-detection performance compared with the existing single
> calibrated similarity threshold?

Logistic regression, chosen because it is interpretable, reproducible, suited to
a nine-feature problem, unlikely to overfit at this size, and emits a
probability that can be calibrated to a target FPIR. **No face-recognition model
is trained or fine-tuned.**

The BFW development identities are split again, by identity and stratified
across the eight subgroups: 70% fit the model, 30% calibrate its probability
threshold. The held-out test identities are untouched by both.

Features are computed from the gallery search alone — top-1 and top-2
similarity, their margin, top-5 mean and standard deviation, the images backing
the top-ranked template, gallery size, probe detection confidence and face-area
ratio. **Demographic subgroup is never an input**; it is used only for
post-evaluation fairness reporting. No identity, filename, path, raw embedding
or label-derived quantity is used. Records missing a feature are excluded and
counted, never imputed.

The model is published as plain JSON numerics — coefficients, intercept, feature
order, scaler mean and scale. No pickle is written or read.

**TPIR is rank-aware.** A mated probe counts as a rank-one identification only
when its own identity is ranked first *and* the probability clears the operating
point. A referral to some other identity is a referral, not an identification,
and is reported separately as `mated_wrong_identity_referred`. The comparator
uses the same definition, so the two are directly comparable.

#### Held-out outcome — the primary hypothesis was not achieved

The hypothesis was that the classifier would **reduce** false review referrals
while retaining detection. It did not. On the held-out partition it refers *more*
innocent registrations than the calibrated threshold, while identifying more
duplicates:

| | Calibrated threshold | Classifier |
| --- | --- | --- |
| FPIR | 0.52% | 0.70% |
| TPIR@1 | 92.57% | 94.27% |
| TPIR@5 | 92.57% | 94.37% |
| False reviews / 1,000 | 5.25 | 7.00 |

Five of six pre-declared criteria are achieved; `fewer_false_reviews_than_threshold_method`
is **not achieved**. Machine learning did not improve the outcome this project
set out to improve. The figures above are a snapshot of
`ml_review_test_metrics.json`, which remains the source of truth.

### Experiment 8 — stronger pretrained pipeline

> Does a stronger pretrained detection and face-embedding pipeline improve
> extraction coverage, open-set duplicate detection and subgroup consistency
> compared with YuNet + SFace under the same BFW protocol?

Evaluated. Status **`evaluated_non_commercial_academic_research`**, backed by
held-out metrics for both pipelines. The authoritative sources are
`results/aggregate/pipeline_comparison_metrics.json` and
`results/aggregate/pretrained_pipeline_comparison.csv`; figures quoted in this
file are a snapshot of those artefacts, and latency in particular is
environment-specific.

Each pipeline receives its own development-only threshold, because
similarity scores from different embedding models are not interchangeable — the
SFace threshold is never applied to ArcFace. Any such comparison is a
**complete-pipeline** comparison: detection, landmarking, alignment,
preprocessing, embedding dimensionality and runtime all differ, so no difference
could be attributed to the embedding model alone.

### Figures

`results/figures/` holds 15 figures, each in PNG (300 dpi) and SVG,
generated from the published JSON and CSV artefacts rather than typed values,
with PNG text metadata stripped and the privacy scan applied:

```text
duplicate_detection_by_method
false_reviews_per_1000_by_method
female_male_aggregate_comparison
female_subgroup_pipeline_comparison
implementation_layers_coverage
implementation_layers_duplicate_detection
implementation_layers_fpir
implementation_layers_performance_latency
male_subgroup_pipeline_comparison
mated_non_mated_similarity_distributions
ml_review_classifier_coefficients
open_set_operating_curve
pipeline_coverage_and_latency
profile_photo_consistency_outcomes
subgroup_fpir_tpir_with_confidence_intervals
```

`results/figures/FIGURE_CAPTIONS.md` accompanies them, stating each figure's
denominator, a short interpretation and the limitations that apply — a chart
without its sample size invites over-reading.

## Reproducibility and the canonical run

Detection is **not bit-stable across processes on this platform**, and the
project does not claim otherwise.

YuNet emits a detection score compared against a 0.9 acceptance threshold. An
image scoring near that line can be accepted in one process and rejected in the
next. Measured over 2,500 BFW images in six fresh interpreters, accepted counts
were 2238, 2238, 2238, 2239, 2239 and 2243 — roughly one image in a thousand
flips. This persists with OpenCL disabled and OpenCV forced single-threaded, so
it is floating-point variation inside OpenCV's DNN backend rather than anything
this project controls.

Two mitigations are applied, and it is worth being clear about which one carries
the guarantee:

1. `configure_deterministic_opencv()` disables OpenCL and forces single-threaded
   execution. This *reduces* variability. Note that under Apple's GCD parallel
   framework `cv2.setNumThreads(1)` is silently a no-op — only
   `setNumThreads(0)` is honoured.
2. **The canonical run cache is what actually guarantees consistency.** The
   primary pipeline's held-out run is computed once, cached in git-ignored
   storage, and reused by Experiment 6, Experiment 7's comparator and features,
   Experiment 8's primary pipeline, implementation layers 1–4, and every
   subgroup and sex breakdown. Each derived artefact publishes a shared
   non-biometric `canonical_run_digest`.

Without the cache, the same method reported three different non-mated scored
counts (2863 / 2860 / 2859) and an FPIR between 0.52% and 0.66% across
artefacts. With it, all report identically.

The cache stores decision outcomes and similarity scores only. No embedding or
biometric template is written, and it never leaves local storage.

## Opaque identifiers and the required key

Public identifiers are HMAC-SHA-256 over the identity or sample name, keyed by
a secret `FACE_ID_HMAC_KEY`. A published fixed salt would leave the mapping
back to a dataset identity recoverable by hashing a candidate name list, which
for a public benchmark is a short list.

```bash
python -c "import secrets;print(secrets.token_urlsafe(32))"
```

At least 32 decoded bytes, URL-safe base64 or hexadecimal. The key is held in
memory only: never printed, never stored in a result, and no digest of it
appears in any artefact. Rotating it changes every identifier, so an existing
local review database is refused with instructions to delete it.

Partitioning deliberately does **not** depend on the key — only on the seed and
the protocol — so published metrics stay reproducible by someone who does not
hold it.

## Limitations

LFW and CPLFW carry their own demographic skew, which limits how
representative any figure here is of a real user population.
Face-extraction failures are excluded from the accuracy metrics but reported
as their own rate, never silently dropped. The gallery experiment is
research-scale rather than production-scale. And "duplicate profile" here
means "the same face was detected in the gallery" — not a legal or
investigative finding about any person.

For the supplementary open-set experiment specifically:

- It remains a proof of concept. No result proves fraud, misuse or
  misrepresentation by anyone, and no automatic sanction is ever applied.
- The BFW open-set protocol is defined by this project, not by BFW's authors.
- Confidence intervals describe sampling uncertainty over these benchmark
  identities only; they do not extend to any other population.
- Benchmark demographics do not represent a real dating-application user
  population, so subgroup figures are not deployment estimates.
- Every rate is conditional on the coverage figures printed beside it. An FPIR
  measured over a small surviving fraction of the protocol is not comparable
  with one measured over nearly all of it, which is why the code refuses to
  print one without the other.
- Duplicate-profile detection is **not** solved by anything here.

## What this artefact is

> ACP-arden is a benchmark-validated, human-review-only academic research proof
> of concept. No face-recognition network is trained or fine-tuned. A small
> logistic-regression review classifier is trained on identity-disjoint BFW
> development data and evaluated on untouched held-out identities.

It is **not** production-ready, not unbiased, not fully secure, not capable of
proving fraud, and not capable of automatically banning anyone.

Two decision directions are involved, and a single threshold statement would
misdescribe one of them. In **duplicate-profile screening** a *high* similarity
to some other enrolled identity opens a duplicate review. In **profile-photo
consistency** a *low* similarity to the profile's own enrolled template opens an
inconsistency review. An extraction failure resolves nothing in either
direction.

## Attribution and licence

All code in this repository is original — no file contains code copied or
materially adapted from an external source. What *is* external is recorded in
full in [REFERENCES.md](REFERENCES.md), and marked in the source itself with a
four-field attribution header at each point where an external model, dataset,
API or published method enters the pipeline:

| What | Source |
|---|---|
| Face detection | Wu, Peng and Yu (2023), *YuNet: A Tiny Millisecond-level Face Detector*, Machine Intelligence Research 20(5) — weights MIT |
| Face embedding | Zhong, Deng, Hu, Zhao, Li and Wen (2021), *SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition*, IEEE TIP 30 — weights Apache-2.0 |
| Primary dataset | Huang, Ramesh, Berg and Learned-Miller (2007), *Labeled Faces in the Wild*, UMass Amherst TR 07-49; funnelled images per Huang, Jain and Learned-Miller (2007), ICCV |
| Secondary dataset | Zheng and Deng (2018), *Cross-Pose LFW*, BUPT TR 18-01 |
| ROC-AUC identity | Hanley and McNeil (1982), Radiology 143(1) |
| Libraries | OpenCV, NumPy, Pillow, Streamlit — used through their documented public APIs |

Neither dataset nor either model file is redistributed here.

Made available for academic assessment and review — see [LICENSE](LICENSE) for
the terms, which reserve all other rights.
