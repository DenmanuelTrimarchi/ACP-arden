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
face-recognition model. Nothing here is trained or fine-tuned, no website is
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
| `FACE_AGEDB_ROOT` | Optional. Flat AgeDB directory, for the cross-dataset transfer test |
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
results/aggregate/agedb_transfer_metrics.json   (only when AgeDB is configured)
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
requirements-comparison.txt  optional pipeline comparison; deliberately unpinned
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

#### AgeDB cross-dataset transfer (`FACE_AGEDB_ROOT`)

Implemented. Distributed for non-commercial research on request from its
authors; when unset the run prints a skipped-with-reason line and fabricates
nothing.

The transfer applies the **BFW-frozen policy unchanged** — no AgeDB identity
contributes to threshold selection — at a gallery size matched to the BFW
held-out test. Enrolment takes each subject's youngest images and probes take
the oldest, maximising the age gap, and the artefact reports that gap's
distribution alongside coverage and FPIR/FNIR/TPIR. A poor result here means
*the policy did not transfer*, not that AgeDB is intrinsically harder.

AgeDB filenames embed subjects' real names (`<index>_<name>_<age>_<gender>.jpg`),
so this adapter is the one place where a filename must never reach an artefact.
Only opaque identifiers, ages and gaps are published, and the parser withholds
the offending filename even in its own error messages.

#### Higher-capacity pipeline comparison (`FACE_ARCFACE_MODEL_ROOT`)

The interface is implemented; the comparison is reported as **not run**. The
InsightFace pretrained recognition models are licensed for non-commercial
research and the project directs users to contact it regarding licensing, and
those terms are unresolved here. No substitute model is used.

Setting the variable is not sufficient. The SHA-256 digests of the approved
weight files must also be pinned in source (`ARCFACE_DETECTOR_SHA256`,
`ARCFACE_RECOGNITION_SHA256`), because a reportable evaluation never accepts
digests as command-line arguments. Optional dependencies live in
`requirements-comparison.txt`, deliberately unpinned until an environment
actually produces a result.

Any such comparison is a **complete-pipeline** comparison — detector,
preprocessing and embedding all differ — so a difference can never be
attributed to the embedding model alone.

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
