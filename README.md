# Duplicate face screening

I built this project to compare pretrained face detection and recognition
pipelines for duplicate-profile screening. The main question is how much of
an apparent accuracy advantage remains when failed image processing, false
review referrals and computational cost are included.

The project runs locally from one Python file, [ACP_arden.py](ACP_arden.py).
It evaluates real benchmark images and produces aggregate metrics, reports
and figures. A high similarity can open a case for human review; it does not
establish fraud, photograph ownership or legal identity, and it never applies
an automatic sanction.

## What the project includes

- Four detector–recogniser combinations: YuNet + SFace, SCRFD + SFace,
  YuNet + ArcFace and SCRFD + ArcFace.
- LFW one-to-one verification and frozen-threshold transfer to raw CPLFW.
- BFW gallery search with separate development and held-out identities.
- One-image and three-image enrolment, gallery-specific calibration and a
  logistic-regression review classifier.
- Processing coverage, rank-aware identification, false referrals, latency
  and model storage alongside conditional accuracy.
- Paired identity-bootstrap comparisons, threshold sensitivity, gallery-size
  analyses and controlled detector-setting experiments.
- A local Streamlit interface for reviewing referrals.

No face-recognition network is trained or fine-tuned. The detectors and
embedders use frozen pretrained weights. The logistic-regression review
classifier is fitted only on development identities.

## Setup

The pinned dependencies require Python 3.12 or later. The saved benchmark runs
used Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Set the local storage paths in `.env`:

- `FACE_DATA_ROOT`: directory containing `lfw_funneled/`.
- `FACE_PROTOCOL_ROOT`: official `pairsDevTrain.txt`, `pairsDevTest.txt`,
  `pairs.txt` and `pairs_CPLFW.txt` files.
- `FACE_CPLFW_RAW_ROOT`: extracted raw CPLFW images from `images.rar`.
- `FACE_MODEL_ROOT`: directory containing the pinned YuNet and SFace weights.
- `FACE_BFW_ROOT`: extracted BFW image tree, needed for Experiments 6–8,
  Experiments 11–12 and the BFW analyses.
- `FACE_BFW_METADATA_ROOT`: BFW metadata directory; defaults to `FACE_BFW_ROOT`.
- `FACE_ARCFACE_MODEL_ROOT`: directory directly containing the SCRFD and
  ArcFace weights, needed for Experiments 8–12 and relevant comparisons.
- `FACE_ID_HMAC_KEY`: a private key for opaque sample and identity identifiers.

Generate an identifier key and put it in `.env`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The key must decode to at least 32 bytes. It is never written into published
results. Changing it changes identifiers and invalidates an existing local
review database. `FACE_CACHE_ROOT` is optional; an embedding cache contains
biometric templates and belongs in private storage.

### Datasets and weights

Obtain each dataset from its official source under its own terms:

- [LFW](https://vis-www.cs.umass.edu/lfw/): funnelled images and official pair files.
- [CPLFW](https://www.whdeng.cn/CPLFW/): raw images and pair protocol.
- [BFW](https://github.com/visionjo/facerec-bias-bfw): official images and datatable.

The raw and pre-aligned CPLFW image sets are different inputs. The saved
transfer results use the raw images, so detection and alignment remain part
of the evaluation.

The required model filenames are:

```text
face_detection_yunet_2023mar.onnx
face_recognition_sface_2021dec.onnx
```

YuNet is obtained from OpenCV Zoo's face-detection model directory, and SFace from its
face-recognition model directory. The loader checks SHA-256 digests before
using either model. Model sources, exact filenames, hashes and licences are
listed in [REFERENCES.md](REFERENCES.md).

For the optional SCRFD + ArcFace components, obtain the official InsightFace
`buffalo_l` weights and store `det_10g.onnx` and `w600k_r50.onnx` in
`FACE_ARCFACE_MODEL_ROOT`. Install the comparison dependencies with:

```bash
bash scripts/install_comparison_environment.sh
```

This installer keeps the pinned `opencv-python-headless` build in place and
checks the imported OpenCV module. Installing InsightFace's dependencies
without this control can install a conflicting `opencv-python` distribution.
`requirements-comparison-deps.txt` records direct dependency pins;
`requirements-comparison-lock.txt` records the verified environment's full
package set.

Datasets, model weights, private caches and `.env` are excluded from Git.
The programme does not download them automatically. Model-weight terms are
separate from library source-code licences; the comparison weights were used
for non-commercial academic research and are not redistributed here.

## Running the project

```bash
python ACP_arden.py
```

This opens the interactive menu. In VS Code, select `.venv/bin/python`, open
`ACP_arden.py` and use **Run Python File** for the same menu.

Every workflow also has a command-line mode:

```bash
# Environment, inputs and dataset-free checks
python ACP_arden.py --mode check
python ACP_arden.py --mode verify
python ACP_arden.py --mode self-test

# Experiments 1–5: LFW, raw CPLFW and the LFW gallery control
python ACP_arden.py --mode full
python ACP_arden.py --mode summary

# Experiments 6–8: BFW gallery calibration, classifier and pipeline comparison
python ACP_arden.py --mode open-set
python ACP_arden.py --mode ml-review
python ACP_arden.py --mode pipeline-compare

# Experiments 9–12: comparison verification, classifier and crossed components
python ACP_arden.py --mode verification-compare
python ACP_arden.py --mode arcface-review
python ACP_arden.py --mode mixed-pipelines

# Controlled comparisons and reports
python ACP_arden.py --mode comparison-diagnostics
python ACP_arden.py --mode detector-settings
python ACP_arden.py --mode comparative-statistics
python ACP_arden.py --mode refresh-reports
python ACP_arden.py --mode experiment-table
```

`--mode all` runs setup checks, all twelve experiments, controlled comparisons
and report generation in dependency order. It requires all datasets and model
files, can take a long time, updates saved results and stops at the first
failed step. `--mode full` runs only Experiments 1–5.

Use `--results-root PATH` to choose an aggregate output directory.
`python ACP_arden.py --help` lists the available modes, including the saved
summary views. Existing reports can be read without rerunning the models.

## Evaluation method

### Verification

LFW development files generate and select a transfer threshold, which is then
frozen. Final LFW evaluation uses the official ten folds, fitting each fold's
threshold on the other nine folds. A development threshold is not applied to
all final pairs because the development and final protocol files overlap.
The separately frozen development threshold is applied unchanged to raw
CPLFW to measure transfer across image conditions.

### Gallery search

BFW uses a project-defined open-set protocol, not an official BFW identification
benchmark. With seed `20260727`, each subgroup's identities are split equally
between development and test, then between enrolled and non-mated roles.
Splits use sorted identities and subgroup-specific seeded shuffles, independent
of the identifier key. Odd counts go to development and then the enrolled side;
identities with too few images for enrolment remain non-mated.

Images are sorted by filename within an identity. The first three are enrolment
candidates and the next five are mated probes. Non-mated identities contribute
up to fifteen probes and never enter the gallery. The three-image method needs
at least two successful enrolment images; the one-image control uses the first.

Each partition of the evaluated corpus contains 400 identities, 200 intended
gallery identities, 600 enrolment images, 1,000 mated probes and 3,000 non-mated
probes. The protocol uses 9,200 of BFW's 20,000 images. The probe ratio is an
experimental sampling choice, not an estimate of duplicate prevalence.

Each configured pipeline gets its own development-calibrated threshold. The
FPIR targets are 0.1%, 0.3% and 1%, with 0.3% as the primary operating point.
These are research budgets, not validated moderation requirements. Thresholds
are frozen before held-out evaluation and are never selected from test results.

### Metrics and uncertainty

Conditional rates use successfully scored cases. End-to-end duplicate detection
uses all intended mated probes, including extraction and enrolment failures.
FPIR is measured per non-mated gallery search and differs from a pairwise
false-match rate.

TPIR is rank-aware: the correct identity must be within the stated rank and
above threshold. Referring a different identity does not count as correct
duplicate identification; those cases are recorded as
`mated_wrong_identity_referred`.

Multiple probes from one person are correlated. Confidence intervals therefore
resample identities rather than independent images. The paired comparisons use
2,000 shared identity-bootstrap draws with subgroup stratification. Their
intervals are exploratory, unadjusted for multiple comparisons and conditional
on fitted models, frozen policies and fixed galleries. An interval containing
zero leaves a difference unresolved; it does not establish equivalence.

## Results

The saved BFW comparison has status
`evaluated_non_commercial_academic_research`:

- **YuNet + SFace:** 87.20% end-to-end duplicate detection, 92.57% conditional
  TPIR@1 and 0.52% FPIR, equivalent to 5.25 false referrals per 1,000 scored
  non-mated searches. Complete-pipeline latency averages 21.72 ms per image;
  model weights occupy approximately 37.1 MiB.
- **SCRFD + ArcFace:** 96.70% end-to-end duplicate detection, 96.80% conditional
  TPIR@1 and 0.13% FPIR, equivalent to 1.34 false referrals per 1,000 scored
  non-mated searches. Complete-pipeline latency averages 80.73 ms per image;
  model weights occupy approximately 182.4 MiB.

The end-to-end difference is 9.50 percentage points, with an exploratory paired
95% interval of 7.30–12.10. Both configured components contribute in the crossed
comparison, but different detector settings, alignment, preprocessing and
separately calibrated thresholds remain part of those configurations. These
results do not isolate an architecture or training-loss effect.

The baseline classifier's primary hypothesis was not achieved: false referrals
increased from 15 to 20, despite higher conditional TPIR@1. Its effect differs
on the ArcFace pipeline, so there is no general claim that the classifier
improves every configuration.

The evaluation was repeated only to test computational reproducibility;
no repeated held-out result influenced model selection, threshold selection
or the reported policy.

The source measurements are
[pipeline_comparison_metrics.json](results/aggregate/pipeline_comparison_metrics.json)
and [pretrained_pipeline_comparison.csv](results/aggregate/pretrained_pipeline_comparison.csv).
The [research report](results/aggregate/RESEARCH_REPORT.md) brings together the
verification, gallery, classifier, sensitivity and cost results. Figures and
their captions are in [results/figures/](results/figures/) and
[FIGURE_CAPTIONS.md](results/figures/FIGURE_CAPTIONS.md).

## Reproducibility and limitations

Seeds, dataset fingerprints, model hashes, detector settings, dependency versions
and frozen thresholds are recorded with the results. Detection is not bit-stable
across processes on the recorded platform: borderline YuNet detections can change
between fresh runs. OpenCL is disabled and OpenCV uses `setNumThreads(0)`, but
shared experiments also reuse a private canonical outcome cache. The cache keeps
those experiments consistent; it does not guarantee identical fresh inference
on another machine.

LFW and CPLFW share identities, and none of the datasets represents a dating
service's upload population. BFW's balanced labels support descriptive subgroup
comparisons, not a fairness certification. Its held-out subgroups have only
25 non-mated identities each, despite repeated probe images.

Lexical image selection, one identity split, fixed research-scale galleries and
unaudited pretraining overlap limit generalisation. The development FPIR budget
can be exceeded on test. Non-primary SCRFD + ArcFace held-out operating points
are absent from the original saved comparison; report generation leaves them
unavailable rather than substituting development rates. Updated comparison
runs save all pre-frozen operating points separately.

Latency is specific to the recorded machine and software. Mixed-pipeline timing
summaries are stage-based estimates, and storage figures cover weights only.
False referrals are a workload proxy: reviewer time, decision quality, real
prevalence and deployment costs were not measured.

## Human review

```bash
python ACP_arden.py --mode review
```

The local interface binds to `127.0.0.1` and reads the ignored
`results/raw/review.sqlite` database. It displays opaque identifiers, scores,
thresholds and case status, without benchmark photographs, names or embeddings.

Duplicate-profile screening refers a **high** similarity to another enrolled
identity. Profile-photo consistency refers a **low** similarity to the profile's
own template. An extraction failure produces no comparison decision. These
are separate tasks and should not share an interpretation simply because both
use a similarity score.

## Development and testing

The project has expanded from pairwise verification into calibrated gallery
search, multi-image enrolment, a review classifier and four component pairings.
The evaluation now retains processing failures in intended-probe denominators,
uses shared canonical outcomes and includes controlled sensitivity analyses.

```bash
python -m py_compile ACP_arden.py
python ACP_arden.py --mode self-test
python -m pytest -q
pyright
```

The self-tests and default pytest checks use synthetic inputs and test doubles
without requiring benchmark images, model files or network access. Optional
checks that need local data or weights are skipped when unavailable. Tests cover metric
accounting, calibration boundaries, role invariants, model contracts, cache
integrity, publication controls and report consistency. They test implementation
behaviour rather than certify recognition accuracy in a new population.

## Repository layout

```text
ACP_arden.py           application, experiments, reports and local review interface
README.md              setup, usage and project overview
REFERENCES.md          datasets, models, methods, hashes and licences
requirements*.txt      runtime, development and comparison dependencies
scripts/               comparison-environment installer
.vscode/               Python launch configuration
results/aggregate/     published metrics and evaluation reports
results/figures/       aggregate figures and captions
results/raw/           ignored private working files
tests/                 dataset-free test suite
```

Source and benchmark references are in [REFERENCES.md](REFERENCES.md).
Use and redistribution of this project are governed by [LICENSE](LICENSE).
