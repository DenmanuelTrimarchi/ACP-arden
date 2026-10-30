# Duplicate face screening

This repository contains my COM7014 Advanced Computing Project. I developed a
local research prototype to compare pretrained face detection and recognition
pipelines for duplicate-profile screening. My main concern was whether a high
recognition accuracy remained useful once failed image processing, false
referrals and computational cost were taken into account.

The programme searches a photograph against a gallery of enrolled identities
and identifies possible matches for human review. A referral is a similarity
result, not evidence of fraud, photograph ownership or legal identity. The
prototype does not reject registrations or apply automatic sanctions.

The implementation is in [ACP_arden.py](ACP_arden.py). The repository also
contains the tests, aggregate results and figures needed to inspect the work.
Benchmark photographs, model weights and private working files are stored
separately and are not redistributed.

## Project development

I began with one-to-one face verification using YuNet and SFace, then extended
the evaluation to gallery search. This distinction matters: a threshold that
works for a pair of photographs may produce many false referrals when one
photograph is compared with hundreds of enrolled identities.

The later experiments introduce three-image enrolment, gallery-specific
threshold calibration and a logistic-regression review classifier. I also
compared SCRFD and ArcFace with the original components, including the crossed
combinations SCRFD + SFace and YuNet + ArcFace. All four face networks remain
pretrained and frozen. Only the logistic-regression review classifier is
fitted within this project, using development identities.

The evaluation now records failed extractions and unsuccessful enrolment
alongside conditional recognition performance. It includes shared cached
outcomes, paired identity-bootstrap comparisons, threshold and gallery-size
analyses, and controlled changes to detector settings. These additions make
it possible to examine where an apparent improvement comes from, rather than
relying on a single accuracy figure.

## Installation

The saved benchmark runs used Python 3.13. The pinned dependencies require
Python 3.12 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Edit `.env` to point to the local datasets and model files:

- `FACE_DATA_ROOT`: directory containing `lfw_funneled/`.
- `FACE_PROTOCOL_ROOT`: directory containing `pairsDevTrain.txt`,
  `pairsDevTest.txt`, `pairs.txt` and `pairs_CPLFW.txt`.
- `FACE_CPLFW_RAW_ROOT`: raw CPLFW images extracted from `images.rar`.
- `FACE_MODEL_ROOT`: directory containing the YuNet and SFace weights.
- `FACE_BFW_ROOT`: BFW image tree for the gallery experiments and analyses.
- `FACE_BFW_METADATA_ROOT`: BFW metadata directory; defaults to `FACE_BFW_ROOT`.
- `FACE_ARCFACE_MODEL_ROOT`: directory containing the SCRFD and ArcFace weights
  for Experiments 8–12 and the relevant comparisons.
- `FACE_ID_HMAC_KEY`: private key used to create opaque identifiers.

Generate the identifier key with the following command, then store the result
in `.env`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The key must decode to at least 32 bytes. It is not written into published
results. Changing it changes the identifiers and invalidates an existing local
review database. The optional `FACE_CACHE_ROOT` contains biometric embeddings
and must remain in private storage. `.env` and private results are ignored by Git.

### Datasets and models

The experiments use [LFW](https://vis-www.cs.umass.edu/lfw/),
[CPLFW](https://www.whdeng.cn/CPLFW/) and
[BFW](https://github.com/visionjo/facerec-bias-bfw), obtained from their official
sources under the applicable terms. LFW provides the verification reference,
raw CPLFW tests transfer to more difficult image conditions, and BFW supplies
the repeated images and metadata for the gallery study.

The CPLFW results use the authors' raw images, not the separately aligned
`cp-aligned.zip` set. Detection and alignment therefore remain part of the
measured pipeline.

The baseline uses these OpenCV Zoo model files:

```text
face_detection_yunet_2023mar.onnx
face_recognition_sface_2021dec.onnx
```

The comparison components use InsightFace's `buffalo_l` files, `det_10g.onnx`
and `w600k_r50.onnx`. Place both directly in `FACE_ARCFACE_MODEL_ROOT` and
install the optional comparison dependencies with:

```bash
bash scripts/install_comparison_environment.sh
```

The installer preserves the pinned `opencv-python-headless` build. Installing
a second OpenCV distribution can change which implementation is imported and
affect the recorded results. Direct pins are listed in
`requirements-comparison-deps.txt`; `requirements-comparison-lock.txt` records
the verified environment's full package set.

All weight files are checked against pinned SHA-256 digests before use. Sources,
filenames, hashes and licences are recorded in [REFERENCES.md](REFERENCES.md).
Model-weight terms are separate from source-code licences. The comparison
weights were used for non-commercial academic research; the programme neither
downloads nor redistributes them.

## Running the project

```bash
python ACP_arden.py
```

This opens an interactive menu. In VS Code, selecting `.venv/bin/python` and
using **Run Python File** on `ACP_arden.py` opens the same menu. The individual
workflows are also available from the command line:

```bash
# Environment, inputs and dataset-free checks
python ACP_arden.py --mode check
python ACP_arden.py --mode verify
python ACP_arden.py --mode self-test

# Experiments 1–5: verification and the LFW gallery control
python ACP_arden.py --mode full
python ACP_arden.py --mode summary

# Experiments 6–8: BFW calibration, classifier and pipeline comparison
python ACP_arden.py --mode open-set
python ACP_arden.py --mode ml-review
python ACP_arden.py --mode pipeline-compare

# Experiments 9–12: comparison verification and crossed components
python ACP_arden.py --mode verification-compare
python ACP_arden.py --mode arcface-review
python ACP_arden.py --mode mixed-pipelines

# Controlled comparisons and saved reports
python ACP_arden.py --mode comparison-diagnostics
python ACP_arden.py --mode detector-settings
python ACP_arden.py --mode comparative-statistics
python ACP_arden.py --mode refresh-reports
python ACP_arden.py --mode experiment-table
```

`--mode full` runs Experiments 1–5. To run the complete sequence of twelve
experiments, controlled comparisons and reports, use `--mode all`. The complete
sequence requires every dataset and model, updates saved results and stops at
the first failed step.

`--results-root PATH` selects an aggregate output directory.
`python ACP_arden.py --help` lists all modes, including saved summary views.
The existing reports can be read without loading models or rerunning inference.

## Evaluation approach

For LFW, the development files generate and select a transfer threshold, which
is then frozen. Final verification uses the official ten folds: each fold's
threshold is fitted on the other nine. This avoids treating the overlapping
development and final pair files as independent datasets. The separate frozen
development threshold is applied unchanged to raw CPLFW.

For BFW, I constructed a project-defined open-set protocol rather than using
an official identification benchmark. Identities are split within each subgroup
between development and test, then between enrolled and non-mated roles. The
splits use sorted identities, subgroup-specific seeded shuffles and seed
`20260727`; they do not depend on the private identifier key. Odd counts go
to development and then the enrolled side. Identities with insufficient images
for enrolment remain non-mated.

Images are sorted by filename within each identity. The first three are
enrolment candidates and the next five are mated probes. Non-mated identities
contribute up to fifteen probes and are absent from the gallery. Three-image
enrolment requires at least two successful extractions; the one-image control
uses the first enrolment photograph.

Each evaluated partition contains 400 identities, 200 intended gallery
identities, 600 enrolment images, 1,000 mated probes and 3,000 non-mated probes.
The protocol uses 9,200 of BFW's 20,000 images. Its probe ratio is a sampling
choice, not an estimate of duplicate prevalence.

Each configured pipeline has its own development-calibrated threshold. The
false positive identification rate (FPIR) targets are 0.1%, 0.3% and 1%, with
0.3% as the primary operating point. These are experimental budgets, not
validated moderation requirements. Thresholds are frozen before evaluation;
held-out results do not determine their selection.

### Interpreting the measures

Conditional rates describe successfully scored cases. End-to-end duplicate
detection uses all intended mated probes, including extraction and enrolment
failures. FPIR is a search-level rate and differs from a pairwise false-match
rate.

True positive identification rate (TPIR) is rank-aware: the correct identity
must be within the stated rank and above threshold. A referral to the wrong
identity is recorded as `mated_wrong_identity_referred` and does not count as
correct duplicate identification.

Repeated photographs of one person are correlated. Confidence intervals
therefore resample identities, with 2,000 shared draws in the paired
comparisons and subgroup stratification. These exploratory intervals are
unadjusted for multiple comparisons and conditional on the fitted models,
frozen thresholds and fixed galleries. An interval containing zero leaves
a difference unresolved; it does not demonstrate equivalence.

## Main findings

Under the saved BFW protocol, YuNet + SFace detected 87.20% of all intended
duplicates. Its conditional TPIR@1 was 92.57%, with 0.52% FPIR, or 5.25 false
referrals per 1,000 scored non-mated searches. Mean complete-pipeline latency
was 21.72 ms per image and the weight files occupied approximately 37.1 MiB.

SCRFD + ArcFace detected 96.70% of intended duplicates. Its conditional TPIR@1
was 96.80%, with 0.13% FPIR, or 1.34 false referrals per 1,000 scored non-mated
searches. Mean complete-pipeline latency was 80.73 ms and weight-file storage
was approximately 182.4 MiB. The end-to-end difference was 9.50 percentage
points, with an exploratory paired 95% interval of 7.30–12.10.

These results compare configured pipelines. Detector settings, alignment,
preprocessing and separately calibrated thresholds remain part of each
configuration, so the comparison does not isolate a network architecture or
training-loss effect.

The logistic-regression review classifier did not reduce false referrals on
the baseline pipeline: they increased from 15 to 20, despite higher conditional
TPIR@1. The outcome differs on the ArcFace pipeline. I therefore treat the
classifier as a component to evaluate, rather than assume that adding a
learned decision layer improves every configuration.

The evaluation was repeated only to test computational reproducibility;
no repeated held-out result influenced model selection, threshold selection
or the reported policy.

The underlying measurements are in
[pipeline_comparison_metrics.json](results/aggregate/pipeline_comparison_metrics.json)
and [pretrained_pipeline_comparison.csv](results/aggregate/pretrained_pipeline_comparison.csv).
The [research report](results/aggregate/RESEARCH_REPORT.md) brings together the
verification, gallery, classifier and cost findings. Aggregate figures are in
[results/figures/](results/figures/), with an accompanying
[caption inventory](results/figures/FIGURE_CAPTIONS.md).

## Limitations and reproducibility

The results apply to the recorded benchmarks and settings. LFW and CPLFW share
identities, and none of the datasets establishes performance on a representative
service population. BFW's balanced labels permit descriptive subgroup analysis;
they do not establish deployment fairness. Each held-out subgroup contains
only 25 non-mated identities, despite its repeated probe photographs.

Lexical image selection, one identity split, small fixed galleries and
unaudited pretraining overlap limit generalisation. A development FPIR budget
may be exceeded on test. The original SCRFD + ArcFace comparison lacks
non-primary held-out operating points; reports leave those results unavailable.
Updated comparison runs record each pre-frozen operating point separately.

Detection is not bit-stable across processes on the recorded platform.
Borderline YuNet detections can change between fresh runs. OpenCL is disabled
and OpenCV uses `setNumThreads(0)`, while related experiments reuse a private
canonical outcome cache. The cache maintains consistency across those
experiments; it does not guarantee identical fresh inference on another machine.
Seeds, model hashes, dataset fingerprints, settings and software versions are
recorded with the results.

Latency is specific to the recorded environment. Mixed-pipeline timing summaries
are stage-based estimates, and storage measurements cover weights only. False
referrals are a workload proxy: reviewer time, decision quality, real duplicate
prevalence and deployment costs were not measured.

## Local review interface

```bash
python ACP_arden.py --mode review
```

The Streamlit interface runs at `127.0.0.1` and reads the ignored
`results/raw/review.sqlite` database. It displays opaque identifiers, scores,
thresholds and case status, without benchmark photographs, names or embeddings.

Duplicate-profile screening refers a high similarity to another enrolled
identity. Profile-photo consistency refers a low similarity to the profile's
own template. An extraction failure produces no comparison decision. These
tasks use the same type of score but answer different questions.

## Testing

```bash
python -m py_compile ACP_arden.py
python ACP_arden.py --mode self-test
python -m pytest -q
pyright
```

The self-tests and default pytest checks use synthetic inputs and test doubles,
without benchmark images, weights or network access. Optional checks requiring
private data or models are skipped when those files are unavailable. The tests
cover metric accounting, calibration boundaries, protocol roles, model
contracts, cache integrity and report consistency. Passing these checks verifies
implementation behaviour rather than performance in a new population.

## Repository structure

```text
ACP_arden.py           implementation, experiments and local review interface
README.md              project overview, setup and usage
REFERENCES.md          model, dataset and method references; hashes and licences
requirements*.txt      runtime, development and comparison dependencies
scripts/               comparison-environment installer
.vscode/               Python launch configuration
results/aggregate/     published metrics and evaluation reports
results/figures/       aggregate figures and captions
results/raw/           ignored private working files
tests/                 automated checks
```

References and component licences are listed in [REFERENCES.md](REFERENCES.md).
Use and redistribution of the project are governed by [LICENSE](LICENSE).
