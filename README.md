# ACP-arden

A single-file, locally runnable research artefact that measures how well a
fixed, pretrained face-verification pipeline can decide whether two
unconstrained facial images show the same person, and whether the same
similarity signal can surface duplicate profiles in a 1:N gallery under a
human-review decision policy.

Everything lives in one executable Python file, [ACP_arden.py](ACP_arden.py).

## Research question

> How effectively can a pretrained face-embedding model verify whether two
> unconstrained facial images belong to the same person and identify potential
> duplicate profiles under a human-review decision policy?

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

Select an option:
```

### Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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
```

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
| `FACE_CACHE_ROOT` | Optional embedding cache; leave unset by default |

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

Gallery size: 986
Duplicate detection rate: 96.58%
Rank-1 identification rate: 92.76%
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
- The **gallery duplicate detection rate is inseparable from the 52.56% false
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
requirements.txt    pinned dependencies
.env.example        template for local storage paths
.vscode/            run configuration for the play button
results/aggregate/  published, privacy-scanned results
results/raw/        local, git-ignored working files
tests/              contract tests, no dataset or model required
```

## Limitations

LFW and CPLFW carry their own demographic skew, which limits how
representative any figure here is of a real user population.
Face-extraction failures are excluded from the accuracy metrics but reported
as their own rate, never silently dropped. The gallery experiment is
research-scale rather than production-scale. And "duplicate profile" here
means "the same face was detected in the gallery" — not a legal or
investigative finding about any person.

## Attribution and licence

All code in this repository is original. The OpenCV, NumPy, Pillow and
Streamlit calls use those libraries' documented public APIs, which is use
rather than adaptation of their source. The two pretrained ONNX files are
external artefacts, not code: they are published in the OpenCV Zoo repository
(`github.com/opencv/opencv_zoo`) under the MIT licence (YuNet) and the
Apache-2.0 licence (SFace), and are pinned here by digest. LFW and CPLFW are
external datasets used under their authors' published terms; neither is
redistributed here.

Made available for academic assessment and review — see [LICENSE](LICENSE) for
the terms, which reserve all other rights.
