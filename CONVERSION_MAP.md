# Conversion map

`ACP_arden.py` is organised into eighteen numbered sections. This table maps
each functional component of the research pipeline to the section that
implements it, so a reviewer can navigate the single file as quickly as a
package of small modules.

| Component | ACP_arden.py section | Status |
|---|---|---|
| Programme metadata, imports, project paths | Section 1 | Preserved |
| Model contract, pinned SHA-256 digests, detector settings, seeds, archive checksums | Section 2 | Preserved |
| CPLFW image-variant provenance fields | Section 2 | Preserved |
| `.env` parsing, storage-root resolution, path validation | Section 3 | Preserved |
| Private-path redaction for everything printed | Section 3 | Added |
| File and text hashing, evaluated-image-set fingerprint | Section 4 | Preserved |
| Model hash verification, dependency contract, software environment report | Section 4 | Preserved |
| Opaque one-way identifiers, filename scrubbing | Section 4 | Preserved |
| Bounded image loading, EXIF orientation, failure taxonomy | Section 5 | Preserved |
| YuNet detection wrapper, exactly-one-face rule, detector interface | Section 6 | Preserved |
| SFace embedding wrapper, 128-dimension contract, embedder interface | Section 7 | Preserved |
| L2 normalisation and cosine similarity | Section 8 | Preserved |
| LFW pair-protocol parsing and header validation | Section 9 | Preserved |
| CPLFW pair-protocol parsing (flat layout, two lines per pair) | Section 9 | Preserved |
| Confusion matrix, derived rates, ROC points, ROC-AUC, EER, percentile | Section 10 | Preserved |
| Threshold-candidate strategies (balanced accuracy, F1, EER, target FMR) | Section 10 | Preserved |
| Stage 1 calibration on the validation split only | Section 11 | Preserved |
| Stage 2 deterministic selection and freezing | Section 11 | Preserved |
| Stage 3 frozen-threshold enforcement for held-out evaluation | Section 11 | Preserved |
| Pair evaluation, embedding cache, per-image latency | Section 12 | Preserved |
| Four-category extraction-failure accounting and reconciliation | Section 12 | Preserved |
| Deterministic gallery manifest, role assignment, calibration exclusion | Section 13 | Preserved |
| 1:N gallery search, rank-1 and duplicate-review metrics | Section 13 | Preserved |
| Gallery manifest read/write (private, git-ignored) | Section 13 | Preserved |
| Atomic, schema-versioned JSON/CSV/Markdown artefact writing | Section 14 | Preserved |
| Run manifest, metrics summary, confusion matrices, ROC points | Section 14 | Preserved |
| Final evaluation report rendering | Section 14 | Preserved |
| Terminal results summary with mandatory limitations | Section 14 | Preserved |
| Forbidden-substring list, path-leak scanning, PNG metadata scanning | Section 15 | Preserved |
| Record-level leakage assertions | Section 15 | Preserved |
| Local review database (SQLite, opaque identifiers only) | Section 16 | Preserved |
| Streamlit human-review page and its launcher | Section 16 | Preserved |
| Deterministic synthetic self-tests and their fakes | Section 17 | Preserved |
| Environment check action | Section 18 | Preserved |
| Model and dataset verification action | Section 18 | Preserved |
| Five-experiment orchestration in required order | Section 18 | Preserved |
| Interactive menu, `--mode` command line, entry point | Section 18 | Added |

## Reference-only

These belong to a dissertation evidence workflow rather than to the
methodology, and are deliberately not part of the runnable artefact:

| Component | Status |
|---|---|
| Matplotlib evidence-figure rendering | Reference-only — not included |
| Rendered command-output screenshot pack and its indices | Reference-only — not included |
| Evidence manifest and chapter-placement index | Reference-only — not included |
| Dataset extraction and flattening helpers | Reference-only — not included |
| Comment-style audit tooling | Reference-only — not included |

Excluding them changes no reported metric: every figure in
`results/aggregate/` is produced by the pipeline in sections 11 to 15, and the
evidence tooling only re-renders those numbers for presentation.

## Notes on the two "Added" rows

- **Private-path redaction (section 3)** is new. The single-file artefact
  prints directly to a terminal rather than through separate scripts, so every
  message it emits is passed through a redactor that replaces a configured
  storage root with its variable name. No storage location can reach the
  screen, a log, or a published artefact.
- **Interactive menu (section 18)** is new. It replaces a set of separate
  command-line entry points with one launcher that requires no arguments, so
  the VS Code play button is safe to press: it shows the menu rather than
  starting a multi-minute benchmark.
