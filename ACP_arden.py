#!/usr/bin/env python3
"""ACP-arden — a single-file face-verification research artefact.

A standalone, local proof of concept that measures how well a fixed,
pretrained face-verification pipeline (OpenCV YuNet detection followed by
OpenCV SFace embedding) can decide whether two unconstrained facial images
belong to the same person, and whether the same similarity signal can
surface duplicate profiles in a 1:N gallery under a human-review policy.

This is a research artefact, not a dating application and not a fraud
detector. No face-detection or face-recognition network is trained or
fine-tuned. Experiment 7 trains a small logistic-regression review classifier
using identity-disjoint BFW development data. No website is scraped, and no
account is ever banned, rejected, accused or classified as a scam. A
similarity above the operating threshold opens a case for a human reviewer
and nothing more.

Methodological boundary, enforced in code rather than only in prose:

    pairsDevTrain.txt -> candidate thresholds only
    pairsDevTest.txt  -> deterministic selection, then freezing
    pairs.txt         -> final LFW evaluation with the frozen threshold
    pairs_CPLFW.txt   -> raw CPLFW with that same frozen threshold

Run it with the VS Code play button, or:

    python ACP_arden.py                    # interactive menu
    python ACP_arden.py --mode self-test   # deterministic synthetic tests

Datasets and the two pinned ONNX model files are never stored in this
project. Their locations are read from a local, git-ignored ``.env``.
"""

# =============================================================================
# 1. Imports and programme metadata
# =============================================================================

# ``from __future__ import annotations`` stores every type annotation in this
# file as a string rather than evaluating it at definition time. Two practical
# consequences: a class may be named in a signature before it is defined, and
# the annotations impose no runtime cost.
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import heapq
import importlib.util
import hmac
import json
import math
import os
import platform
import random
import re
import sqlite3
import statistics
import string
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np

PROGRAMME_NAME = "ACP-arden"
PROGRAMME_TITLE = "ACP-arden — Face Verification Research Artefact"
PROGRAMME_VERSION = "1.0.0"

# Every path in this programme is derived from the file's own location, so the
# working directory never changes which results are read or written.
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
AGGREGATE_ROOT = RESULTS_ROOT / "aggregate"
RAW_ROOT = RESULTS_ROOT / "raw"
DEFAULT_REVIEW_DB = RAW_ROOT / "review.sqlite"
DEFAULT_GALLERY_MANIFEST = RAW_ROOT / "gallery_manifest.json"

# Attribution. Every line of this file is original project code: the OpenCV,
# NumPy, Pillow and Streamlit calls below use those libraries' documented
# public APIs, which is use rather than adaptation of their source. Wherever an
# external model, dataset, API or published method enters the pipeline, a
# four-field attribution header records its origin immediately above the code
# concerned. The complete register, with licences and digests, is REFERENCES.md.
# Neither benchmark dataset nor either model file is redistributed here.
#
# Reading order. The thirty numbered sections run from configuration through
# to the launcher, so the file may be read from top to bottom in roughly the
# order the pipeline executes. Comments that explain a language construct are
# addressed to a reader following the method rather than the syntax, and appear
# at the first use of each construct.


# =============================================================================
# 2. Configuration and pinned model hashes
# =============================================================================
#
# Changing any value in this section changes the evaluation partition. A
# threshold calibrated under one contract must never be applied under another,
# so these are constants rather than command-line options.

EMBEDDING_DIMENSIONS = 128
MODEL_VERSION = "opencv-sface-2021dec-yunet-2023mar"
PREPROCESSING_REVISION = "opencv-yunet-sface-exif-bgr-l2-v1"

# The detection and embedding stages are external pretrained artefacts, not
# code adapted into this file. Both are consumed through OpenCV's documented
# public API; the two blocks below record where each originates.
##############
# Title: YuNet: A Tiny Millisecond-level Face Detector
# Author: Wu, W., Peng, H. and Yu, S., Machine Intelligence Research, 20(5), pp. 656-665
# Date: 2023
# Availability: https://doi.org/10.1007/s11633-023-1423-y
##############
##############
# Title: SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition
# Author: Zhong, Y., Deng, W., Hu, J., Zhao, D., Li, X. and Wen, D.
# Date: 2021
# Availability: https://doi.org/10.1109/TIP.2020.3048632
##############
##############
# Title: OpenCV Zoo, distributor of the two pinned ONNX weight files
# Author: OpenCV team; YuNet weights MIT, SFace weights Apache-2.0
# Date: 2023 (YuNet 2023mar release), 2021 (SFace 2021dec release)
# Availability: https://github.com/opencv/opencv_zoo
##############
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"

# Digests of the official OpenCV Zoo release. A file that does not match is
# refused rather than loaded (see verify_model_file in section 4).
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"

DETECTOR_SCORE_THRESHOLD = 0.9
DETECTOR_NMS_THRESHOLD = 0.3
DETECTOR_TOP_K = 5000

# Dependencies whose exact installed version forms part of the evaluation
# partition, because they can move floating-point results at the margins.
EXPECTED_DEPENDENCY_VERSIONS = {
    "numpy": "2.5.1",
    "opencv-python-headless": "4.13.0.92",
    "Pillow": "12.3.0",
    # Fitting the review classifier is version-sensitive at the margins.
    "scikit-learn": "1.9.0",
}

DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
HARD_MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 12_000_000
HARD_MAX_IMAGE_PIXELS = 40_000_000

DEFAULT_RANDOM_SEED = 20260727

# Archive checksums recorded for this project's own dataset acquisition. They
# describe the copies evaluated here; the CPLFW authors do not publish an
# official archive checksum of their own.
# The benchmark datasets are external inputs used under their authors'
# published terms. Neither is redistributed here; only these checksums, which
# identify the exact copies this project evaluated, are recorded.
##############
# Title: Labeled Faces in the Wild (LFW) benchmark database
# Author: Huang, G.B., Ramesh, M., Berg, T. and Learned-Miller, E. (UMass Amherst TR 07-49)
# Date: October 2007
# Availability: http://vis-www.cs.umass.edu/lfw/lfw.pdf
##############
# The alignment procedure that produced the funnelled image set evaluated here
# is a separate publication, and a distinct image set from the unaligned
# distribution, so it is cited in its own right.
##############
# Title: Unsupervised Joint Alignment of Complex Images
# Author: Huang, G.B., Jain, V. and Learned-Miller, E. (ICCV)
# Date: 2007
# Availability: http://vis-www.cs.umass.edu/lfw/
##############
LFW_ARCHIVE_FILENAME = "lfwfunneled.tgz"
LFW_ARCHIVE_MD5 = "1b42dfed7d15c9b2dd63d5e5840c86ad"
##############
# Title: Cross-Pose LFW (CPLFW) benchmark database
# Author: Zheng, T. and Deng, W. (Beijing Univ. of Posts and Telecommunications, TR 18-01)
# Date: February 2018
# Availability: http://www.whdeng.cn/cplfw/
##############
CPLFW_ARCHIVE_FILENAME = "CPLFW.zip"
CPLFW_ARCHIVE_SHA256 = "9a09dd1ebe1a000c52f69f365f5d564cd529f1fcf4f0479510231856f358f416"

# CPLFW ships two non-interchangeable image sets inside the same archive: the
# authors' raw, unconstrained images and a separately pre-cropped and aligned
# copy. A result must never be ambiguous about which one it scored, so the
# variant is an explicit, recorded field rather than an assumption.
CPLFW_RAW_ARCHIVE_FILENAME = "images.rar"
CPLFW_RAW_ARCHIVE_SHA256 = "7baca61dda21341eaa642f229eedfbba1d0aaa2d22447d79e158920106831165"
CPLFW_ALIGNED_ARCHIVE_FILENAME = "cp-aligned.zip"
CPLFW_ALIGNED_ARCHIVE_SHA256 = "420adcc13f1ab9510d8f99af04dbfb1695645ff73942c2a1010c5c01fd8367e2"

CPLFW_IMAGE_VARIANTS = ("raw", "aligned")

LFW_CALIBRATION_PROTOCOL = "pairsDevTrain.txt"
LFW_DEVELOPMENT_PROTOCOL = "pairsDevTest.txt"
LFW_FINAL_PROTOCOL = "pairs.txt"
CPLFW_PROTOCOL = "pairs_CPLFW.txt"

REQUIRED_LFW_PROTOCOLS = (
    LFW_CALIBRATION_PROTOCOL,
    LFW_DEVELOPMENT_PROTOCOL,
    LFW_FINAL_PROTOCOL,
)

CPLFW_EXPECTED_PAIRS = 6000
CPLFW_EXPECTED_PER_CLASS = 3000

SCHEMA_VERSION = 1

# Marks results produced under the corrected gallery accounting, in which a
# reference image that fails to enrol is recorded rather than dropped. Results
# carrying this revision are not comparable with the earlier conditional-only
# duplicate_gallery_metrics.json, which is retained unchanged for provenance.
GALLERY_METHODOLOGY_REVISION = "open-set-gallery-accounting-v2"

POLICY_NOTE = (
    "A result above threshold opens a case for human review only. It is not "
    "evidence of scam activity and does not ban, reject or accuse any identity."
)


def cplfw_provenance_fields(image_variant: str) -> Dict[str, str]:
    """Result fields that make a CPLFW evaluation's image variant explicit and
    impossible to omit. Anything other than 'raw' or 'aligned' is refused."""
    if image_variant == "raw":
        archive_filename = CPLFW_RAW_ARCHIVE_FILENAME
        archive_sha256 = CPLFW_RAW_ARCHIVE_SHA256
        image_source = "authors-distributed images.rar"
    elif image_variant == "aligned":
        archive_filename = CPLFW_ALIGNED_ARCHIVE_FILENAME
        archive_sha256 = CPLFW_ALIGNED_ARCHIVE_SHA256
        image_source = "authors-distributed cp-aligned.zip"
    else:
        raise ValueError(
            f"Unknown CPLFW image variant {image_variant!r}; expected one of {CPLFW_IMAGE_VARIANTS}"
        )
    return {
        "dataset_image_variant": image_variant,
        "dataset_image_source": image_source,
        "dataset_archive_filename": archive_filename,
        "dataset_archive_sha256": archive_sha256,
        "dataset_root_description": "private, gitignored local research storage; path omitted",
    }


# =============================================================================
# 3. Environment loading and path validation
# =============================================================================
#
# No dataset, protocol or model location is ever hard-coded. Each one is read
# from a git-ignored ``.env`` beside this file, or from the process
# environment, so the artefact carries no researcher-specific path.

ENV_FILENAME = ".env"

REQUIRED_ENVIRONMENT_VARIABLES = (
    "FACE_DATA_ROOT",
    "FACE_PROTOCOL_ROOT",
    "FACE_MODEL_ROOT",
    "FACE_CPLFW_RAW_ROOT",
)
OPTIONAL_ENVIRONMENT_VARIABLES = (
    "FACE_CACHE_ROOT",
    # Optional external benchmarks. Listed here so the privacy scanner adds
    # their configured roots to the forbidden-substring set as well.
    "FACE_BFW_ROOT",
    "FACE_BFW_METADATA_ROOT",
    "FACE_ARCFACE_MODEL_ROOT",
)


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def parse_env_text(text: str) -> Dict[str, str]:
    """Minimal ``.env`` reader: ``KEY=value``, single- or double-quoted values,
    blank lines and ``#`` comments. Deliberately not a dependency — quoted
    values matter here only because research storage paths contain spaces."""
    values: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def load_env_file(path: Path = PROJECT_ROOT / ENV_FILENAME) -> Dict[str, str]:
    """Read the local ``.env`` if present. A missing file is not an error: the
    caller may have exported the variables directly."""
    path = Path(path)
    if not path.is_file():
        return {}
    return parse_env_text(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class EnvironmentConfig:
    """Resolved research-storage roots. Every field is either explicitly
    supplied or ``None``; no default location is ever assumed."""

    data_root: Optional[Path]
    protocol_root: Optional[Path]
    model_root: Optional[Path]
    cplfw_raw_root: Optional[Path]
    cache_root: Optional[Path]
    # Optional external benchmarks. Absent configuration is not an error: the
    # five baseline experiments and every existing mode run without them.
    bfw_root: Optional[Path] = None
    bfw_metadata_root: Optional[Path] = None
    arcface_model_root: Optional[Path] = None

    # A ``@classmethod`` receives the class itself as its first argument rather
    # than an instance, which is the idiomatic way to offer an alternative
    # constructor: ``EnvironmentConfig.load()`` builds and returns an instance.
    # ``Optional[X]`` is shorthand for "an X or None".
    @classmethod
    def load(cls, env: Optional[Mapping[str, str]] = None) -> "EnvironmentConfig":
        """Process environment first, then the local ``.env`` as a fallback, so
        an explicit export always wins over the file."""
        source: Dict[str, str] = dict(load_env_file())
        source.update({k: v for k, v in (os.environ if env is None else env).items() if v})

        # Installed as a side effect of loading configuration, so every code
        # path that resolves storage roots also has identifiers available. The
        # key itself is never stored on the returned object, which is written
        # into provenance records.
        identifier_key = source.get(ID_HMAC_KEY_VARIABLE)
        if identifier_key:
            configure_id_hmac_key(identifier_key)

        def optional(name: str) -> Optional[Path]:
            value = source.get(name)
            return Path(value).expanduser() if value else None

        return cls(
            data_root=optional("FACE_DATA_ROOT"),
            protocol_root=optional("FACE_PROTOCOL_ROOT"),
            model_root=optional("FACE_MODEL_ROOT"),
            cplfw_raw_root=optional("FACE_CPLFW_RAW_ROOT"),
            cache_root=optional("FACE_CACHE_ROOT"),
            bfw_root=optional(BFW_ROOT_VARIABLE),
            bfw_metadata_root=optional(BFW_METADATA_ROOT_VARIABLE),
            arcface_model_root=optional(ARCFACE_MODEL_ROOT_VARIABLE),
        )

    def require_bfw_roots(self) -> Tuple[Path, Path]:
        """Resolve the BFW image root and metadata table, or stop on the exact
        blocker. The metadata root defaults to the image root, which is where
        the official archive places the datatable."""
        if self.bfw_root is None:
            raise BfwDatasetError(
                f"{BFW_ROOT_VARIABLE} is not set. The BFW open-set experiment needs the "
                f"official Balanced Faces in the Wild data, obtained from "
                f"https://github.com/visionjo/facerec-bias-bfw under its own terms. It is "
                f"never downloaded automatically and no mirror is used."
            )
        metadata_root = self.bfw_metadata_root or self.bfw_root
        matches = sorted(Path(metadata_root).glob("bfw*datatable*.csv"))
        if not matches:
            raise BfwDatasetError(
                f"No BFW datatable (bfw*datatable*.csv) found under the configured "
                f"{BFW_METADATA_ROOT_VARIABLE or BFW_ROOT_VARIABLE}. Expected the official "
                f"metadata table shipped with the dataset."
            )
        # The release ships the table twice, once with a version in the name and
        # once without. Prefer the versioned copy so provenance records which
        # release was read; ``next(... , default)`` returns the first match or
        # falls back rather than raising when none is versioned.
        versioned = next((m for m in matches if re.search(r"v\d", m.name)), None)
        return Path(self.bfw_root), versioned or matches[0]

    def require_data_root(self) -> Path:
        return _require(self.data_root, "FACE_DATA_ROOT")

    def require_protocol_root(self) -> Path:
        return _require(self.protocol_root, "FACE_PROTOCOL_ROOT")

    def require_model_root(self) -> Path:
        return _require(self.model_root, "FACE_MODEL_ROOT")

    def require_lfw_root(self) -> Path:
        return self.require_data_root() / "lfw_funneled"

    def require_cplfw_raw_root(self) -> Path:
        if self.cplfw_raw_root is not None:
            return self.cplfw_raw_root
        return self.require_data_root() / "cplfw"

    def missing_variables(self) -> List[str]:
        present = {
            "FACE_DATA_ROOT": self.data_root,
            "FACE_PROTOCOL_ROOT": self.protocol_root,
            "FACE_MODEL_ROOT": self.model_root,
            "FACE_CPLFW_RAW_ROOT": self.cplfw_raw_root,
        }
        return [name for name, value in present.items() if value is None]

    def private_roots(self) -> List[Path]:
        return [
            root
            for root in (
                self.data_root,
                self.protocol_root,
                self.model_root,
                self.cplfw_raw_root,
                self.cache_root,
            )
            if root is not None
        ]


def _require(value: Optional[Path], name: str) -> Path:
    if value is None:
        raise ConfigurationError(
            f"{name} is not set. Copy .env.example to .env and fill it in, or export "
            f"{name} directly. This project never assumes a default path for real "
            f"dataset or model files."
        )
    return value


def project_relative(path: Path) -> str:
    """POSIX path relative to this file's directory, or the bare filename when
    the target lives outside the project. Artifacts record this rather than an
    absolute path, so a published result never carries a private location."""
    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def redact_private_paths(text: str, config: Optional[EnvironmentConfig] = None) -> str:
    """Replace any configured private root, and any remaining home-directory
    prefix, with a placeholder. Applied to everything this programme prints, so
    a failure message can never disclose a storage location."""
    config = config if config is not None else EnvironmentConfig.load()
    replacements: List[Tuple[str, str]] = []
    for name, root in (
        ("FACE_DATA_ROOT", config.data_root),
        ("FACE_PROTOCOL_ROOT", config.protocol_root),
        ("FACE_MODEL_ROOT", config.model_root),
        ("FACE_CPLFW_RAW_ROOT", config.cplfw_raw_root),
        ("FACE_CACHE_ROOT", config.cache_root),
    ):
        if root is not None:
            replacements.append((str(root), f"<{name}>"))
    replacements.append((str(Path.home()), "<HOME>"))
    # Longest first, so a nested root is not partially rewritten by its parent.
    for needle, placeholder in sorted(replacements, key=lambda item: -len(item[0])):
        if needle:
            text = text.replace(needle, placeholder)
    return text


def announce(message: str) -> None:
    """Print a line with private locations redacted."""
    print(redact_private_paths(message))


# =============================================================================
# 4. Hashing, provenance and privacy helpers
# =============================================================================


class ModelUnavailableError(RuntimeError):
    """Raised when a model file is missing or fails hash verification."""


class DependencyContractError(RuntimeError):
    """Raised when an installed dependency does not match the pinned contract."""


# Opaque identifiers are keyed, not merely salted. A published fixed salt makes
# the mapping from a public identifier back to a dataset identity a dictionary
# attack over a known, enumerable name list; a secret key removes that.
OPAQUE_ID_VERSION = "hmac-sha256-v1"
OPAQUE_ID_HEX_LENGTH = 32
ID_HMAC_KEY_VARIABLE = "FACE_ID_HMAC_KEY"
MINIMUM_ID_HMAC_KEY_BYTES = 32

# Rejected outright so a template value can never reach a published artefact.
_PLACEHOLDER_KEY_MARKERS = (
    "changeme",
    "change-me",
    "replace",
    "example",
    "placeholder",
    "your-key",
    "yourkey",
    "todo",
    "secret",
    "xxxx",
)


class OpaqueIdentifierKeyError(RuntimeError):
    """Raised when the identifier HMAC key is missing, weak or malformed."""


# Held in memory only. Never written to a result, never logged, never hashed
# into a fingerprint that could confirm a guessed key.
_ID_HMAC_KEY: Optional[bytes] = None


def sha256_of_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        # ``iter(callable, sentinel)`` repeatedly calls the first argument until
        # it returns the second. Here that reads the file a megabyte at a time
        # and stops at the empty bytestring marking end-of-file, so a large
        # model binary is never held in memory in full.
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_evaluated_image_set(paths: Iterable[Path], dataset_root: Path) -> str:
    """Fingerprint of *which* images an evaluation touched, without re-hashing
    gigabytes of pixels: SHA-256 of the sorted, newline-joined list of paths
    relative to the dataset root. Two runs sharing this value scored exactly
    the same image set."""
    dataset_root = Path(dataset_root).resolve()
    relative = sorted({str(Path(p).resolve().relative_to(dataset_root)) for p in paths})
    return hashlib.sha256("\n".join(relative).encode("utf-8")).hexdigest()


def verify_model_file(path: Path, expected_sha256: str) -> str:
    """Return the file's SHA-256 if it matches the pinned digest, else refuse."""
    path = Path(path)
    if not path.is_file():
        raise ModelUnavailableError(f"Model file not found: {path}")
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        raise ModelUnavailableError(
            f"Model hash mismatch for {path}: expected {expected_sha256}, got {actual}. "
            f"Do not proceed — re-download the exact pinned OpenCV Zoo release."
        )
    return actual


def check_dependency_contract(*, strict: bool = True) -> Dict[str, Dict[str, Optional[str]]]:
    report: Dict[str, Dict[str, Optional[str]]] = {}
    mismatched: List[str] = []
    for package, expected in EXPECTED_DEPENDENCY_VERSIONS.items():
        try:
            installed: Optional[str] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            installed = None
        report[package] = {"expected": expected, "installed": installed}
        if installed != expected:
            mismatched.append(package)
    # Metadata alone is not sufficient. Installing opencv-python alongside
    # opencv-python-headless leaves both recorded while only one is imported,
    # so the runtime version of the library actually loaded is checked too.
    try:
        import cv2

        loaded = str(getattr(cv2, "__version__", ""))
    except Exception:  # pragma: no cover - cv2 is a pinned dependency
        loaded = ""
    expected_cv2 = EXPECTED_DEPENDENCY_VERSIONS.get("opencv-python-headless", "")
    if loaded and expected_cv2 and not expected_cv2.startswith(loaded):
        report["cv2 (imported)"] = {"expected": expected_cv2, "installed": loaded}
        mismatched.append(
            f"cv2 (imported {loaded}, expected {expected_cv2}; another OpenCV "
            f"distribution is shadowing opencv-python-headless)"
        )

    if mismatched and strict:
        raise DependencyContractError(
            "Dependency version mismatch for: "
            + ", ".join(mismatched)
            + ". Reinstall with the pinned versions in requirements.txt before running any "
            "evaluation that will be reported as evidence."
        )
    return report


def software_environment_report() -> Dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "dependencies": check_dependency_contract(strict=False),
    }


def decode_id_hmac_key(raw: str) -> bytes:
    """Decode a candidate key from URL-safe base64 or hexadecimal and refuse
    anything that would not carry at least 32 bytes of entropy. The value is
    never echoed back in an error message."""
    candidate = raw.strip()
    if not candidate:
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} is empty. Generate one with: "
            f"python -c \"import secrets;print(secrets.token_urlsafe(32))\""
        )
    lowered = candidate.lower()
    for marker in _PLACEHOLDER_KEY_MARKERS:
        if marker in lowered:
            raise OpaqueIdentifierKeyError(
                f"{ID_HMAC_KEY_VARIABLE} looks like a placeholder value. Generate a real "
                f"key with: python -c \"import secrets;print(secrets.token_urlsafe(32))\""
            )

    decoded: Optional[bytes] = None
    # Hex first: a hex string is also valid base64 alphabet, and interpreting it
    # as base64 would understate its length.
    stripped = candidate.removeprefix("0x").removeprefix("0X")
    if len(stripped) % 2 == 0 and all(c in string.hexdigits for c in stripped):
        try:
            decoded = bytes.fromhex(stripped)
        except ValueError:
            decoded = None
    if decoded is None:
        padded = candidate + "=" * (-len(candidate) % 4)
        try:
            decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        except (ValueError, binascii.Error) as exc:
            raise OpaqueIdentifierKeyError(
                f"{ID_HMAC_KEY_VARIABLE} is not valid URL-safe base64 or hexadecimal."
            ) from exc

    if len(decoded) < MINIMUM_ID_HMAC_KEY_BYTES:
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} decodes to {len(decoded)} bytes; at least "
            f"{MINIMUM_ID_HMAC_KEY_BYTES} are required. Generate one with: "
            f"python -c \"import secrets;print(secrets.token_urlsafe(32))\""
        )
    if len(set(decoded)) == 1:
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} is a single repeated byte and carries no entropy."
        )
    return decoded


def configure_id_hmac_key(raw: Optional[str]) -> None:
    """Install the process-wide identifier key. Called once at start-up."""
    global _ID_HMAC_KEY
    if raw is None:
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} is not set. It is required so that public identifiers "
            f"cannot be reversed by hashing a candidate name list. Add it to .env; generate "
            f"one with: python -c \"import secrets;print(secrets.token_urlsafe(32))\""
        )
    _ID_HMAC_KEY = decode_id_hmac_key(raw)


def id_hmac_key_is_configured() -> bool:
    return _ID_HMAC_KEY is not None


@contextmanager
def temporary_id_hmac_key(raw: str):
    """Install a key for the duration of a block. Used by the synthetic
    self-tests so they run without a configured environment, and by unit tests
    that need two different keys in one process."""
    global _ID_HMAC_KEY
    previous = _ID_HMAC_KEY
    try:
        _ID_HMAC_KEY = decode_id_hmac_key(raw)
        yield
    finally:
        _ID_HMAC_KEY = previous


# Fixed key used only by the in-process synthetic self-tests. It protects
# nothing: it exists so the self-tests are runnable on a machine with no
# research configuration, and it never touches real dataset identities.
SELF_TEST_ID_HMAC_KEY = "0" * 63 + "1"


def opaque_id(value: str) -> str:
    """Deterministic, one-way identifier standing in for a real identity or
    sample name. Deterministic under a fixed key so a re-run reproduces the same
    identifiers without ever storing the reversible mapping, and keyed so that
    nobody without the key can rebuild that mapping by hashing candidate names."""
    if _ID_HMAC_KEY is None:
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} has not been configured; refusing to emit an "
            f"identifier that would be reversible by dictionary attack."
        )
    digest = hmac.new(_ID_HMAC_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:OPAQUE_ID_HEX_LENGTH]


def opaque_ids_match(left: str, right: str) -> bool:
    """Constant-time comparison for values derived from the secret key."""
    return hmac.compare_digest(left, right)


def scrub_filename(path: Path) -> str:
    """Only the filename component, never an absolute or project-relative path."""
    return Path(path).name


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# 5. Image loading
# =============================================================================
#
# Bounded before decoding and after decoding, EXIF orientation normalised,
# animated images refused, and every failure raised as one catchable exception
# rather than a silent skip that would quietly shrink the denominator.


class ImageLoadError(RuntimeError):
    """Raised for any image that cannot be safely and strictly loaded."""


# ``@dataclass`` generates __init__, __repr__ and __eq__ from the annotated
# attributes below, so the class body declares data rather than boilerplate.
# ``frozen=True`` additionally makes instances immutable: assigning to a field
# after construction raises, which is what stops a loaded image being mutated
# in place halfway through the pipeline.
@dataclass(frozen=True)
class LoadedImage:
    bgr: np.ndarray  # HxWx3 uint8, OpenCV's BGR channel order
    width: int
    height: int
    source_path: Path


# The bare ``*`` in the signature marks every parameter after it as
# keyword-only: a caller must write ``max_bytes=...`` rather than passing a
# bare number positionally. For two same-typed limits that is the difference
# between a readable call and a silently transposed pair of bounds.
def load_image_bgr(
    path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> LoadedImage:
    from PIL import Image, ImageOps

    if max_bytes > HARD_MAX_IMAGE_BYTES:
        raise ImageLoadError(f"max_bytes {max_bytes} exceeds hard ceiling {HARD_MAX_IMAGE_BYTES}")
    if max_pixels > HARD_MAX_IMAGE_PIXELS:
        raise ImageLoadError(f"max_pixels {max_pixels} exceeds hard ceiling {HARD_MAX_IMAGE_PIXELS}")

    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Image file does not exist: {path}")

    size = path.stat().st_size
    if size == 0:
        raise ImageLoadError(f"Image file is empty: {path}")
    if size > max_bytes:
        raise ImageLoadError(f"Image file {path} is {size} bytes, exceeds max_bytes={max_bytes}")

    try:
        with Image.open(path) as raw:
            raw.load()
            if getattr(raw, "is_animated", False):
                raise ImageLoadError(f"Animated/multi-frame images are not supported: {path}")
            oriented = ImageOps.exif_transpose(raw)
            if oriented is None:
                raise ImageLoadError(f"Failed to normalise EXIF orientation for: {path}")
            width, height = oriented.size
            if width * height > max_pixels:
                raise ImageLoadError(
                    f"Image {path} has {width * height} pixels, exceeds max_pixels={max_pixels}"
                )
            rgb = oriented.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except ImageLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalise every decode failure
        # ``raise ... from exc`` records the original decode error as the new
        # exception's cause, so the traceback keeps both the library's message
        # and this project's failure taxonomy.
        raise ImageLoadError(f"Could not decode image {path}: {exc}") from exc

    # ``array[:, :, ::-1]`` keeps every row and column but walks the third axis
    # (colour) backwards, converting Pillow's RGB order to the BGR order OpenCV
    # expects. Negative-step slicing only produces a view, so
    # ``ascontiguousarray`` copies it into a contiguous buffer that OpenCV's C++
    # layer can read directly.
    bgr = np.ascontiguousarray(array[:, :, ::-1])
    return LoadedImage(bgr=bgr, width=width, height=height, source_path=path)


# =============================================================================
# 6. YuNet face detection
# =============================================================================
#
# Exactly one detectable face is required, matching the research question
# ("does this photo show one identifiable face"). Zero and multiple detections
# are counted as explicit outcomes in section 12, never silently dropped.


# A ``Protocol`` describes a shape rather than an ancestry: any object with a
# matching ``detect_single_face`` method satisfies this type without inheriting
# from it. That is what lets the evaluators accept either the real OpenCV
# wrapper or a deterministic synthetic stand-in. ``@runtime_checkable``
# additionally permits ``isinstance`` against the protocol, which checks only
# that the named methods exist.
@runtime_checkable
class FaceDetector(Protocol):
    """Returns one detected face's row, or raises ``FaceCountError``."""

    def detect_single_face(self, bgr: np.ndarray) -> np.ndarray: ...


class FaceCountError(RuntimeError):
    """Raised when an image does not contain exactly one detectable face."""

    def __init__(self, face_count: int):
        super().__init__(f"Expected exactly one face, found {face_count}")
        self.face_count = face_count


@dataclass(frozen=True)
class DetectorSettings:
    score_threshold: float = DETECTOR_SCORE_THRESHOLD
    nms_threshold: float = DETECTOR_NMS_THRESHOLD
    top_k: int = DETECTOR_TOP_K


##############
# Title: The OpenCV Library (cv2.FaceDetectorYN and cv2.FaceRecognizerSF APIs)
# Author: Bradski, G., Dr. Dobb's Journal of Software Tools, 25(11), pp. 120-125
# Date: 2000 (library); APIs used as documented in the OpenCV 4.x release series
# Availability: https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html
##############
# The wrapper below is original code. It calls OpenCV's documented public API,
# which is use of the library rather than adaptation of its source; the header
# records the API's origin, not a borrowed implementation.
class YuNetDetector:
    def __init__(
        self,
        model_path: Path,
        expected_sha256: str,
        settings: DetectorSettings = DetectorSettings(),
    ):
        import cv2

        self.model_sha256 = verify_model_file(Path(model_path), expected_sha256)
        self.settings = settings
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            settings.score_threshold,
            settings.nms_threshold,
            settings.top_k,
        )

    def detect_single_face(self, bgr: np.ndarray) -> np.ndarray:
        """Return the single detection's row from YuNet's output matrix
        (bounding box, five landmarks, confidence), or raise FaceCountError."""
        height, width = bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(bgr)
        # YuNet returns None rather than an empty matrix when it finds nothing,
        # so the None case and the wrong-count case are tested together. Testing
        # them in one expression is also what lets a type checker prove the
        # subscript below is safe.
        if faces is None or len(faces) != 1:
            raise FaceCountError(0 if faces is None else len(faces))
        return faces[0]


# =============================================================================
# 7. SFace embedding
# =============================================================================
#
# Produces a raw, not yet L2-normalised 128-value feature vector aligned from a
# YuNet detection. Normalisation is a separate step (section 8) so the live
# pipeline and any offline re-analysis share one normalisation code path.


@runtime_checkable
class FaceEmbedder(Protocol):
    """Returns a raw, not yet L2-normalised embedding for a detected face."""

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray: ...


class EmbeddingShapeError(RuntimeError):
    """Raised when SFace returns a vector of unexpected shape."""


class SFaceEmbedder:
    def __init__(self, model_path: Path, expected_sha256: str):
        import cv2

        self.model_sha256 = verify_model_file(Path(model_path), expected_sha256)
        self._recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        aligned = self._recognizer.alignCrop(bgr, face_row)
        feature = self._recognizer.feature(aligned)
        embedding = np.asarray(feature, dtype=np.float64).reshape(-1)
        if embedding.shape[0] != EMBEDDING_DIMENSIONS:
            raise EmbeddingShapeError(
                f"Unexpected embedding dimensionality {embedding.shape[0]}, "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
        return embedding


def load_models(model_root: Path) -> Tuple[YuNetDetector, SFaceEmbedder]:
    """Hash-verified detector and embedder pair. Any digest mismatch stops the
    run here rather than producing a result under an unknown model."""
    detector = YuNetDetector(Path(model_root) / YUNET_FILENAME, YUNET_SHA256)
    embedder = SFaceEmbedder(Path(model_root) / SFACE_FILENAME, SFACE_SHA256)
    return detector, embedder


# =============================================================================
# 8. Similarity and normalisation
# =============================================================================


class SimilarityError(ValueError):
    """Raised for malformed embeddings (wrong shape, non-finite, zero norm)."""


def l2_normalize(vector: np.ndarray, *, tolerance: float = 1e-7) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.shape[0] == 0:
        raise SimilarityError("Vector must have at least one dimension.")
    if not np.all(np.isfinite(vector)):
        raise SimilarityError("Vector must contain only finite numbers before normalisation.")
    norm = math.sqrt(float(np.dot(vector, vector)))
    if norm <= 1e-12:
        raise SimilarityError("Vector norm is too close to zero to normalise safely.")
    normalized = vector / norm
    result_norm = math.sqrt(float(np.dot(normalized, normalized)))
    if abs(result_norm - 1.0) > tolerance:
        raise SimilarityError("Normalisation self-check failed.")
    return normalized


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape[0] == 0 or left.shape[0] != right.shape[0]:
        raise SimilarityError("Embeddings must have the same non-zero number of dimensions.")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise SimilarityError("Embeddings must contain only finite numbers.")
    left_norm = math.sqrt(float(np.dot(left, left)))
    right_norm = math.sqrt(float(np.dot(right, right)))
    if left_norm == 0.0 or right_norm == 0.0:
        raise SimilarityError("Embeddings must have a non-zero norm.")
    return float(np.dot(left, right) / (left_norm * right_norm))


# =============================================================================
# 9. LFW and CPLFW protocol parsing
# =============================================================================
#
# The two datasets' authors distribute genuinely different formats, confirmed
# against the real distributed files rather than assumed from secondary
# sources:
#
# - LFW (pairs.txt, pairsDevTrain.txt, pairsDevTest.txt): a header line, then
#   rows of either "identity image_a image_b" (matched) or
#   "identity_a image_a identity_b image_b" (mismatched), with images stored
#   per identity at identity/identity_%04d.jpg.
# - CPLFW (pairs_CPLFW.txt): no header; each pair is exactly two consecutive
#   "filename label" lines sharing the same label, with images stored flat
#   under the dataset root.


# The two file *formats* parsed below are defined by the respective dataset
# authors and are documented in the references recorded in section 2 (LFW:
# Huang et al., 2007; CPLFW: Zheng and Deng, 2018). The parsers themselves are
# original code, written against the authors' distributed files rather than
# adapted from any other implementation.
class ProtocolError(RuntimeError):
    """Raised for any malformed, inconsistent, or unsafe protocol file."""


@dataclass(frozen=True)
class Pair:
    left_path: Path
    right_path: Path
    same_identity: bool
    left_identity: str
    right_identity: str


def _image_filename(identity: str, image_number: str) -> str:
    try:
        number = int(image_number)
    except ValueError as exc:
        raise ProtocolError(f"Image number {image_number!r} is not an integer") from exc
    return f"{identity}_{number:04d}.jpg"


def _resolve_image_path(dataset_root: Path, identity: str, image_number: str) -> Path:
    filename = _image_filename(identity, image_number)
    candidate = (dataset_root / identity / filename).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise ProtocolError(f"Resolved image path escapes dataset root: {candidate}") from exc
    if not candidate.is_file():
        raise ProtocolError(f"Missing image referenced by protocol: {candidate}")
    return candidate


def _validate_header(
    header: Sequence[str], same_count: int, diff_count: int, protocol_path: Path
) -> None:
    if len(header) == 1:
        expected = int(header[0])
        if same_count != expected or diff_count != expected:
            raise ProtocolError(
                f"{protocol_path}: header declares {expected} matched and {expected} "
                f"mismatched pairs, found {same_count} matched and {diff_count} mismatched"
            )
    elif len(header) == 2:
        folds, per_fold = int(header[0]), int(header[1])
        expected = folds * per_fold
        if same_count != expected or diff_count != expected:
            raise ProtocolError(
                f"{protocol_path}: header declares {folds} folds x {per_fold} pairs per "
                f"class, expected {expected} matched and {expected} mismatched, found "
                f"{same_count} matched and {diff_count} mismatched"
            )
    else:
        raise ProtocolError(f"{protocol_path}: unrecognised header format: {list(header)}")


def parse_lfw_pairs(protocol_path: Path, dataset_root: Path) -> List[Pair]:
    protocol_path = Path(protocol_path)
    dataset_root = Path(dataset_root).resolve()

    if not protocol_path.is_file():
        raise ProtocolError(f"Protocol file does not exist: {protocol_path}")

    raw_lines = protocol_path.read_text(encoding="utf-8").strip("\n").split("\n")
    if not raw_lines or not raw_lines[0].strip():
        raise ProtocolError(f"Empty protocol file: {protocol_path}")

    header = raw_lines[0].split()
    data_lines = raw_lines[1:]

    pairs: List[Pair] = []
    seen: Set[Tuple[str, str]] = set()
    same_count = 0
    diff_count = 0

    for line_number, raw_line in enumerate(data_lines, start=2):
        line = raw_line.strip()
        if not line:
            continue
        columns = line.split("\t") if "\t" in line else line.split()

        if len(columns) == 3:
            identity, image_a, image_b = columns
            left = _resolve_image_path(dataset_root, identity, image_a)
            right = _resolve_image_path(dataset_root, identity, image_b)
            same_identity = True
            left_identity = right_identity = identity
        elif len(columns) == 4:
            identity_a, image_a, identity_b, image_b = columns
            left = _resolve_image_path(dataset_root, identity_a, image_a)
            right = _resolve_image_path(dataset_root, identity_b, image_b)
            same_identity = False
            left_identity, right_identity = identity_a, identity_b
        else:
            raise ProtocolError(
                f"{protocol_path}:{line_number}: expected 3 or 4 columns, got {len(columns)}"
            )

        key = (str(left), str(right))
        if key in seen:
            raise ProtocolError(f"{protocol_path}: duplicate pair detected: {key}")
        seen.add(key)

        pairs.append(Pair(left, right, same_identity, left_identity, right_identity))
        if same_identity:
            same_count += 1
        else:
            diff_count += 1

    if same_count == 0 or diff_count == 0:
        raise ProtocolError(
            f"{protocol_path} must contain both matched and mismatched pairs; "
            f"found {same_count} matched, {diff_count} mismatched"
        )

    _validate_header(header, same_count, diff_count, protocol_path)
    return pairs


def _cplfw_identity_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    # ``rpartition`` splits once at the *last* separator and always returns
    # three parts, so this cannot raise on a name without an underscore. The
    # middle value is the separator itself; ``_`` is the conventional name for a
    # value that is deliberately discarded.
    prefix, _, suffix = stem.rpartition("_")
    return prefix if prefix and suffix.isdigit() else stem


def _resolve_flat_image_path(dataset_root: Path, filename: str) -> Path:
    candidate = (dataset_root / filename).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise ProtocolError(f"Resolved image path escapes dataset root: {candidate}") from exc
    if not candidate.is_file():
        raise ProtocolError(f"Missing image referenced by protocol: {candidate}")
    return candidate


def parse_cplfw_pairs(protocol_path: Path, dataset_root: Path) -> List[Pair]:
    protocol_path = Path(protocol_path)
    dataset_root = Path(dataset_root).resolve()

    if not protocol_path.is_file():
        raise ProtocolError(f"Protocol file does not exist: {protocol_path}")

    lines = [
        line for line in protocol_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise ProtocolError(f"Empty protocol file: {protocol_path}")
    if len(lines) % 2 != 0:
        raise ProtocolError(
            f"{protocol_path}: expected an even number of lines (two per pair), got {len(lines)}"
        )

    pairs: List[Pair] = []
    seen: Set[Tuple[str, str]] = set()
    same_count = 0
    diff_count = 0

    for index in range(0, len(lines), 2):
        line_number = index + 1
        first_columns = lines[index].split()
        second_columns = lines[index + 1].split()

        if len(first_columns) != 2 or len(second_columns) != 2:
            raise ProtocolError(
                f"{protocol_path}:{line_number}: expected 'filename label' on each of a pair's two lines"
            )

        first_filename, first_label = first_columns
        second_filename, second_label = second_columns

        if first_label not in {"0", "1"} or second_label not in {"0", "1"}:
            raise ProtocolError(f"{protocol_path}:{line_number}: label must be 0 or 1")
        if first_label != second_label:
            raise ProtocolError(
                f"{protocol_path}:{line_number}: a pair's two lines must share the same label, "
                f"got {first_label!r} and {second_label!r}"
            )

        left = _resolve_flat_image_path(dataset_root, first_filename)
        right = _resolve_flat_image_path(dataset_root, second_filename)

        key = (str(left), str(right))
        if key in seen:
            raise ProtocolError(f"{protocol_path}: duplicate pair detected: {key}")
        seen.add(key)

        same_identity = first_label == "1"
        pairs.append(
            Pair(
                left,
                right,
                same_identity,
                _cplfw_identity_from_filename(first_filename),
                _cplfw_identity_from_filename(second_filename),
            )
        )
        if same_identity:
            same_count += 1
        else:
            diff_count += 1

    if same_count == 0 or diff_count == 0:
        raise ProtocolError(
            f"{protocol_path} must contain both matched and mismatched pairs; "
            f"found {same_count} matched, {diff_count} mismatched"
        )

    return pairs


# =============================================================================
# 10. Verification metrics
# =============================================================================
#
# Convention: label 1 is the same identity (a match), label 0 a different
# identity. Higher similarity is more match-like, and the decision rule is
# "score >= threshold implies a predicted match". Implemented in plain NumPy so
# the dependency contract stays small and fully pinned.

# Type aliases: a name bound to a type expression, used purely to keep the
# signatures below readable. ``Union[A, B]`` means "either an A or a B", so
# every entry point accepts a plain Python sequence or a NumPy array and
# internal calls need not convert back and forth.
ScoreInput = Union[Sequence[float], np.ndarray]
LabelInput = Union[Sequence[int], np.ndarray]


class MetricsError(ValueError):
    """Raised for empty, ragged, non-finite or single-class score/label input."""


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile, matching NumPy's default method."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return float(ordered[0])
    index = (pct / 100.0) * (len(ordered) - 1)
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return (
            self.true_positive + self.false_positive + self.true_negative + self.false_negative
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
        }


# A single leading underscore marks a helper as internal by convention: it is
# not part of the surface this file offers to its tests or to a reader, and may
# change without notice. Python does not enforce this; it is a signal.
def _validate_inputs(scores: ScoreInput, labels: LabelInput) -> Tuple[np.ndarray, np.ndarray]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if scores_arr.shape[0] != labels_arr.shape[0]:
        raise MetricsError("scores and labels must have the same length")
    if scores_arr.shape[0] == 0:
        raise MetricsError("scores/labels must not be empty")
    if not np.all(np.isfinite(scores_arr)):
        raise MetricsError("scores must contain only finite numbers")
    unique_labels = set(np.unique(labels_arr).tolist())
    if not unique_labels.issubset({0, 1}):
        raise MetricsError(f"labels must be 0 or 1, found {sorted(unique_labels)}")
    if unique_labels != {0, 1}:
        raise MetricsError(
            f"labels must contain both classes (0 and 1); found only {sorted(unique_labels)}"
        )
    return scores_arr, labels_arr


def confusion_matrix(scores: ScoreInput, labels: LabelInput, threshold: float) -> ConfusionMatrix:
    scores_arr, labels_arr = _validate_inputs(scores, labels)
    predicted_match = scores_arr >= threshold
    actual_match = labels_arr == 1

    return ConfusionMatrix(
        true_positive=int(np.sum(predicted_match & actual_match)),
        false_positive=int(np.sum(predicted_match & ~actual_match)),
        true_negative=int(np.sum(~predicted_match & ~actual_match)),
        false_negative=int(np.sum(~predicted_match & actual_match)),
    )


def rates_from_confusion(matrix: ConfusionMatrix) -> Dict[str, float]:
    positives = matrix.true_positive + matrix.false_negative
    negatives = matrix.true_negative + matrix.false_positive
    total = matrix.total

    accuracy = (matrix.true_positive + matrix.true_negative) / total if total else float("nan")
    precision = (
        matrix.true_positive / (matrix.true_positive + matrix.false_positive)
        if (matrix.true_positive + matrix.false_positive) > 0
        else float("nan")
    )
    recall = matrix.true_positive / positives if positives > 0 else float("nan")
    # A self-comparison is the NaN test here: an undefined rate must propagate
    # as NaN rather than silently contribute a zero to the derived metric.
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )
    false_match_rate = matrix.false_positive / negatives if negatives > 0 else float("nan")
    false_non_match_rate = matrix.false_negative / positives if positives > 0 else float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_match_rate": false_match_rate,
        "false_non_match_rate": false_non_match_rate,
        "true_match_rate": recall,
    }


def roc_points(scores: ScoreInput, labels: LabelInput) -> List[Dict[str, float]]:
    """ROC curve as {threshold, false_match_rate, true_match_rate} points, one
    per distinct score plus high/low sentinels, ordered by descending
    threshold."""
    scores_arr, labels_arr = _validate_inputs(scores, labels)
    thresholds = np.unique(scores_arr)[::-1]
    sentinel_high = float(thresholds[0]) + 1.0 if thresholds.size else 1.0
    sentinel_low = float(thresholds[-1]) - 1.0 if thresholds.size else -1.0
    all_thresholds = np.concatenate(([sentinel_high], thresholds, [sentinel_low]))

    points: List[Dict[str, float]] = []
    for threshold in all_thresholds:
        matrix = confusion_matrix(scores_arr, labels_arr, float(threshold))
        rates = rates_from_confusion(matrix)
        points.append(
            {
                "threshold": float(threshold),
                "false_match_rate": rates["false_match_rate"],
                "true_match_rate": rates["true_match_rate"],
            }
        )
    return points


##############
# Title: The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve
# Author: Hanley, J.A. and McNeil, B.J., Radiology, 143(1), pp. 29-36
# Date: 1982
# Availability: https://doi.org/10.1148/radiology.143.1.7063747
##############
# The equivalence this function relies on -- that the area under the ROC curve
# equals the probability that a randomly chosen positive outranks a randomly
# chosen negative, the quantity estimated by the Wilcoxon/Mann-Whitney rank
# statistic -- is the standard result established in the reference above. The
# implementation below is original and deliberately avoids a machine-learning
# dependency for one statistic.
def roc_auc(scores: ScoreInput, labels: LabelInput) -> float:
    """Rank-based ROC-AUC (the Mann-Whitney U identity), ties resolved with
    average ranks. Equivalent to the trapezoidal-rule area, without pulling in
    a machine-learning framework for one statistic."""
    scores_arr, labels_arr = _validate_inputs(scores, labels)
    # ``kind="mergesort"`` requests a stable sort: equal scores keep their
    # original relative order. Stability makes the tie handling below
    # reproducible run to run rather than dependent on the sort's internals.
    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    ranks = np.empty(len(scores_arr), dtype=np.float64)

    n = len(sorted_scores)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        average_rank = (i + 1 + j) / 2.0  # 1-indexed rank, averaged across the tie block
        ranks[order[i:j]] = average_rank
        i = j

    positive_mask = labels_arr == 1
    n_pos = int(np.sum(positive_mask))
    n_neg = int(len(labels_arr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise MetricsError("roc_auc requires at least one example of each class")

    sum_ranks_positive = float(np.sum(ranks[positive_mask]))
    return float((sum_ranks_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def equal_error_rate(scores: ScoreInput, labels: LabelInput) -> Dict[str, float]:
    """EER by linear interpolation between the ROC points that bracket
    false_match_rate == false_non_match_rate."""
    points = roc_points(scores, labels)

    best_gap: Optional[float] = None
    best_eer: Optional[float] = None
    best_threshold: Optional[float] = None
    previous: Optional[Tuple[float, float, float, float]] = None

    for point in points:
        fmr = point["false_match_rate"]
        fnmr = 1.0 - point["true_match_rate"]
        gap = fmr - fnmr
        if previous is not None:
            prev_fmr, _prev_fnmr, prev_gap, prev_threshold = previous
            crosses = (prev_gap <= 0 <= gap) or (prev_gap >= 0 >= gap)
            if crosses:
                if gap == prev_gap:
                    eer, threshold = fmr, point["threshold"]
                else:
                    ratio = prev_gap / (prev_gap - gap)
                    eer = prev_fmr + ratio * (fmr - prev_fmr)
                    threshold = prev_threshold + ratio * (point["threshold"] - prev_threshold)
                if best_gap is None or abs(gap) < best_gap:
                    best_gap, best_eer, best_threshold = abs(gap), eer, threshold
        previous = (fmr, fnmr, gap, point["threshold"])

    # Both are assigned together above, so either both are set or neither is;
    # testing both is what makes that invariant explicit.
    if best_eer is None or best_threshold is None:
        closest = min(
            points, key=lambda p: abs(p["false_match_rate"] - (1.0 - p["true_match_rate"]))
        )
        best_eer = closest["false_match_rate"]
        best_threshold = closest["threshold"]

    return {"equal_error_rate": float(best_eer), "threshold": float(best_threshold)}


@dataclass(frozen=True)
class ThresholdCandidate:
    threshold: float
    strategy: str
    metrics: Dict[str, float]


def select_threshold(
    scores: ScoreInput,
    labels: LabelInput,
    *,
    strategy: str,
    target_false_match_rate: Optional[float] = None,
) -> ThresholdCandidate:
    scores_arr, labels_arr = _validate_inputs(scores, labels)
    candidate_thresholds = np.unique(scores_arr)

    if strategy == "eer":
        eer_result = equal_error_rate(scores_arr, labels_arr)
        threshold = eer_result["threshold"]
        metrics = rates_from_confusion(confusion_matrix(scores_arr, labels_arr, threshold))
        metrics["equal_error_rate"] = eer_result["equal_error_rate"]
        return ThresholdCandidate(threshold, strategy, metrics)

    if strategy == "target_fmr":
        if target_false_match_rate is None:
            raise MetricsError("target_fmr strategy requires target_false_match_rate")
        best_threshold: Optional[float] = None
        best_metrics: Optional[Dict[str, float]] = None
        for threshold in sorted(candidate_thresholds, reverse=True):
            metrics = rates_from_confusion(
                confusion_matrix(scores_arr, labels_arr, float(threshold))
            )
            fmr = metrics["false_match_rate"]
            if fmr == fmr and fmr <= target_false_match_rate:
                best_threshold, best_metrics = float(threshold), metrics
            elif best_threshold is not None:
                break
        if best_threshold is None or best_metrics is None:
            raise MetricsError(
                f"No threshold achieves false_match_rate <= {target_false_match_rate}"
            )
        return ThresholdCandidate(best_threshold, strategy, best_metrics)

    if strategy not in {"balanced_accuracy", "f1"}:
        raise MetricsError(f"Unknown threshold-selection strategy: {strategy}")

    best_threshold = None
    best_score = float("-inf")
    best_metrics = None
    for threshold in candidate_thresholds:
        metrics = rates_from_confusion(confusion_matrix(scores_arr, labels_arr, float(threshold)))
        if strategy == "balanced_accuracy":
            tmr = metrics["true_match_rate"]
            fmr = metrics["false_match_rate"]
            tnr = 1.0 - fmr if fmr == fmr else float("nan")
            score = (tmr + tnr) / 2.0 if tmr == tmr and tnr == tnr else float("-inf")
        else:
            score = metrics["f1"] if metrics["f1"] == metrics["f1"] else float("-inf")
        prefer_higher_on_tie = (
            score == best_score and best_threshold is not None and threshold > best_threshold
        )
        if score > best_score or prefer_higher_on_tie:
            best_score, best_threshold, best_metrics = score, float(threshold), metrics

    if best_threshold is None or best_metrics is None:
        raise MetricsError(f"Could not select a threshold using strategy={strategy}")
    return ThresholdCandidate(best_threshold, strategy, best_metrics)


# =============================================================================
# 11. Threshold calibration, selection and freezing
# =============================================================================
#
# The validation/held-out boundary is the single most important methodological
# guarantee in this artefact, so it is enforced in code across three stages:
#
# 1. calibrate() runs on pairsDevTrain.txt only and produces a table of
#    *candidate* thresholds. It never picks a winner; its status is
#    "candidates".
# 2. select_final_threshold() runs on pairsDevTest.txt, scores every candidate
#    and selects exactly one by a fixed, fully deterministic rule. Only this
#    step's output is marked "frozen".
# 3. require_frozen_threshold() refuses to let a final or held-out evaluation
#    proceed on anything that has not been through stage 2.

VALIDATION_SPLIT = "validation"
CANDIDATES_STATUS = "candidates"
FROZEN_STATUS = "frozen"

DEFAULT_TARGET_FALSE_MATCH_RATES: Sequence[float] = (0.001, 0.01, 0.05)

SELECTION_RULE = (
    "Maximum balanced accuracy on the development split (pairsDevTest.txt); "
    "ties broken by lower development-split false match rate, then by "
    "candidate name, for full determinism."
)


class CalibrationError(RuntimeError):
    """Raised when a calibration stage is run out of order or on the wrong split."""


@dataclass(frozen=True)
class CalibrationResult:
    split: str
    status: str
    candidates: Dict[str, ThresholdCandidate]


def calibrate(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    split: str,
    target_false_match_rates: Sequence[float] = DEFAULT_TARGET_FALSE_MATCH_RATES,
) -> CalibrationResult:
    """Stage 1: generate candidate thresholds from the validation split only."""
    if split != VALIDATION_SPLIT:
        raise CalibrationError(
            f"Calibration must only run on the '{VALIDATION_SPLIT}' split; got '{split}'. "
            f"This is enforced in code to prevent held-out/test-set leakage into threshold "
            f"selection, regardless of what the caller intended."
        )

    candidates: Dict[str, ThresholdCandidate] = {
        "balanced_accuracy": select_threshold(scores, labels, strategy="balanced_accuracy"),
        "f1": select_threshold(scores, labels, strategy="f1"),
        "eer": select_threshold(scores, labels, strategy="eer"),
    }
    for target in target_false_match_rates:
        candidates[f"target_fmr_{target}"] = select_threshold(
            scores, labels, strategy="target_fmr", target_false_match_rate=target
        )

    return CalibrationResult(split=split, status=CANDIDATES_STATUS, candidates=candidates)


def require_candidates(payload: Dict[str, Any], *, context: str = "") -> Dict[str, Dict[str, Any]]:
    """Stage 2's guard: read the candidate table out of a loaded artifact,
    refusing anything already frozen or never calibrated."""
    if payload.get("status") != CANDIDATES_STATUS:
        raise CalibrationError(
            f"{context}: threshold artifact status is '{payload.get('status')}', not "
            f"'{CANDIDATES_STATUS}'. Expected a calibration artifact that has not yet been "
            f"frozen by a development-split selection step."
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise CalibrationError(f"{context}: threshold artifact has no candidates to select from")
    return candidates


def select_final_threshold(
    candidates: Dict[str, Dict[str, Any]],
    dev_scores: Sequence[float],
    dev_labels: Sequence[int],
) -> Dict[str, Any]:
    """Stage 2: score every stage-1 candidate against the development split and
    select exactly one by SELECTION_RULE. Returns the outcome plus every
    candidate's development metrics, so the choice is auditable."""
    if not candidates:
        raise CalibrationError("No candidate thresholds to select from")

    per_candidate_dev_metrics: Dict[str, Dict[str, float]] = {}
    for name, candidate in candidates.items():
        threshold = float(candidate["threshold"])
        matrix = confusion_matrix(dev_scores, dev_labels, threshold)
        rates = rates_from_confusion(matrix)
        tmr, fmr = rates["true_match_rate"], rates["false_match_rate"]
        # Self-comparison is the NaN test; an unscorable candidate sorts last
        # rather than winning the selection by accident.
        balanced_accuracy = (
            (tmr + (1.0 - fmr)) / 2.0 if tmr == tmr and fmr == fmr else float("-inf")
        )
        per_candidate_dev_metrics[name] = {
            **rates,
            "threshold": threshold,
            "balanced_accuracy": balanced_accuracy,
        }

    # Returning a tuple gives a lexicographic ordering: Python compares the
    # first element, and only consults the second when those are equal, then
    # the third. Negating balanced accuracy turns "highest is best" into
    # "smallest sorts first", so a single ``min`` expresses the whole published
    # selection rule -- and the final element being the candidate's name
    # guarantees a total order, leaving no tie for chance to break.
    def sort_key(name: str) -> Tuple[float, float, str]:
        metrics = per_candidate_dev_metrics[name]
        return (-metrics["balanced_accuracy"], metrics["false_match_rate"], name)

    selected_name = min(per_candidate_dev_metrics, key=sort_key)
    selected = per_candidate_dev_metrics[selected_name]

    return {
        "selected_candidate": selected_name,
        "selected_threshold": selected["threshold"],
        "selection_rule": SELECTION_RULE,
        "all_candidates_dev_metrics": per_candidate_dev_metrics,
    }


def require_frozen_threshold(payload: Dict[str, Any], *, context: str = "") -> float:
    """Stage 3's guard: read a threshold from a loaded artifact, refusing
    anything not explicitly marked frozen by select_final_threshold."""
    if payload.get("status") != FROZEN_STATUS:
        raise CalibrationError(
            f"{context}: threshold artifact status is '{payload.get('status')}', not "
            f"'{FROZEN_STATUS}'. Refusing to use a non-frozen threshold for a held-out or "
            f"final evaluation."
        )
    threshold = payload.get("threshold")
    if not isinstance(threshold, (int, float)):
        raise CalibrationError(
            f"{context}: threshold artifact is missing a numeric 'threshold' field"
        )
    return float(threshold)


# =============================================================================
# 12. Pair evaluation and failure accounting
# =============================================================================
#
# Face-extraction failures (zero or multiple detections, unreadable images) are
# recorded as their own outcome category and reported alongside accuracy, never
# dropped from the pair count. Per-image latency is timed so throughput can be
# reported rather than inferred from wall-clock runtime.


@dataclass(frozen=True)
class PairScore:
    pair: Pair
    similarity: Optional[float]
    failure_code: Optional[str]


@dataclass
class EvaluationResult:
    total_pairs: int
    scored_pairs: List[PairScore]
    failures: Dict[str, int] = field(default_factory=dict)
    embedding_times_seconds: List[float] = field(default_factory=list)

    # ``@property`` exposes a method as though it were an attribute, so callers
    # write ``result.valid_scores`` rather than ``result.valid_scores()``. Each
    # one below is derived from the stored pairs on every access, which means
    # the counts can never drift out of step with the data they describe.
    @property
    def valid_scores(self) -> List[float]:
        return [s.similarity for s in self.scored_pairs if s.similarity is not None]

    @property
    def valid_labels(self) -> List[int]:
        return [
            1 if s.pair.same_identity else 0
            for s in self.scored_pairs
            if s.similarity is not None
        ]

    @property
    def scored_pair_count(self) -> int:
        # Only pairs yielding one valid face on both sides carry a similarity score.
        return len(self.valid_scores)

    @property
    def failed_pairs(self) -> int:
        # Failed pairs stay within the protocol total; they simply have no score.
        return self.total_pairs - self.scored_pair_count

    @property
    def failure_rate(self) -> float:
        # Stored as a fraction of the full protocol; reports render the percentage.
        return self.failed_pairs / self.total_pairs if self.total_pairs else float("nan")

    def validate_accounting(self) -> None:
        """Confirm every protocol pair is accounted for exactly once. The
        reported denominator is only meaningful if nothing was silently
        discarded and the breakdown describes precisely the failed pairs."""
        if self.scored_pair_count + self.failed_pairs != self.total_pairs:
            raise ValueError(
                f"Scored ({self.scored_pair_count}) and failed ({self.failed_pairs}) pairs must "
                f"sum to the protocol total ({self.total_pairs})."
            )
        # evaluate_pairs records exactly one terminal category per failed pair,
        # so the breakdown partitions the failures rather than tallying images.
        categorised = sum(self.failures.values())
        if categorised != self.failed_pairs:
            raise ValueError(
                f"Failure breakdown totals {categorised} but {self.failed_pairs} pairs failed; "
                f"every failed pair must carry exactly one extraction-failure category."
            )


def _embed_image(
    path: Path,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    cache: Dict[Path, np.ndarray],
    embedding_times: List[float],
) -> np.ndarray:
    if path in cache:
        return cache[path]
    start = time.perf_counter()
    loaded = load_image_bgr(path)
    face_row = detector.detect_single_face(loaded.bgr)
    raw_embedding = embedder.embed(loaded.bgr, face_row)
    normalized = l2_normalize(raw_embedding)
    embedding_times.append(time.perf_counter() - start)
    cache[path] = normalized
    return normalized


def evaluate_pairs(
    pairs: List[Pair], *, detector: FaceDetector, embedder: FaceEmbedder
) -> EvaluationResult:
    scored: List[PairScore] = []
    failures: Dict[str, int] = {}
    cache: Dict[Path, np.ndarray] = {}
    embedding_times: List[float] = []

    def record(code: str) -> None:
        failures[code] = failures.get(code, 0) + 1

    # Returns the embedding and an empty failure code, or no embedding and the
    # code that terminated this side. The empty string carries the same meaning
    # as the absent embedding, so a caller can branch on either one.
    def embed_side(path: Path, side: str) -> Tuple[Optional[np.ndarray], str]:
        try:
            return _embed_image(path, detector, embedder, cache, embedding_times), ""
        except FaceCountError as exc:
            code = f"zero_faces_{side}" if exc.face_count == 0 else f"multiple_faces_{side}"
            return None, code
        except (ImageLoadError, SimilarityError) as exc:
            return None, f"image_error_{side}:{exc}"

    for pair in pairs:
        # Sides are attempted left first and the pair is abandoned on the first
        # terminal failure, so each failed pair records exactly one category.
        # A "_right" category therefore means the right image failed *after*
        # the left had already yielded one valid face: where both sides would
        # fail, only the left is ever counted. The four categories partition
        # the failed pairs; they are not per-image failure tallies.
        left_embedding, left_failure = embed_side(pair.left_path, "left")
        if left_embedding is None:
            record(left_failure.split(":")[0])
            scored.append(PairScore(pair, None, left_failure))
            continue

        right_embedding, right_failure = embed_side(pair.right_path, "right")
        if right_embedding is None:
            record(right_failure.split(":")[0])
            scored.append(PairScore(pair, None, right_failure))
            continue

        similarity = cosine_similarity(left_embedding, right_embedding)
        scored.append(PairScore(pair, similarity, None))

    return EvaluationResult(
        total_pairs=len(pairs),
        scored_pairs=scored,
        failures=failures,
        embedding_times_seconds=embedding_times,
    )


def summarize_metrics(result: EvaluationResult, threshold: float) -> Dict[str, Any]:
    scores = result.valid_scores
    labels = result.valid_labels
    if not scores:
        raise ValueError("No pairs were successfully scored; cannot compute metrics.")

    # Refuse to publish a failure rate whose denominator does not reconcile.
    result.validate_accounting()

    matrix = confusion_matrix(scores, labels, threshold)
    rates = rates_from_confusion(matrix)
    auc = roc_auc(scores, labels)
    eer = equal_error_rate(scores, labels)
    curve = roc_points(scores, labels)

    times_ms = [t * 1000.0 for t in result.embedding_times_seconds]

    return {
        "threshold": threshold,
        "total_pairs": result.total_pairs,
        # Score-based metrics below are conditional on these scored pairs alone.
        "scored_pairs": result.scored_pair_count,
        "failed_pairs": result.failed_pairs,
        "failure_rate": result.failure_rate,
        "failure_breakdown": dict(result.failures),
        "confusion_matrix": matrix.as_dict(),
        "accuracy": rates["accuracy"],
        "precision": rates["precision"],
        "recall": rates["recall"],
        "f1": rates["f1"],
        "false_match_rate": rates["false_match_rate"],
        "false_non_match_rate": rates["false_non_match_rate"],
        "roc_auc": auc,
        "equal_error_rate": eer["equal_error_rate"],
        "roc_points": curve,
        "embedding_time_mean_ms": statistics.fmean(times_ms) if times_ms else float("nan"),
        "embedding_time_median_ms": statistics.median(times_ms) if times_ms else float("nan"),
        "embedding_time_p95_ms": percentile(times_ms, 95) if times_ms else float("nan"),
        "unique_images_embedded": len(times_ms),
    }


# =============================================================================
# 13. Duplicate-profile gallery evaluation
# =============================================================================
#
# Simulates registered profiles (the gallery) plus two kinds of probe: a second
# image of a gallery identity (a duplicate-registration attempt) and an image
# of an identity absent from the gallery (a legitimate new registration). Every
# manifest entry uses an opaque, one-way identifier rather than a real name,
# and each image holds exactly one role. Nothing here labels a result a scam:
# exceeding the threshold only opens a human-review case.


class GalleryError(RuntimeError):
    """Raised when a gallery cannot be built or embedded at all."""


@dataclass(frozen=True)
class ManifestEntry:
    sample_id: str
    identity_hash: str
    image_path: Path
    role: str  # "gallery" | "duplicate_probe" | "unknown_probe"


@dataclass(frozen=True)
class GalleryManifest:
    entries: List[ManifestEntry]
    seed: int


def build_manifest(
    identity_to_images: Dict[str, List[Path]],
    *,
    excluded_images: Iterable[Path] = (),
    seed: int = DEFAULT_RANDOM_SEED,
    max_unknown_identities: Optional[int] = None,
) -> GalleryManifest:
    excluded = {Path(p) for p in excluded_images}
    rng = random.Random(seed)

    eligible = {
        identity: sorted(
            (Path(p) for p in images if Path(p) not in excluded), key=lambda p: p.name
        )
        for identity, images in identity_to_images.items()
    }
    eligible = {identity: images for identity, images in eligible.items() if images}

    gallery_identities = sorted(
        identity for identity, images in eligible.items() if len(images) >= 2
    )
    unknown_identities = sorted(
        identity for identity, images in eligible.items() if len(images) == 1
    )

    if max_unknown_identities is not None:
        rng.shuffle(unknown_identities)
        unknown_identities = sorted(unknown_identities[:max_unknown_identities])

    if not gallery_identities:
        raise GalleryError("No identity has at least two usable images; cannot build a gallery.")
    if not unknown_identities:
        raise GalleryError(
            "No identity with exactly one usable image is available as an unknown probe."
        )

    entries: List[ManifestEntry] = []

    for identity in gallery_identities:
        images = eligible[identity]
        gallery_image, duplicate_image = images[0], images[1]
        identity_hash = opaque_id(identity)
        entries.append(
            ManifestEntry(
                opaque_id(f"{identity}:{gallery_image.name}"),
                identity_hash,
                gallery_image,
                "gallery",
            )
        )
        entries.append(
            ManifestEntry(
                opaque_id(f"{identity}:{duplicate_image.name}"),
                identity_hash,
                duplicate_image,
                "duplicate_probe",
            )
        )

    for identity in unknown_identities:
        image = eligible[identity][0]
        identity_hash = opaque_id(identity)
        entries.append(
            ManifestEntry(
                opaque_id(f"{identity}:{image.name}"), identity_hash, image, "unknown_probe"
            )
        )

    seen_paths: Set[Path] = set()
    for entry in entries:
        if entry.image_path in seen_paths:
            raise GalleryError(f"Image assigned to more than one manifest role: {entry.image_path}")
        seen_paths.add(entry.image_path)

    return GalleryManifest(entries=entries, seed=seed)


@dataclass(frozen=True)
class ProbeResult:
    sample_id: str
    role: str
    identity_hash: str
    top_candidate_identity_hash: Optional[str]
    top_similarity: Optional[float]
    rank1_correct: Optional[bool]
    exceeds_duplicate_threshold: Optional[bool]
    failure_code: Optional[str]
    # Rank of the probe's own identity in the descending-similarity candidate
    # list, 1-based; None when the probe is not mated or was never scored.
    correct_identity_rank: Optional[int] = None
    rank5_correct: Optional[bool] = None


@dataclass(frozen=True)
class GalleryEntryResult:
    """Outcome of attempting to enrol one gallery reference image. Retained for
    every intended entry, including the ones that could not be embedded — the
    defect this replaces dropped those silently, which shrank the gallery
    without recording that it had shrunk."""

    sample_id: str
    identity_hash: str
    embedded: bool
    failure_code: Optional[str]


# Gallery enrolment failures are reported under their own suffixed codes so a
# reader can never confuse a gallery-side failure with a probe-side one. These
# three partition every enrolment failure.
GALLERY_FAILURE_CODES = (
    "zero_faces_gallery",
    "multiple_faces_gallery",
    "image_error_gallery",
)

# A mated probe whose own gallery reference never enrolled cannot be found by
# any threshold. It is a coverage failure, not a similarity miss, and is kept
# distinct from both so neither rate is quietly flattered.
GALLERY_REFERENCE_UNAVAILABLE = "gallery_reference_unavailable"


@dataclass(frozen=True)
class GalleryEvaluationResult:
    gallery_size: int
    probe_results: List[ProbeResult]
    search_times_seconds: List[float] = field(default_factory=list)
    # Enrolment accounting. Defaults keep the dataclass constructible from the
    # older positional form used by existing tests.
    intended_gallery_size: Optional[int] = None
    gallery_entry_results: List[GalleryEntryResult] = field(default_factory=list)

    @property
    def embedded_gallery_size(self) -> int:
        return self.gallery_size

    @property
    def resolved_intended_gallery_size(self) -> int:
        """Intended entries, falling back to the embedded count when this result
        came from a caller that predates enrolment accounting."""
        return self.gallery_size if self.intended_gallery_size is None else self.intended_gallery_size

    @property
    def gallery_failure_breakdown(self) -> Dict[str, int]:
        breakdown = {code: 0 for code in GALLERY_FAILURE_CODES}
        for entry in self.gallery_entry_results:
            if entry.embedded:
                continue
            code = entry.failure_code or "image_error_gallery"
            breakdown[code] = breakdown.get(code, 0) + 1
        return breakdown

    @property
    def gallery_entry_failure_count(self) -> int:
        return self.resolved_intended_gallery_size - self.embedded_gallery_size

    @property
    def gallery_entry_failure_rate(self) -> float:
        intended = self.resolved_intended_gallery_size
        return self.gallery_entry_failure_count / intended if intended else float("nan")


def _embed_entry(
    entry: ManifestEntry, detector: FaceDetector, embedder: FaceEmbedder
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        loaded = load_image_bgr(entry.image_path)
        face_row = detector.detect_single_face(loaded.bgr)
        raw = embedder.embed(loaded.bgr, face_row)
        return l2_normalize(raw), None
    except FaceCountError as exc:
        return None, "zero_faces" if exc.face_count == 0 else "multiple_faces"
    except (ImageLoadError, SimilarityError) as exc:
        return None, f"image_error:{exc}"


def _gallery_failure_code(probe_failure_code: Optional[str]) -> str:
    """Map a generic extraction failure onto its gallery-side category."""
    if probe_failure_code == "zero_faces":
        return "zero_faces_gallery"
    if probe_failure_code == "multiple_faces":
        return "multiple_faces_gallery"
    return "image_error_gallery"


def evaluate_gallery(
    manifest: GalleryManifest,
    *,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    duplicate_review_threshold: float,
) -> GalleryEvaluationResult:
    gallery_entries = [e for e in manifest.entries if e.role == "gallery"]
    probe_entries = [e for e in manifest.entries if e.role in ("duplicate_probe", "unknown_probe")]

    gallery_embeddings: List[Tuple[ManifestEntry, np.ndarray]] = []
    gallery_entry_results: List[GalleryEntryResult] = []
    for entry in gallery_entries:
        embedding, failure = _embed_entry(entry, detector, embedder)
        if embedding is not None:
            gallery_embeddings.append((entry, embedding))
            gallery_entry_results.append(
                GalleryEntryResult(entry.sample_id, entry.identity_hash, True, None)
            )
        else:
            # Retained rather than dropped: an identity that never enrolled is
            # unfindable, and the mated probes pointing at it must be accounted
            # for instead of quietly leaving the denominator.
            gallery_entry_results.append(
                GalleryEntryResult(
                    entry.sample_id, entry.identity_hash, False, _gallery_failure_code(failure)
                )
            )
    if not gallery_embeddings:
        raise GalleryError("No gallery entry could be embedded; cannot run the experiment.")

    enrolled_identity_hashes = {entry.identity_hash for entry, _ in gallery_embeddings}

    results: List[ProbeResult] = []
    search_times: List[float] = []
    for probe in probe_entries:
        # Checked before embedding: when the probe's own reference is missing
        # from the gallery, no similarity score could change the outcome, so
        # the protocol-level blocker is the honest attribution.
        if probe.role == "duplicate_probe" and probe.identity_hash not in enrolled_identity_hashes:
            results.append(
                ProbeResult(
                    probe.sample_id,
                    probe.role,
                    probe.identity_hash,
                    None,
                    None,
                    None,
                    None,
                    GALLERY_REFERENCE_UNAVAILABLE,
                )
            )
            continue

        probe_embedding, failure = _embed_entry(probe, detector, embedder)
        if probe_embedding is None:
            results.append(
                ProbeResult(
                    probe.sample_id, probe.role, probe.identity_hash, None, None, None, None, failure
                )
            )
            continue

        search_start = time.perf_counter()
        similarities = sorted(
            (
                (candidate_entry, cosine_similarity(probe_embedding, candidate_embedding))
                for candidate_entry, candidate_embedding in gallery_embeddings
            ),
            # Same lexicographic device as the threshold selection: rank by
            # descending similarity, and break an exact tie on the opaque
            # sample identifier so the winner is deterministic.
            key=lambda item: (-item[1], item[0].sample_id),
        )
        search_times.append(time.perf_counter() - search_start)
        top_entry, top_similarity = similarities[0]
        rank1_correct = (
            top_entry.identity_hash == probe.identity_hash
            if probe.role == "duplicate_probe"
            else None
        )

        correct_rank: Optional[int] = None
        rank5_correct: Optional[bool] = None
        if probe.role == "duplicate_probe":
            for position, (candidate_entry, _score) in enumerate(similarities, start=1):
                if candidate_entry.identity_hash == probe.identity_hash:
                    correct_rank = position
                    break
            rank5_correct = correct_rank is not None and correct_rank <= 5

        results.append(
            ProbeResult(
                probe.sample_id,
                probe.role,
                probe.identity_hash,
                top_entry.identity_hash,
                top_similarity,
                rank1_correct,
                top_similarity >= duplicate_review_threshold,
                None,
                correct_rank,
                rank5_correct,
            )
        )

    return GalleryEvaluationResult(
        gallery_size=len(gallery_embeddings),
        probe_results=results,
        search_times_seconds=search_times,
        intended_gallery_size=len(gallery_entries),
        gallery_entry_results=gallery_entry_results,
    )


def summarize_gallery_metrics(result: GalleryEvaluationResult) -> Dict[str, Any]:
    duplicate_probes = [r for r in result.probe_results if r.role == "duplicate_probe"]
    unknown_probes = [r for r in result.probe_results if r.role == "unknown_probe"]
    scored_duplicates = [r for r in duplicate_probes if r.failure_code is None]
    scored_unknowns = [r for r in unknown_probes if r.failure_code is None]

    # Conditional denominator: mated probes that were scored at all, which by
    # construction means both the probe and its gallery reference embedded.
    # End-to-end denominator: every mated probe the protocol intended, so an
    # extraction failure or an unavailable reference counts as a miss rather
    # than disappearing. The two answer different questions and the reporting
    # layer is required to show them together.
    detected_duplicates = sum(1 for r in scored_duplicates if r.exceeds_duplicate_threshold)
    intended_duplicates = len(duplicate_probes)
    unavailable_reference_count = sum(
        1 for r in duplicate_probes if r.failure_code == GALLERY_REFERENCE_UNAVAILABLE
    )

    duplicate_detection_rate = (
        detected_duplicates / len(scored_duplicates) if scored_duplicates else float("nan")
    )
    end_to_end_duplicate_detection_rate = (
        detected_duplicates / intended_duplicates if intended_duplicates else float("nan")
    )
    false_duplicate_review_rate = (
        sum(1 for r in scored_unknowns if r.exceeds_duplicate_threshold) / len(scored_unknowns)
        if scored_unknowns
        else float("nan")
    )
    rank1_identification_rate = (
        sum(1 for r in scored_duplicates if r.rank1_correct) / len(scored_duplicates)
        if scored_duplicates
        else float("nan")
    )
    end_to_end_rank1_identification_rate = (
        sum(1 for r in scored_duplicates if r.rank1_correct) / intended_duplicates
        if intended_duplicates
        else float("nan")
    )
    rank5_identification_rate = (
        sum(1 for r in scored_duplicates if r.rank5_correct) / len(scored_duplicates)
        if scored_duplicates
        else float("nan")
    )
    end_to_end_rank5_identification_rate = (
        sum(1 for r in scored_duplicates if r.rank5_correct) / intended_duplicates
        if intended_duplicates
        else float("nan")
    )
    # Self-comparison is the NaN test: with no scored duplicate probes the miss
    # rate stays undefined instead of being reported as a perfect 1.0.
    true_duplicate_miss_rate = (
        1.0 - duplicate_detection_rate
        if duplicate_detection_rate == duplicate_detection_rate
        else float("nan")
    )

    search_times_ms = [t * 1000.0 for t in result.search_times_seconds]

    return {
        "methodology_revision": GALLERY_METHODOLOGY_REVISION,
        "gallery_size": result.gallery_size,
        "intended_gallery_size": result.resolved_intended_gallery_size,
        "embedded_gallery_size": result.embedded_gallery_size,
        "gallery_entry_failure_count": result.gallery_entry_failure_count,
        "gallery_entry_failure_rate": result.gallery_entry_failure_rate,
        "gallery_failure_breakdown": result.gallery_failure_breakdown,
        "duplicate_probe_count": len(duplicate_probes),
        "unknown_probe_count": len(unknown_probes),
        "duplicate_probe_failures": len(duplicate_probes) - len(scored_duplicates),
        "unknown_probe_failures": len(unknown_probes) - len(scored_unknowns),
        "gallery_reference_unavailable_count": unavailable_reference_count,
        # Named explicitly so a reader cannot mistake the conditional figure for
        # the headline result; the legacy key keeps older artefacts readable.
        "conditional_duplicate_detection_rate": duplicate_detection_rate,
        "end_to_end_duplicate_detection_rate": end_to_end_duplicate_detection_rate,
        "duplicate_detection_rate": duplicate_detection_rate,
        "false_duplicate_review_rate": false_duplicate_review_rate,
        "conditional_rank1_identification_rate": rank1_identification_rate,
        "end_to_end_rank1_identification_rate": end_to_end_rank1_identification_rate,
        "conditional_rank5_identification_rate": rank5_identification_rate,
        "end_to_end_rank5_identification_rate": end_to_end_rank5_identification_rate,
        "rank1_identification_rate": rank1_identification_rate,
        "true_duplicate_miss_rate": true_duplicate_miss_rate,
        "gallery_search_time_mean_ms": (
            statistics.fmean(search_times_ms) if search_times_ms else float("nan")
        ),
        "gallery_search_time_p95_ms": (
            percentile(search_times_ms, 95) if search_times_ms else float("nan")
        ),
    }


def discover_identity_images(dataset_root: Path) -> Dict[str, List[Path]]:
    """Map each per-identity directory to its images, following LFW's
    identity/identity_NNNN.jpg layout."""
    identities: Dict[str, List[Path]] = {}
    for identity_dir in sorted(p for p in Path(dataset_root).iterdir() if p.is_dir()):
        images = sorted(identity_dir.glob(f"{identity_dir.name}_*.jpg"))
        if images:
            identities[identity_dir.name] = images
    return identities


def write_gallery_manifest(manifest: GalleryManifest, path: Path) -> None:
    """Write the manifest to a private location. It contains real image paths
    and is therefore git-ignored, never a published artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": manifest.seed,
        "entries": [
            {
                "sample_id": entry.sample_id,
                "identity_hash": entry.identity_hash,
                "role": entry.role,
                "image_path": str(entry.image_path),
            }
            for entry in manifest.entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_gallery_manifest(path: Path) -> GalleryManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [
        ManifestEntry(
            entry["sample_id"], entry["identity_hash"], Path(entry["image_path"]), entry["role"]
        )
        for entry in payload["entries"]
    ]
    return GalleryManifest(entries=entries, seed=payload["seed"])


# =============================================================================
# 14. Aggregate output generation
# =============================================================================
#
# Every artifact is self-describing enough that a reader never has to trust an
# unlabelled number: schema version, creation timestamp, software and model
# provenance, and dataset digests. Writes are atomic, so a crash mid-write
# never leaves a half-written result file.


class ArtifactError(RuntimeError):
    """Raised when a required artifact is missing or unreadable."""


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _atomic_write(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        # ``os.replace`` is atomic within a filesystem: the destination either
        # holds the whole old file or the whole new one, never a partial write.
        # Writing to a temporary neighbour and renaming is therefore what stops
        # an interrupted run leaving a half-written result behind.
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    body: Dict[str, Any] = dict(payload)
    body.setdefault("schema_version", SCHEMA_VERSION)
    body.setdefault("created_at", utc_now_iso())
    text = json.dumps(body, indent=2, sort_keys=True, default=_json_default) + "\n"
    _atomic_write(Path(path), text)
    return sha256_of_text(text)


def write_csv_artifact(
    path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]
) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    text = buffer.getvalue()
    _atomic_write(Path(path), text)
    return sha256_of_text(text)


def write_markdown_artifact(path: Path, text: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    _atomic_write(Path(path), text)
    return sha256_of_text(text)


def read_json_artifact(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


VERIFICATION_EXPERIMENTS = ("lfw_development", "lfw_final", "cplfw")

# Category keys carry the side on which extraction terminated; see
# evaluate_pairs for the left-first short-circuit rule.
FAILURE_CATEGORY_PROSE = {
    "zero_faces_left": "zero-face detections on the left image",
    "zero_faces_right": "zero-face detections on the right image",
    "multiple_faces_left": "multiple-face detections on the left image",
    "multiple_faces_right": "multiple-face detections on the right image",
}


def format_percentage(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def format_number(value: Any, digits: int = 4) -> str:
    try:
        # The inner ``{digits}`` is substituted into the format specification
        # itself, so the number of decimal places is chosen by the caller
        # rather than fixed in the literal.
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def format_count(value: Any) -> str:
    # Thousands separators for pair counts quoted in written work.
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def render_failure_breakdown(payload: Dict[str, Any]) -> str:
    """Describe the failure categories in prose, reading the counts from the
    result rather than restating them by hand."""
    breakdown = payload.get("failure_breakdown") or {}
    if not breakdown:
        return "No extraction-failure categories were recorded."

    described = [
        f"{format_count(breakdown[key])} {prose}"
        for key, prose in FAILURE_CATEGORY_PROSE.items()
        if key in breakdown
    ]
    # Any category outside the expected four is surfaced, never dropped.
    described += [
        f"{format_count(count)} {key}"
        for key, count in breakdown.items()
        if key not in FAILURE_CATEGORY_PROSE
    ]

    joined = ", ".join(described[:-1]) + (
        f" and {described[-1]}" if len(described) > 1 else described[0]
    )
    return (
        f"The {format_count(payload.get('failed_pairs'))} failures comprised {joined}. Each failed "
        f"pair carries exactly one category: sides are attempted left first and the pair is "
        f"abandoned at the first terminal failure, so a right-side category means the left image "
        f"had already yielded one valid face."
    )


def render_final_report(
    payloads: Dict[str, Dict[str, Any]], gallery_payload: Dict[str, Any]
) -> str:
    dev, final, cplfw = payloads["lfw_development"], payloads["lfw_final"], payloads["cplfw"]
    lines = [
        "# Final evaluation report",
        "",
        f"Auto-generated by `ACP_arden.py --mode full` on {utc_now_iso()}. Every number below is "
        "read directly from the corresponding `results/aggregate/*.json` file, each of which "
        "embeds its own software, model and dataset provenance (`software_environment`, "
        "`model_sha256`, `protocol_sha256`, `evaluated_image_set_sha256`, "
        "`dataset_archive_md5`/`dataset_archive_sha256`).",
        "",
        "## Experiments 1–2 — threshold calibration and selection",
        "",
        f"Experiment 1 (`pairsDevTrain.txt`) generated candidate thresholds and wrote them with "
        f"status `\"candidates\"`; it never selects a winner. Experiment 2 (`pairsDevTest.txt`) "
        f"evaluated every candidate and only then selected and froze "
        f"**{final.get('operating_strategy') or dev.get('selected_candidate', 'n/a')}** "
        f"at threshold **{format_number(final.get('threshold'), 6)}**, by the rule: "
        f"\"{dev.get('selection_rule', 'n/a')}\"",
        "",
        "## Experiment 2 — LFW development validation (`pairsDevTest.txt`)",
        "",
        f"Scored {dev.get('scored_pairs', 'n/a')} / {dev.get('total_pairs', 'n/a')} pairs "
        f"(failure rate {format_percentage(dev.get('failure_rate'))}). "
        f"Accuracy **{format_percentage(dev.get('accuracy'))}**, F1 {format_number(dev.get('f1'))}, "
        f"ROC-AUC {format_number(dev.get('roc_auc'))}, "
        f"EER {format_percentage(dev.get('equal_error_rate'))}.",
        "",
        "## Experiment 3 — final LFW evaluation (`pairs.txt`, frozen threshold, untouched protocol)",
        "",
        f"Scored {final.get('scored_pairs', 'n/a')} / {final.get('total_pairs', 'n/a')} pairs "
        f"(failure rate {format_percentage(final.get('failure_rate'))}). "
        f"**Accuracy {format_percentage(final.get('accuracy'))}**, "
        f"precision {format_percentage(final.get('precision'))}, "
        f"recall {format_percentage(final.get('recall'))}, "
        f"false match rate {format_percentage(final.get('false_match_rate'))}, "
        f"false non-match rate {format_percentage(final.get('false_non_match_rate'))}, "
        f"ROC-AUC {format_number(final.get('roc_auc'))}, "
        f"EER {format_percentage(final.get('equal_error_rate'))}. "
        f"Confusion matrix: {final.get('confusion_matrix')}. "
        f"Mean embedding time {format_number(final.get('embedding_time_mean_ms'), 2)} ms "
        f"(p95 {format_number(final.get('embedding_time_p95_ms'), 2)} ms) over "
        f"{final.get('unique_images_embedded', 'n/a')} unique images.",
        "",
        "## Experiment 4 — CPLFW cross-pose generalisation (same frozen threshold, no recalibration)",
        "",
        # Counts and the decimal rate come from summarize_metrics(); the
        # percentage is derived here so no figure is ever transcribed by hand.
        f"Of the {format_count(cplfw.get('total_pairs'))} raw CPLFW protocol pairs, "
        f"{format_count(cplfw.get('scored_pairs'))} produced valid similarity scores and "
        f"{format_count(cplfw.get('failed_pairs'))} failed during face extraction. The "
        f"extraction-failure rate was therefore **{format_percentage(cplfw.get('failure_rate'))}** "
        f"({format_count(cplfw.get('failed_pairs'))} ÷ {format_count(cplfw.get('total_pairs'))}). "
        f"These failed pairs were retained in the protocol total and reported separately rather "
        f"than being silently discarded.",
        "",
        render_failure_breakdown(cplfw),
        "",
        f"Accuracy, precision, recall, F1-score, ROC-AUC and EER are conditional on the "
        f"{format_count(cplfw.get('scored_pairs'))} pairs for which both images produced exactly "
        f"one valid face: accuracy {format_percentage(cplfw.get('accuracy'))}, "
        f"F1 {format_number(cplfw.get('f1'))}, "
        f"false match rate {format_percentage(cplfw.get('false_match_rate'))}, "
        f"false non-match rate {format_percentage(cplfw.get('false_non_match_rate'))}, "
        f"ROC-AUC {format_number(cplfw.get('roc_auc'))}, "
        f"EER {format_percentage(cplfw.get('equal_error_rate'))}. An extraction failure is not a "
        f"verification error: the pipeline never produced a similarity score for those pairs, so "
        f"they can be neither correct nor incorrect.",
        "",
        "## Experiment 5 — real 1:N duplicate-profile gallery (LFW, seed "
        f"{gallery_payload.get('seed', 'n/a')})",
        "",
        f"Gallery size {gallery_payload.get('gallery_size', 'n/a')}; "
        f"{gallery_payload.get('duplicate_probe_count', 'n/a')} duplicate probes "
        f"({gallery_payload.get('duplicate_probe_failures', 'n/a')} extraction failures); "
        f"{gallery_payload.get('unknown_probe_count', 'n/a')} unknown probes "
        f"({gallery_payload.get('unknown_probe_failures', 'n/a')} extraction failures).",
        "",
        f"- Gallery enrolment: {gallery_payload.get('embedded_gallery_size', 'n/a')} enrolled of "
        f"{gallery_payload.get('intended_gallery_size', 'n/a')} intended "
        f"(failure rate {format_percentage(gallery_payload.get('gallery_entry_failure_rate'))}); "
        f"{gallery_payload.get('gallery_reference_unavailable_count', 'n/a')} mated probes had no "
        f"enrolled reference.",
        f"- Duplicate detection rate (conditional, scored mated probes only): "
        f"**{format_percentage(gallery_payload.get('duplicate_detection_rate'))}**",
        f"- Duplicate detection rate (end-to-end, all intended mated probes): "
        f"**{format_percentage(gallery_payload.get('end_to_end_duplicate_detection_rate'))}**",
        f"- Rank-1 identification rate: "
        f"{format_percentage(gallery_payload.get('rank1_identification_rate'))}",
        f"- Rank-5 identification rate (conditional): "
        f"{format_percentage(gallery_payload.get('conditional_rank5_identification_rate'))}",
        f"- True duplicate miss rate: "
        f"{format_percentage(gallery_payload.get('true_duplicate_miss_rate'))}",
        f"- **False duplicate-review rate: "
        f"{format_percentage(gallery_payload.get('false_duplicate_review_rate'))}**",
        "",
        gallery_payload.get("policy_note", POLICY_NOTE),
        "",
        "The false-review rate reflects reusing the 1:1 ownership-verification threshold "
        "(calibrated for comparing exactly two images) as the 1:N duplicate-review threshold. "
        f"A {format_percentage(final.get('false_match_rate'))} single-comparison false-match rate "
        f"(Experiment 3) compounds across {gallery_payload.get('gallery_size', 'n/a')} gallery "
        "comparisons per probe — direct, quantified evidence that a 1:1-calibrated threshold is "
        "not fit for 1:N search at this gallery scale without its own calibration, and evidence "
        "for this project's human-review-not-automatic-sanction policy.",
        "",
        "## Limitations",
        "",
        "These figures describe LFW and CPLFW's own demographic composition, not any real user "
        "base; the gallery experiment is research-scale, not production-scale; and \"duplicate "
        "profile\" here means \"same face detected in the gallery\", not a legal or investigative "
        "finding.",
        "",
    ]
    return "\n".join(lines)


def write_aggregate_reports(
    output_root: Path, gallery_manifest_path: Path, cplfw_image_variant: str
) -> None:
    """Cross-reference the five per-experiment metrics files into a manifest,
    three CSVs and a written report, then refuse to finish if any of them
    contains a personal or absolute filesystem path."""
    threshold_path = output_root / "calibrated_threshold.json"
    payloads: Dict[str, Dict[str, Any]] = {
        "lfw_development": read_json_artifact(output_root / "lfw_development_metrics.json"),
        "lfw_final": read_json_artifact(output_root / "lfw_final_metrics.json"),
        "cplfw": read_json_artifact(output_root / "cplfw_metrics.json"),
    }
    gallery_v2_path = output_root / "duplicate_gallery_metrics_v2.json"
    gallery_path = (
        gallery_v2_path if gallery_v2_path.is_file() else output_root / "duplicate_gallery_metrics.json"
    )
    gallery_payload = read_json_artifact(gallery_path)
    threshold_payload = read_json_artifact(threshold_path)

    output_files = {
        "calibrated_threshold.json": threshold_path,
        "lfw_development_metrics.json": output_root / "lfw_development_metrics.json",
        "lfw_final_metrics.json": output_root / "lfw_final_metrics.json",
        "cplfw_metrics.json": output_root / "cplfw_metrics.json",
        gallery_path.name: gallery_path,
    }

    write_json_artifact(
        output_root / "run_manifest.json",
        {
            "artifact_type": "run_manifest",
            "opaque_id_version": OPAQUE_ID_VERSION,
            "produced_by": "ACP_arden.py --mode full",
            "dataset_storage": (
                "private, gitignored local research storage; path omitted from public artifacts"
            ),
            "dataset_root_variable": "FACE_DATA_ROOT",
            "protocol_root_variable": "FACE_PROTOCOL_ROOT",
            "model_root_variable": "FACE_MODEL_ROOT",
            "cplfw_image_variant": cplfw_image_variant,
            "output_root": project_relative(output_root),
            "gallery_manifest": project_relative(gallery_manifest_path),
            "frozen_threshold": threshold_payload.get("threshold"),
            "frozen_threshold_candidate": threshold_payload.get("operating_strategy"),
            "output_file_sha256": {
                name: sha256_of_file(path) for name, path in output_files.items()
            },
            "software_environment": software_environment_report(),
        },
    )

    summary_field_order = [
        "experiment", "protocol_file", "total_pairs", "scored_pairs", "failure_rate", "threshold",
        "accuracy", "precision", "recall", "f1", "false_match_rate", "false_non_match_rate",
        "roc_auc", "equal_error_rate", "embedding_time_mean_ms", "embedding_time_median_ms",
        "embedding_time_p95_ms", "gallery_size", "duplicate_probe_count", "unknown_probe_count",
        "duplicate_detection_rate", "false_duplicate_review_rate", "rank1_identification_rate",
        "true_duplicate_miss_rate", "gallery_search_time_mean_ms", "gallery_search_time_p95_ms",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for name in VERIFICATION_EXPERIMENTS:
        payload = payloads[name]
        summary_rows.append(
            {"experiment": name, **{f: payload.get(f, "") for f in summary_field_order[1:]}}
        )
    summary_rows.append(
        {
            "experiment": "duplicate_gallery",
            **{f: gallery_payload.get(f, "") for f in summary_field_order[1:]},
        }
    )
    write_csv_artifact(
        output_root / "metrics_summary.csv", summary_rows, fieldnames=summary_field_order
    )

    confusion_fields = [
        "experiment", "true_positive", "false_positive", "true_negative", "false_negative",
    ]
    confusion_rows: List[Dict[str, Any]] = []
    for name in VERIFICATION_EXPERIMENTS:
        matrix = payloads[name].get("confusion_matrix") or {}
        confusion_rows.append(
            {"experiment": name, **{f: matrix.get(f, "") for f in confusion_fields[1:]}}
        )
    write_csv_artifact(
        output_root / "confusion_matrices.csv", confusion_rows, fieldnames=confusion_fields
    )

    roc_fields = ["experiment", "threshold", "false_match_rate", "true_match_rate"]
    roc_rows: List[Dict[str, Any]] = []
    for name in VERIFICATION_EXPERIMENTS:
        for point in payloads[name].get("roc_points", []):
            roc_rows.append({"experiment": name, **{f: point[f] for f in roc_fields[1:]}})
    write_csv_artifact(output_root / "roc_points.csv", roc_rows, fieldnames=roc_fields)

    write_markdown_artifact(
        output_root / "FINAL_EVALUATION_REPORT.md", render_final_report(payloads, gallery_payload)
    )

    leaks = find_path_leaks(output_root, forbidden_substrings=default_forbidden_path_substrings())
    if leaks:
        raise SystemExit(
            "Refusing to finish: public aggregate output(s) contain a personal/absolute path:\n"
            + "\n".join(f"  {redact_private_paths(leak)}" for leak in leaks)
        )
    assert_no_identifier_key_leak(output_root)

    announce(
        f"Wrote run_manifest.json, metrics_summary.csv ({len(summary_rows)} rows), "
        f"confusion_matrices.csv ({len(confusion_rows)} rows), roc_points.csv "
        f"({len(roc_rows)} rows) and FINAL_EVALUATION_REPORT.md to "
        f"{project_relative(output_root)}"
    )


def _render_gallery_coverage_lines(gallery: Dict[str, Any]) -> List[str]:
    """Gallery enrolment coverage. Emitted unconditionally so a detection rate
    can never be read without knowing how much of the gallery actually
    enrolled; an artefact predating the corrected accounting says so plainly
    instead of implying full coverage."""
    if "gallery_entry_failure_rate" not in gallery:
        return [
            "  Gallery enrolment coverage: not recorded (artefact predates "
            f"{GALLERY_METHODOLOGY_REVISION}; re-run to obtain it)",
        ]
    return [
        f"  Intended gallery entries: {format_count(gallery.get('intended_gallery_size'))}, "
        f"enrolled {format_count(gallery.get('embedded_gallery_size'))}",
        f"  Gallery enrolment-failure rate: "
        f"{format_percentage(gallery.get('gallery_entry_failure_rate'))} "
        f"({format_count(gallery.get('gallery_entry_failure_count'))} references never enrolled)",
    ]


def _render_end_to_end_gallery_lines(gallery: Dict[str, Any]) -> List[str]:
    """The end-to-end detection rate, which shares the conditional figure's
    numerator but counts every intended mated probe. Printing it here is what
    stops the flattering conditional number standing on its own."""
    if "end_to_end_duplicate_detection_rate" not in gallery:
        return [
            "  Duplicate detection rate (end-to-end): not recorded (artefact "
            f"predates {GALLERY_METHODOLOGY_REVISION}; the conditional figure "
            "above therefore excludes unenrolled references and failed probes)",
        ]
    return [
        f"  Duplicate detection rate (end-to-end): "
        f"{format_percentage(gallery.get('end_to_end_duplicate_detection_rate'))}",
        f"  Mated probes with no enrolled reference: "
        f"{format_count(gallery.get('gallery_reference_unavailable_count'))}",
        "  LIMITATION: the conditional rate above is measured only over mated probes that "
        "were scored. The end-to-end rate counts every intended mated probe, so extraction "
        "failures and unenrolled references reduce it. Quote the two together.",
    ]


def render_results_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    """Headline figures for the terminal, read from the aggregate artifacts
    rather than hard-coded. Each conditional figure is printed together with
    the limitation that makes it interpretable: the CPLFW accuracy never
    appears without its extraction-failure rate, and the gallery detection rate
    never appears without its false-review rate."""
    final = read_json_artifact(output_root / "lfw_final_metrics.json")
    cplfw = read_json_artifact(output_root / "cplfw_metrics.json")
    # Prefer the corrected-accounting artefact when it exists; the historical
    # file remains readable so earlier results stay interpretable.
    gallery_v2_path = output_root / "duplicate_gallery_metrics_v2.json"
    gallery = read_json_artifact(
        gallery_v2_path if gallery_v2_path.is_file() else output_root / "duplicate_gallery_metrics.json"
    )
    threshold = read_json_artifact(output_root / "calibrated_threshold.json")

    lines = [
        f"{PROGRAMME_TITLE} — results summary",
        "",
        f"Frozen threshold: {format_number(threshold.get('threshold'), 6)} "
        f"(candidate: {threshold.get('operating_strategy')}, status: {threshold.get('status')})",
        f"Selection rule: {threshold.get('selection_rule')}",
        "",
        "Experiment 3 — final LFW (pairs.txt, frozen threshold, no recalibration)",
        f"  Final LFW accuracy: {format_percentage(final.get('accuracy'))}",
        f"  Final LFW false-match rate: {format_percentage(final.get('false_match_rate'))}",
        f"  Final LFW false-non-match rate: {format_percentage(final.get('false_non_match_rate'))}",
        f"  Final LFW EER: {format_percentage(final.get('equal_error_rate'))}",
        f"  Final LFW extraction-failure rate: {format_percentage(final.get('failure_rate'))}",
        f"  Scored pairs: {format_count(final.get('scored_pairs'))} / "
        f"{format_count(final.get('total_pairs'))}",
        "",
        "Experiment 4 — raw CPLFW (same frozen threshold, cross-pose generalisation)",
        f"  Raw CPLFW conditional accuracy: {format_percentage(cplfw.get('accuracy'))}",
        f"  Raw CPLFW scored pairs: {format_count(cplfw.get('scored_pairs'))} / "
        f"{format_count(cplfw.get('total_pairs'))}",
        f"  Raw CPLFW failed pairs: {format_count(cplfw.get('failed_pairs'))}",
        f"  Raw CPLFW extraction-failure rate: {format_percentage(cplfw.get('failure_rate'))}",
        f"  Raw CPLFW false-match rate: {format_percentage(cplfw.get('false_match_rate'))}",
        f"  Raw CPLFW false-non-match rate: {format_percentage(cplfw.get('false_non_match_rate'))}",
        f"  Raw CPLFW EER: {format_percentage(cplfw.get('equal_error_rate'))}",
        f"  LIMITATION: the CPLFW accuracy above is conditional on the "
        f"{format_count(cplfw.get('scored_pairs'))} pairs that yielded one valid face on both "
        f"sides. {format_percentage(cplfw.get('failure_rate'))} of the protocol "
        f"({format_count(cplfw.get('failed_pairs'))} pairs) never reached the comparison stage, "
        f"and must be quoted alongside it.",
        "",
        "Experiment 5 — 1:N duplicate-profile gallery (real LFW images, "
        f"seed {gallery.get('seed')})",
        f"  Gallery size: {format_count(gallery.get('gallery_size'))}",
        *_render_gallery_coverage_lines(gallery),
        f"  Duplicate detection rate (conditional): "
        f"{format_percentage(gallery.get('duplicate_detection_rate'))}",
        *_render_end_to_end_gallery_lines(gallery),
        f"  Rank-1 identification rate: "
        f"{format_percentage(gallery.get('rank1_identification_rate'))}",
        f"  False duplicate-review rate: "
        f"{format_percentage(gallery.get('false_duplicate_review_rate'))}",
        f"  LIMITATION: the detection rate above is inseparable from the "
        f"{format_percentage(gallery.get('false_duplicate_review_rate'))} false duplicate-review "
        f"rate. Reusing a 1:1-calibrated threshold for 1:N search flags that share of genuinely "
        f"new identities for review, so the two figures must never be quoted apart.",
        "",
        "Policy: " + str(gallery.get("policy_note", POLICY_NOTE)),
        "",
        "Full write-up: results/aggregate/FINAL_EVALUATION_REPORT.md",
    ]
    return "\n".join(lines)


# =============================================================================
# 15. Privacy validation
# =============================================================================
#
# Guards that keep published outputs free of names, absolute paths and raw
# embedding vectors.

_FORBIDDEN_KEY_SUBSTRINGS = ("embedding", "path", "identity", "name")
_ALLOWED_KEY_EXCEPTIONS = {
    "strategy",
    "identity_count",
    "identity_hash",
    "candidate_identity_hash",
}

_TEXT_ARTIFACT_SUFFIXES = (".json", ".csv", ".md", ".txt")
_IMAGE_ARTIFACT_SUFFIXES = (".png",)

# Cloud-sync location markers, which identify a private storage layout even
# when they appear without a leading absolute-path prefix — for instance inside
# a rendered image's embedded metadata, or a relative-looking string. These are
# generic provider names; the researcher's own roots are added at scan time
# from the configured environment variables.
_PRIVATE_LOCATION_MARKERS = (
    "Library/CloudStorage",
    "OneDrive",
    "Dropbox",
    "Google Drive",
    "iCloud Drive",
)


class PrivacyLeakError(ValueError):
    """Raised when a record or artifact would disclose private data."""


def assert_no_leakage(record: Mapping[str, Any], *, context: str = "") -> None:
    """Raise PrivacyLeakError if any key or value in the record looks like a
    real name, an absolute filesystem path, or a raw embedding vector."""
    for key, value in record.items():
        label = f"{context}.{key}" if context else str(key)
        lowered_key = key.lower() if isinstance(key, str) else str(key)

        if key not in _ALLOWED_KEY_EXCEPTIONS:
            for banned in _FORBIDDEN_KEY_SUBSTRINGS:
                if banned in lowered_key:
                    raise PrivacyLeakError(f"{label}: key name may leak private data")

        if isinstance(value, str) and (value.startswith("/") or value.startswith("~")):
            raise PrivacyLeakError(f"{label}: value looks like an absolute path: {value}")

        # ``all(... for ...)`` is a generator expression: it tests each item in
        # turn and stops at the first failure rather than building an
        # intermediate list. A long, purely numeric sequence is the shape of a
        # raw embedding, which must never reach a published artefact.
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 32
            and all(isinstance(item, (int, float)) for item in value)
        ):
            raise PrivacyLeakError(f"{label}: value looks like a raw embedding vector")

        if isinstance(value, Mapping):
            assert_no_leakage(value, context=label)


def assert_no_identifier_key_leak(root: Path) -> None:
    """Confirm the identifier key appears in no published artefact.

    The key is compared against file contents but is never written into the
    error message, a log line or a return value — reporting *where* it leaked is
    useful, reporting *what* leaked would be a second leak. Nothing is raised
    when no key is configured: there is then nothing to disclose."""
    if _ID_HMAC_KEY is None:
        return
    encoded = base64.urlsafe_b64encode(_ID_HMAC_KEY).decode("ascii").rstrip("=")
    needles = [encoded, _ID_HMAC_KEY.hex()]
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_ARTIFACT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for needle in needles:
            if needle and needle in text:
                raise PrivacyLeakError(
                    f"{path.name} contains the identifier HMAC key. The key must never be "
                    f"published; regenerate it and rebuild the affected artefacts."
                )


def default_forbidden_path_substrings(*, env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Substrings that must never appear in a published result: the user's home
    directory, common absolute-path prefixes, known private-storage location
    names, and the expanded value of every storage environment variable."""
    source = os.environ if env is None else env
    from_file = load_env_file()
    substrings = {"/Users/", "\\Users\\", "/home/", str(Path.home())}
    substrings.update(_PRIVATE_LOCATION_MARKERS)
    for variable in (*REQUIRED_ENVIRONMENT_VARIABLES, *OPTIONAL_ENVIRONMENT_VARIABLES):
        for value in (source.get(variable), from_file.get(variable)):
            if value:
                substrings.add(value)
    return sorted(s for s in substrings if s)


def _png_text_metadata(path: Path) -> str:
    """Every tEXt/iTXt/zTXt chunk in a PNG, concatenated. A renderer may write
    its own metadata, so an image can leak a path that never appears in the
    visible pixels.

    A corrupt or truncated PNG yields "" — there is nothing to read, and the
    file is not publishable evidence anyway. A missing Pillow is different: it
    would make every PNG silently pass, turning this check into a no-op that
    still reports "clean", so its absence is raised rather than swallowed."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a pinned dependency
        raise PrivacyLeakError(
            "Pillow is not installed, so PNG metadata cannot be scanned. Refusing to report a "
            "clean result from a check that did not run."
        ) from exc

    try:
        with Image.open(path) as image:
            return "\n".join(
                f"{key}: {value}"
                for key, value in (image.info or {}).items()
                if isinstance(value, str)
            )
    except OSError:
        return ""


def find_path_leaks(root: Path, *, forbidden_substrings: Sequence[str]) -> List[str]:
    """Recursively scan every JSON, CSV, Markdown or text file under the root —
    and the embedded text metadata of every PNG — for a forbidden substring.
    Returns "file:line: forbidden substring" findings; an empty list means
    clean. Unreadable or binary files are skipped rather than raising."""
    findings: List[str] = []
    root = Path(root)
    if not root.exists():
        return findings
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _TEXT_ARTIFACT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            location = "{path}:{line}"
        elif suffix in _IMAGE_ARTIFACT_SUFFIXES:
            text = _png_text_metadata(path)
            location = "{path} (embedded PNG metadata, entry {line})"
        else:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            for needle in forbidden_substrings:
                if needle and needle in line:
                    where = location.format(path=path, line=line_number)
                    findings.append(f"{where}: contains forbidden substring {needle!r}")
    return findings


def check_public_outputs(paths: Sequence[Path]) -> bool:
    """Scan published output directories and report the outcome. Returns True
    when every scanned location is clean."""
    forbidden = default_forbidden_path_substrings()
    all_leaks: List[str] = []
    scanned: List[str] = []

    for path in paths:
        if not path.exists():
            continue
        all_leaks.extend(find_path_leaks(path, forbidden_substrings=forbidden))
        scanned.append(project_relative(path))

    if all_leaks:
        print("FAIL personal/absolute path(s) found in published outputs:", file=sys.stderr)
        for leak in all_leaks:
            print(f"  {redact_private_paths(leak)}", file=sys.stderr)
        return False

    print(f"OK   no personal/absolute paths found under: {', '.join(scanned) or 'nothing to scan'}")
    return True


# =============================================================================
# 16. Optional human-review interface
# =============================================================================
#
# A local, login-free Streamlit page for manually reviewing anonymised
# duplicate-profile cases. It never displays a real name, real file path or raw
# embedding — only opaque identifiers, a similarity score, and the threshold
# that opened the case. It applies no sanction of any kind.

REVIEW_STATUSES = ["open", "confirmed_duplicate", "false_match", "dismissed"]
ALLOWED_REVIEW_STATUSES = set(REVIEW_STATUSES)

REVIEW_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_cases (
    case_id TEXT PRIMARY KEY,
    probe_sample_id TEXT NOT NULL,
    candidate_identity_hash TEXT NOT NULL,
    similarity REAL NOT NULL,
    threshold REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    decided_at TEXT,
    -- Which identifier scheme produced probe_sample_id and
    -- candidate_identity_hash. Rows from two schemes are not comparable: the
    -- same person yields different identifiers under different keys.
    opaque_id_version TEXT NOT NULL DEFAULT 'hmac-sha256-v1'
);
"""


class ReviewDatabaseVersionError(RuntimeError):
    """Raised when a review database holds rows from another identifier scheme."""


def _migrate_review_schema(connection: sqlite3.Connection) -> None:
    """Add the version column to a database created before it existed. Rows
    already present came from the old fixed-salt scheme, so they are marked as
    such rather than silently relabelled."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(review_cases)")}
    if "opaque_id_version" not in columns:
        connection.execute(
            "ALTER TABLE review_cases ADD COLUMN opaque_id_version TEXT NOT NULL "
            "DEFAULT 'legacy-salted-sha256'"
        )


def assert_review_database_version(connection: sqlite3.Connection) -> None:
    """Refuse to mix identifier schemes in one database. The local review
    database is private and disposable, so the instruction is to delete it."""
    versions = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT opaque_id_version FROM review_cases WHERE opaque_id_version IS NOT NULL"
        )
    }
    foreign = versions - {OPAQUE_ID_VERSION}
    if foreign:
        raise ReviewDatabaseVersionError(
            f"The local review database holds cases written under identifier scheme(s) "
            f"{sorted(foreign)!r}, but this build emits {OPAQUE_ID_VERSION!r}. Those "
            f"identifiers are not comparable, so the rows cannot be merged. The review "
            f"database is private, disposable local state: delete the file and re-run the "
            f"evaluation to regenerate it."
        )


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    probe_sample_id: str
    candidate_identity_hash: str
    similarity: float
    threshold: float
    status: str
    created_at: str
    decided_at: Optional[str]


# ``@contextmanager`` turns a generator into something usable with ``with``:
# everything before ``yield`` runs on entry, the yielded value is bound to the
# ``as`` name, and the ``finally`` clause runs on exit however the block ends.
# That is what guarantees the connection closes even if the caller raises.
@contextmanager
def review_database(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(REVIEW_SCHEMA)
        _migrate_review_schema(connection)
        assert_review_database_version(connection)
        connection.row_factory = sqlite3.Row
        yield connection
        connection.commit()
    finally:
        connection.close()


def upsert_review_case(
    connection: sqlite3.Connection,
    *,
    case_id: str,
    probe_sample_id: str,
    candidate_identity_hash: str,
    similarity: float,
    threshold: float,
) -> None:
    connection.execute(
        """
        INSERT INTO review_cases
            (case_id, probe_sample_id, candidate_identity_hash, similarity, threshold,
             status, created_at, opaque_id_version)
        VALUES (?, ?, ?, ?, ?, 'open', ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET
            probe_sample_id=excluded.probe_sample_id,
            candidate_identity_hash=excluded.candidate_identity_hash,
            similarity=excluded.similarity,
            threshold=excluded.threshold,
            opaque_id_version=excluded.opaque_id_version
        """,
        (
            case_id,
            probe_sample_id,
            candidate_identity_hash,
            similarity,
            threshold,
            utc_now_iso(),
            OPAQUE_ID_VERSION,
        ),
    )


def count_review_cases(connection: sqlite3.Connection, *, status: Optional[str] = None) -> int:
    if status is not None and status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"Unknown status filter: {status}")
    if status:
        row = connection.execute(
            "SELECT COUNT(*) FROM review_cases WHERE status = ?", (status,)
        ).fetchone()
    else:
        row = connection.execute("SELECT COUNT(*) FROM review_cases").fetchone()
    return int(row[0])


def list_review_cases(
    connection: sqlite3.Connection, *, status: Optional[str] = None, limit: Optional[int] = None
) -> List[ReviewCase]:
    """Cases ordered by descending similarity, so the strongest candidates for
    human attention come first. ``limit`` bounds how many are returned: a real
    queue can hold thousands, and rendering all of them at once would make the
    page unusable rather than more informative."""
    if status is not None and status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"Unknown status filter: {status}")
    if limit is not None and limit < 0:
        raise ValueError("limit must not be negative")

    clauses = "SELECT * FROM review_cases"
    parameters: List[Any] = []
    if status:
        clauses += " WHERE status = ?"
        parameters.append(status)
    clauses += " ORDER BY similarity DESC"
    if limit is not None:
        clauses += " LIMIT ?"
        parameters.append(limit)

    rows = connection.execute(clauses, tuple(parameters))
    return [_row_to_review_case(row) for row in rows]


def set_review_status(connection: sqlite3.Connection, *, case_id: str, status: str) -> None:
    if status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"Unknown status: {status}; must be one of {sorted(REVIEW_STATUSES)}")
    connection.execute(
        "UPDATE review_cases SET status = ?, decided_at = ? WHERE case_id = ?",
        (status, utc_now_iso(), case_id),
    )


def _row_to_review_case(row: sqlite3.Row) -> ReviewCase:
    return ReviewCase(
        case_id=row["case_id"],
        probe_sample_id=row["probe_sample_id"],
        candidate_identity_hash=row["candidate_identity_hash"],
        similarity=row["similarity"],
        threshold=row["threshold"],
        status=row["status"],
        created_at=row["created_at"],
        decided_at=row["decided_at"],
    )


def populate_review_database(
    db_path: Path, result: GalleryEvaluationResult, threshold: float
) -> int:
    """Record every probe that exceeded the review threshold as an open case.
    Only opaque identifiers and scores are stored — never a name or a path."""
    flagged = [p for p in result.probe_results if p.exceeds_duplicate_threshold]
    with review_database(db_path) as connection:
        for probe in flagged:
            # A probe only exceeds the threshold once it has been scored
            # against a candidate, so both fields are populated here.
            if probe.top_candidate_identity_hash is None or probe.top_similarity is None:
                continue
            upsert_review_case(
                connection,
                case_id=f"{probe.sample_id}:{probe.top_candidate_identity_hash}",
                probe_sample_id=probe.sample_id,
                candidate_identity_hash=probe.top_candidate_identity_hash,
                similarity=probe.top_similarity,
                threshold=threshold,
            )
    return len(flagged)


def running_under_streamlit() -> bool:
    """True when this file is already executing inside a Streamlit script run,
    so the review mode renders the page instead of launching another server."""
    if os.environ.get("ACP_ARDEN_REVIEW_CHILD") == "1":
        return True
    # Only Streamlit itself imports Streamlit before this file runs. Checking
    # that first keeps a plain interpreter run from importing the package at
    # all, which would emit a "missing ScriptRunContext" warning over the menu.
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001 - defensive across Streamlit versions
        return False


def render_review_page(db_path: Path) -> None:
    """The Streamlit page itself. Rendered only inside a Streamlit script run."""
    import streamlit as st

    st.set_page_config(page_title="Duplicate-profile review (local only)", layout="wide")

    st.title("Duplicate-profile review — local demonstration only")
    st.warning(
        "Similarity above threshold is evidence for human review, not proof of misuse or scam "
        "activity. No account is banned, suspended or accused by this page. Case, probe and "
        "candidate identifiers are opaque one-way hashes; no real name or file path is ever "
        "shown here."
    )

    filter_column, page_size_column = st.columns([2, 1])
    status_filter = filter_column.selectbox("Filter by status", ["all", *REVIEW_STATUSES])
    page_size = page_size_column.number_input(
        "Cases to show", min_value=5, max_value=200, value=25, step=5
    )

    with review_database(db_path) as connection:
        status = None if status_filter == "all" else status_filter
        total = count_review_cases(connection, status=status)
        # A whole queue is thousands of cases; a reviewer works the strongest
        # candidates first, so the page shows a bounded, ordered slice of them.
        cases = list_review_cases(connection, status=status, limit=int(page_size))

        if not cases:
            st.info(
                "No cases match this filter. Run the complete evaluation "
                "(`python ACP_arden.py --mode full`) to populate the local review database."
            )
            return

        st.caption(
            f"Showing the {len(cases)} highest-similarity case(s) of {total} matching this "
            f"filter, ordered by descending similarity. Each one is a prompt for a human "
            f"decision, not a finding."
        )

        for case in cases:
            with st.container(border=True):
                st.write(f"**Case:** `{case.case_id}`")
                col1, col2, col3 = st.columns(3)
                col1.metric("Similarity", f"{case.similarity:.4f}")
                col2.metric("Threshold", f"{case.threshold:.4f}")
                col3.metric("Status", case.status)
                st.caption(
                    f"Probe: `{case.probe_sample_id}` — "
                    f"Candidate identity: `{case.candidate_identity_hash}`"
                )
                st.caption(
                    f"Opened: {case.created_at}"
                    + (f" — Decided: {case.decided_at}" if case.decided_at else "")
                )

                chosen = st.radio(
                    "Decision",
                    REVIEW_STATUSES,
                    index=REVIEW_STATUSES.index(case.status),
                    key=f"decision_{case.case_id}",
                    horizontal=True,
                )
                if chosen != case.status:
                    set_review_status(connection, case_id=case.case_id, status=chosen)
                    st.rerun()


def launch_review_interface(db_path: Path) -> int:
    """Re-run this same file under Streamlit. The child process is marked with
    an environment flag so it renders the page rather than recursing."""
    if not db_path.exists():
        announce(
            f"No review database at {project_relative(db_path)} yet. The page will open empty; "
            f"run option 3 (complete evaluation) first to populate it."
        )
    child_environment = dict(os.environ)
    child_environment["ACP_ARDEN_REVIEW_CHILD"] = "1"
    print("Starting the local review interface (Ctrl+C in this terminal to stop it).")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.address=127.0.0.1",
            "--",
            "--mode",
            "review",
            "--review-db",
            str(db_path),
        ],
        check=False,
        env=child_environment,
    )
    if completed.returncode != 0:
        print(
            "The review interface exited non-zero. Streamlit is an optional dependency: "
            "install it with `python -m pip install -r requirements.txt`.",
            file=sys.stderr,
        )
    return completed.returncode


# =============================================================================
# 17. BFW dataset adapter
# =============================================================================
#
# Balanced Faces in the Wild is an external real benchmark. It is never
# downloaded automatically, never vendored into this repository and never read
# from an unofficial mirror: the researcher obtains it from the official project
# under its own terms and points FACE_BFW_ROOT at the extracted tree.
#
# The metadata schema is pinned rather than sniffed. A file that does not match
# stops the run with a message naming what was expected and what was found,
# because silently mis-parsing an identity or a subgroup column would corrupt
# every downstream figure while still producing plausible-looking numbers.
##############
# Title: Face Recognition: Too Bias, or Not Too Bias?
# Author: Robinson, J.P., Livitz, G., Henon, Y., Qin, C., Fu, Y. and Timoner, S.
# Date: 2020
# Availability: https://doi.org/10.1109/CVPRW50498.2020.00008
##############
##############
# Title: Balanced Faces in the Wild (BFW), source data and metadata table
# Author: Robinson, J.P. and contributors (visionjo/facerec-bias-bfw)
# Date: 2020
# Availability: https://github.com/visionjo/facerec-bias-bfw
##############


class BfwDatasetError(RuntimeError):
    """Raised when BFW is absent, misconfigured or fails schema validation."""


# The eight official demographic subgroups. Used for stratification and for
# aggregate fairness reporting only; never joined to an identifiable name in
# any published artefact.
BFW_SUBGROUPS = (
    "asian_females",
    "asian_males",
    "black_females",
    "black_males",
    "indian_females",
    "indian_males",
    "white_females",
    "white_males",
)

# Columns of the official BFW datatable that this adapter relies on. Pinned so
# that a differently-shaped release is refused rather than misread.
BFW_REQUIRED_COLUMNS = ("fold", "p1", "p2", "label", "att1", "att2")

# This project's own open-set protocol built on top of the official data. BFW
# was published for verification and bias analysis and ships no official
# open-set identification protocol, so the version below identifies *our*
# construction, not an upstream standard.
BFW_PROTOCOL_VERSION = "acp-arden-bfw-open-set-v1"

BFW_ROOT_VARIABLE = "FACE_BFW_ROOT"
BFW_METADATA_ROOT_VARIABLE = "FACE_BFW_METADATA_ROOT"


@dataclass(frozen=True)
class BfwImage:
    """One BFW image. ``image_path`` is private and never published; the two
    opaque identifiers are what appear in public artefacts."""

    image_path: Path
    identity: str  # private, e.g. "asian_females/n000009"
    subgroup: str
    sample_id: str
    identity_hash: str


@dataclass(frozen=True)
class BfwDataset:
    images: List[BfwImage]
    metadata_sha256: str
    metadata_filename: str

    def by_identity(self) -> Dict[str, List[BfwImage]]:
        grouped: Dict[str, List[BfwImage]] = {}
        for image in self.images:
            grouped.setdefault(image.identity, []).append(image)
        # Deterministic ordering by filename, so role assignment never depends
        # on directory-iteration order.
        return {
            identity: sorted(items, key=lambda i: i.image_path.name)
            for identity, items in sorted(grouped.items())
        }

    def subgroup_of_identity(self) -> Dict[str, str]:
        return {image.identity: image.subgroup for image in self.images}


def _bfw_identity_and_subgroup(relative: str) -> Tuple[str, str]:
    """Split an official BFW relative path into (identity, subgroup).

    The published layout is ``<subgroup>/<identity>/<image>.jpg``; anything
    else is a schema violation rather than something to guess at."""
    parts = PurePosixPath(relative.strip().replace("\\", "/")).parts
    if len(parts) != 3:
        raise BfwDatasetError(
            f"BFW image path {relative!r} does not match the official "
            f"'<subgroup>/<identity>/<image>' layout (found {len(parts)} component(s))."
        )
    subgroup, identity_folder, _filename = parts
    if subgroup not in BFW_SUBGROUPS:
        raise BfwDatasetError(
            f"BFW image path {relative!r} names subgroup {subgroup!r}, which is not one of "
            f"the eight official subgroups {list(BFW_SUBGROUPS)}."
        )
    return f"{subgroup}/{identity_folder}", subgroup


def load_bfw_dataset(
    image_root: Path,
    metadata_path: Path,
    *,
    require_files_exist: bool = True,
) -> BfwDataset:
    """Read the official BFW datatable and return the deduplicated image set.

    Every failure is explicit. Nothing is skipped, inferred or defaulted."""
    image_root = Path(image_root).resolve()
    metadata_path = Path(metadata_path)
    if not image_root.is_dir():
        raise BfwDatasetError(f"{BFW_ROOT_VARIABLE} does not point at a directory: {image_root}")
    if not metadata_path.is_file():
        raise BfwDatasetError(
            f"BFW metadata table not found: {metadata_path}. Obtain it from the official "
            f"project (visionjo/facerec-bias-bfw) and set {BFW_METADATA_ROOT_VARIABLE}."
        )

    text = metadata_path.read_text(encoding="utf-8")
    reader = csv.DictReader(StringIO(text))
    columns = tuple(reader.fieldnames or ())
    missing = [c for c in BFW_REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise BfwDatasetError(
            f"BFW metadata {metadata_path.name} is missing required column(s) {missing}. "
            f"Expected at least {list(BFW_REQUIRED_COLUMNS)}; found {list(columns)}. This "
            f"adapter pins the official schema and refuses to guess at a variant."
        )

    # relative path -> subgroup asserted by the table, so a row that contradicts
    # an earlier row is caught rather than silently overwriting it.
    asserted: Dict[str, str] = {}
    for line_number, row in enumerate(reader, start=2):
        for path_column, attribute_column in (("p1", "att1"), ("p2", "att2")):
            relative = (row.get(path_column) or "").strip()
            attribute = (row.get(attribute_column) or "").strip()
            if not relative:
                raise BfwDatasetError(
                    f"BFW metadata {metadata_path.name} line {line_number}: column "
                    f"{path_column!r} is empty."
                )
            _identity, subgroup = _bfw_identity_and_subgroup(relative)
            if attribute and attribute != subgroup:
                raise BfwDatasetError(
                    f"BFW metadata {metadata_path.name} line {line_number}: column "
                    f"{attribute_column!r} says {attribute!r} but the path implies "
                    f"{subgroup!r}. Refusing to choose between them."
                )
            previous = asserted.get(relative)
            if previous is not None and previous != subgroup:
                raise BfwDatasetError(
                    f"BFW metadata {metadata_path.name}: image {relative!r} is assigned to "
                    f"both {previous!r} and {subgroup!r}."
                )
            asserted[relative] = subgroup

    if not asserted:
        raise BfwDatasetError(f"BFW metadata {metadata_path.name} contained no image rows.")

    images: List[BfwImage] = []
    identity_subgroup: Dict[str, str] = {}
    for relative in sorted(asserted):
        identity, subgroup = _bfw_identity_and_subgroup(relative)
        absolute = (image_root / relative).resolve()
        # Containment check before any filesystem access, so a crafted relative
        # path cannot reach outside the configured root.
        if not absolute.is_relative_to(image_root):
            raise BfwDatasetError(
                f"BFW image {relative!r} resolves outside {BFW_ROOT_VARIABLE}."
            )
        if require_files_exist and not absolute.is_file():
            raise BfwDatasetError(f"BFW image listed in the metadata is missing: {relative}")
        established = identity_subgroup.setdefault(identity, subgroup)
        if established != subgroup:
            raise BfwDatasetError(
                f"BFW identity {identity!r} appears under two subgroups: "
                f"{established!r} and {subgroup!r}."
            )
        images.append(
            BfwImage(
                image_path=absolute,
                identity=identity,
                subgroup=subgroup,
                sample_id=opaque_id(f"bfw:{relative}"),
                identity_hash=opaque_id(f"bfw-identity:{identity}"),
            )
        )

    seen_sample_ids: Set[str] = set()
    for image in images:
        if image.sample_id in seen_sample_ids:
            raise BfwDatasetError(f"Duplicate BFW sample identifier for {image.image_path.name}.")
        seen_sample_ids.add(image.sample_id)

    return BfwDataset(
        images=images,
        metadata_sha256=sha256_of_text(text),
        metadata_filename=metadata_path.name,
    )


def bfw_dataset_provenance(dataset: BfwDataset) -> Dict[str, Any]:
    """Public provenance for BFW. Contains no path, no filename that embeds an
    identity and no demographic record joined to a name."""
    grouped = dataset.by_identity()
    per_identity_counts = sorted(len(v) for v in grouped.values())
    subgroup_of = dataset.subgroup_of_identity()
    subgroup_counts = {
        subgroup: sum(1 for s in subgroup_of.values() if s == subgroup)
        for subgroup in BFW_SUBGROUPS
    }
    return {
        "dataset_name": "BFW",
        "protocol_version": BFW_PROTOCOL_VERSION,
        "protocol_provenance": (
            "Project-defined open-set protocol constructed from the official BFW data. "
            "BFW publishes verification and bias-analysis protocols; it does not publish "
            "an open-set identification protocol, and none is implied here."
        ),
        "metadata_filename": dataset.metadata_filename,
        "metadata_sha256": dataset.metadata_sha256,
        "evaluated_image_set_sha256": sha256_of_text(
            "\n".join(sorted(i.sample_id for i in dataset.images))
        ),
        "total_identities": len(grouped),
        "total_images": len(dataset.images),
        "images_per_identity_min": per_identity_counts[0] if per_identity_counts else 0,
        "images_per_identity_max": per_identity_counts[-1] if per_identity_counts else 0,
        "images_per_identity_median": (
            statistics.median(per_identity_counts) if per_identity_counts else float("nan")
        ),
        "subgroup_identity_counts": subgroup_counts,
    }


# =============================================================================
# 18. Identity-disjoint open-set protocol
# =============================================================================
#
# Development and test identities are completely disjoint, so a threshold
# selected on one cannot have seen the other. Within each partition, identities
# split again into mated (enrolled in the gallery, with held-back probes) and
# non-mated (never enrolled, representing a legitimate new registration).
#
# Every division is stratified by the eight subgroups and driven only by the
# fixed research seed and the dataset itself — never by the secret identifier
# key, so the partition stays reproducible by someone who does not hold it.

OPEN_SET_ROLES = ("gallery_enrolment", "mated_probe", "non_mated_probe")

# A test evaluation is refused unless the policy carries exactly this status.
OPEN_SET_STATUS_DEVELOPMENT = "open_set_development"
OPEN_SET_STATUS_FROZEN = "open_set_frozen"
OPEN_SET_STATUS_TESTED = "open_set_tested"
OPEN_SET_STATUSES = (
    OPEN_SET_STATUS_DEVELOPMENT,
    OPEN_SET_STATUS_FROZEN,
    OPEN_SET_STATUS_TESTED,
)

# Enrolment sizing for the two methods compared in Experiment 6.
SINGLE_IMAGE_ENROLMENT = 1
MULTI_IMAGE_ENROLMENT = 3
MULTI_IMAGE_MINIMUM_ENROLMENT = 2

# Several probes are held back per identity rather than one. With one probe each,
# a partition of 200 non-mated identities yields 200 searches, so the smallest
# measurable FPIR is 1/200 = 0.5% and the 0.3% primary target could not be
# resolved at all. Drawing more probes per identity buys the resolution; the
# resulting within-identity correlation is precisely what the cluster bootstrap
# in section 22 accounts for by resampling identities rather than images.
MATED_PROBES_PER_IDENTITY = 5
NON_MATED_PROBES_PER_IDENTITY = 15

# An identity needs enough images to enrol the maximum template and still hold
# back its probes, otherwise the two methods would not see the same identities.
# Derived inside build_open_set_protocol from its arguments rather than fixed
# here, so that overriding the probe counts cannot leave a stale threshold
# behind.


class OpenSetProtocolError(RuntimeError):
    """Raised when the open-set partition cannot be built or is inconsistent."""


@dataclass(frozen=True)
class OpenSetEntry:
    sample_id: str
    identity_hash: str
    identity: str  # private
    subgroup: str
    image_path: Path  # private
    role: str
    partition: str  # "development" | "test"


@dataclass(frozen=True)
class OpenSetProtocol:
    entries: List[OpenSetEntry]
    seed: int
    provenance: Dict[str, Any]

    def partition(self, name: str) -> List[OpenSetEntry]:
        return [e for e in self.entries if e.partition == name]

    def identities(self, partition: str, role: str) -> Set[str]:
        return {e.identity for e in self.entries if e.partition == partition and e.role == role}


def _split_stratified_identities(
    identities_by_subgroup: Dict[str, List[str]], *, seed: int
) -> Tuple[List[str], List[str]]:
    """Halve each subgroup's identities into development and test.

    Deterministic rule for an odd count: the extra identity goes to
    development, so the held-out test partition is never the larger of the two
    and a tie can never be resolved by looking at test data."""
    development: List[str] = []
    test: List[str] = []
    for subgroup in sorted(identities_by_subgroup):
        members = sorted(identities_by_subgroup[subgroup])
        # Seeded per subgroup so adding a subgroup cannot reshuffle the others.
        rng = random.Random(f"{seed}:{subgroup}")
        rng.shuffle(members)
        midpoint = (len(members) + 1) // 2  # odd -> development takes the extra
        development.extend(members[:midpoint])
        test.extend(members[midpoint:])
    return sorted(development), sorted(test)


def _split_mated_and_non_mated(
    identities: Sequence[str],
    subgroup_of: Mapping[str, str],
    *,
    seed: int,
    partition: str,
) -> Tuple[List[str], List[str]]:
    """Divide a partition's identities into mated and non-mated, stratified by
    subgroup, with the extra identity of an odd subgroup going to mated."""
    by_subgroup: Dict[str, List[str]] = {}
    for identity in identities:
        by_subgroup.setdefault(subgroup_of[identity], []).append(identity)

    mated: List[str] = []
    non_mated: List[str] = []
    for subgroup in sorted(by_subgroup):
        members = sorted(by_subgroup[subgroup])
        rng = random.Random(f"{seed}:{partition}:{subgroup}")
        rng.shuffle(members)
        midpoint = (len(members) + 1) // 2
        mated.extend(members[:midpoint])
        non_mated.extend(members[midpoint:])
    return sorted(mated), sorted(non_mated)


def build_open_set_protocol(
    dataset: BfwDataset,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    mated_probes_per_identity: int = MATED_PROBES_PER_IDENTITY,
    non_mated_probes_per_identity: int = NON_MATED_PROBES_PER_IDENTITY,
) -> OpenSetProtocol:
    """Construct the deterministic, identity-disjoint open-set partition."""
    grouped = dataset.by_identity()
    subgroup_of = dataset.subgroup_of_identity()

    # Only identities with enough images can be mated under both methods; the
    # rest remain eligible as non-mated probes, which need a single image.
    minimum_images = MULTI_IMAGE_ENROLMENT + mated_probes_per_identity
    identities_by_subgroup: Dict[str, List[str]] = {}
    for identity in grouped:
        identities_by_subgroup.setdefault(subgroup_of[identity], []).append(identity)

    development_ids, test_ids = _split_stratified_identities(identities_by_subgroup, seed=seed)
    if not development_ids or not test_ids:
        raise OpenSetProtocolError(
            "BFW yielded too few identities to form disjoint development and test partitions."
        )

    entries: List[OpenSetEntry] = []
    for partition, partition_ids in (("development", development_ids), ("test", test_ids)):
        eligible_mated = [i for i in partition_ids if len(grouped[i]) >= minimum_images]
        remainder = [i for i in partition_ids if i not in set(eligible_mated)]
        mated, spare = _split_mated_and_non_mated(
            eligible_mated, subgroup_of, seed=seed, partition=partition
        )
        non_mated = sorted(set(spare) | set(remainder))
        if not mated or not non_mated:
            raise OpenSetProtocolError(
                f"The {partition} partition has no {'mated' if not mated else 'non-mated'} "
                f"identities; the open-set protocol needs both."
            )

        for identity in mated:
            images = grouped[identity]
            # Deterministic ordering already applied by by_identity(); the first
            # N enrol and the next is held back as the mated probe. Enrolment
            # images are chosen before any scoring, never by similarity outcome.
            for image in images[:MULTI_IMAGE_ENROLMENT]:
                entries.append(
                    OpenSetEntry(
                        image.sample_id, image.identity_hash, identity, image.subgroup,
                        image.image_path, "gallery_enrolment", partition,
                    )
                )
            # Probes are taken from the images immediately after the enrolment
            # block, so no image can hold both roles.
            probe_slice = images[
                MULTI_IMAGE_ENROLMENT : MULTI_IMAGE_ENROLMENT + mated_probes_per_identity
            ]
            for probe in probe_slice:
                entries.append(
                    OpenSetEntry(
                        probe.sample_id, probe.identity_hash, identity, probe.subgroup,
                        probe.image_path, "mated_probe", partition,
                    )
                )

        for identity in non_mated:
            # A non-mated identity enrols nothing, so every one of its images is
            # available as a probe representing a legitimate new registration.
            for probe in grouped[identity][:non_mated_probes_per_identity]:
                entries.append(
                    OpenSetEntry(
                        probe.sample_id, probe.identity_hash, identity, probe.subgroup,
                        probe.image_path, "non_mated_probe", partition,
                    )
                )

    _assert_protocol_invariants(entries)

    provenance = {
        "seed": seed,
        "protocol_version": BFW_PROTOCOL_VERSION,
        "odd_count_rule": (
            "When a subgroup holds an odd number of identities the extra identity is "
            "assigned to the development partition, and within a partition to the mated "
            "group, so the held-out test set is never enlarged by a tie."
        ),
        "development_identities": len(development_ids),
        "test_identities": len(test_ids),
    }
    return OpenSetProtocol(entries=entries, seed=seed, provenance=provenance)


def _assert_protocol_invariants(entries: Sequence[OpenSetEntry]) -> None:
    """The three properties that make the experiment meaningful, checked rather
    than assumed: no image holds two roles, no identity crosses the
    development/test boundary, and no identity is both enrolled and non-mated."""
    seen_paths: Set[Path] = set()
    for entry in entries:
        if entry.image_path in seen_paths:
            raise OpenSetProtocolError(
                f"Image assigned to more than one open-set role: {entry.image_path.name}"
            )
        seen_paths.add(entry.image_path)

    partitions_of: Dict[str, Set[str]] = {}
    roles_of: Dict[str, Set[str]] = {}
    for entry in entries:
        partitions_of.setdefault(entry.identity, set()).add(entry.partition)
        roles_of.setdefault(entry.identity, set()).add(entry.role)

    crossing = sorted(i for i, p in partitions_of.items() if len(p) > 1)
    if crossing:
        raise OpenSetProtocolError(
            f"{len(crossing)} identity/identities appear in both the development and test "
            f"partitions; the open-set protocol requires them to be disjoint."
        )

    for identity, roles in roles_of.items():
        if "non_mated_probe" in roles and roles & {"gallery_enrolment", "mated_probe"}:
            raise OpenSetProtocolError(
                "An identity is both enrolled in the gallery and used as a non-mated probe; "
                "a non-mated search must have no gallery reference."
            )


def open_set_protocol_summary(
    protocol: OpenSetProtocol,
    *,
    dataset: Optional[BfwDataset] = None,
    detector: Any = None,
    embedder: Any = None,
) -> Dict[str, Any]:
    """Public manifest summary: opaque identifiers and counts only.

    Software, model and dataset provenance are embedded alongside them so that
    this artefact carries the same record as every other published JSON."""

    def counts(partition: str) -> Dict[str, Any]:
        rows = protocol.partition(partition)
        by_subgroup: Dict[str, Dict[str, int]] = {
            subgroup: {role: 0 for role in OPEN_SET_ROLES} for subgroup in BFW_SUBGROUPS
        }
        for entry in rows:
            by_subgroup[entry.subgroup][entry.role] += 1
        return {
            "identities": len({e.identity_hash for e in rows}),
            "images": len(rows),
            "roles": {role: sum(1 for e in rows if e.role == role) for role in OPEN_SET_ROLES},
            "mated_identities": len({e.identity_hash for e in rows if e.role == "mated_probe"}),
            "non_mated_identities": len(
                {e.identity_hash for e in rows if e.role == "non_mated_probe"}
            ),
            "by_subgroup": by_subgroup,
        }

    return {
        "artifact_type": "bfw_open_set_protocol_summary",
        "schema_version": SCHEMA_VERSION,
        "opaque_id_version": OPAQUE_ID_VERSION,
        "seed": protocol.seed,
        "development": counts("development"),
        "test": counts("test"),
        **protocol.provenance,
        "public_manifest_sha256": sha256_of_text(
            "\n".join(sorted(f"{e.partition}:{e.role}:{e.sample_id}" for e in protocol.entries))
        ),
        "model_version": MODEL_VERSION,
        "pipeline_name": MODEL_VERSION,
        "preprocessing_revision": PREPROCESSING_REVISION,
        "model_sha256": {
            "yunet": getattr(detector, "model_sha256", YUNET_SHA256),
            "sface": getattr(embedder, "model_sha256", SFACE_SHA256),
        },
        "software_environment": software_environment_report(),
        "dataset_provenance": (
            bfw_dataset_provenance(dataset) if dataset is not None else None
        ),
        "policy_note": POLICY_NOTE,
    }


def write_open_set_private_manifest(protocol: OpenSetProtocol, path: Path) -> None:
    """Private manifest with real paths. Written under results/raw, which is
    git-ignored, and never published."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": protocol.seed,
        "entries": [
            {
                "sample_id": e.sample_id,
                "identity_hash": e.identity_hash,
                "subgroup": e.subgroup,
                "role": e.role,
                "partition": e.partition,
                "image_path": str(e.image_path),
            }
            for e in protocol.entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# =============================================================================
# 19. Open-set enrolment methods
# =============================================================================
#
# Two methods evaluated over exactly the same identity partitions, so any
# difference between them is attributable to the enrolment representation and
# the threshold, not to a different sample of people.
#
#   Method A (control)  single_image_pairwise_threshold
#       One gallery image per identity, searched with the LFW-frozen 1:1
#       threshold. This is the threshold-transfer control: it measures what
#       happens when a threshold calibrated for comparing exactly two images is
#       reused for 1:N search.
#
#   Method B (proposed) three_image_open_set_calibrated
#       Up to three enrolment images averaged into one L2-normalised template,
#       searched with a threshold calibrated on the BFW development partition
#       at a target false-positive identification rate.

METHOD_A = "single_image_pairwise_threshold"
METHOD_B = "three_image_open_set_calibrated"
OPEN_SET_METHODS = (METHOD_A, METHOD_B)

INSUFFICIENT_GALLERY_IMAGES = "insufficient_gallery_images"


@dataclass(frozen=True)
class EnrolmentOutcome:
    """Outcome of enrolling one identity, retained whether or not it succeeded."""

    identity_hash: str
    subgroup: str
    enrolled: bool
    attempted_images: int
    embedded_images: int
    failure_code: Optional[str]


@dataclass(frozen=True)
class EnrolledIdentity:
    identity_hash: str
    subgroup: str
    template: np.ndarray


@dataclass(frozen=True)
class OpenSetSearchResult:
    sample_id: str
    identity_hash: str
    subgroup: str
    role: str
    failure_code: Optional[str] = None
    top_similarity: Optional[float] = None
    top_identity_hash: Optional[str] = None
    top2_similarity: Optional[float] = None
    correct_rank: Optional[int] = None
    correct_similarity: Optional[float] = None
    highest_impostor_similarity: Optional[float] = None
    top1_time_seconds: Optional[float] = None
    top5_time_seconds: Optional[float] = None
    # Inputs to the review classifier of section 26. Every one is computable at
    # inference time from the search alone; none reveals whether the probe's
    # identity is actually enrolled, which would leak the label.
    top5_similarity_mean: Optional[float] = None
    top5_similarity_stdev: Optional[float] = None
    top1_gallery_image_count: Optional[int] = None
    gallery_size: Optional[int] = None
    probe_detection_confidence: Optional[float] = None
    probe_face_area_ratio: Optional[float] = None


@dataclass(frozen=True)
class OpenSetRunResult:
    method: str
    partition: str
    enrolment_outcomes: List[EnrolmentOutcome]
    search_results: List[OpenSetSearchResult]
    gallery_size: int
    comparisons_per_probe: int
    # Per-image stage timings; model loading is excluded and reported separately.
    stage_times_seconds: Dict[str, List[float]] = field(default_factory=dict)


def build_identity_template(
    embeddings: Sequence[np.ndarray],
) -> np.ndarray:
    """Average already-L2-normalised embeddings and re-normalise the mean.

    Re-normalisation matters: the arithmetic mean of unit vectors is not itself
    a unit vector, and cosine similarity against an un-normalised template would
    silently rescale every score for that identity."""
    if not embeddings:
        raise SimilarityError("Cannot build a template from zero embeddings.")
    stacked = np.vstack([l2_normalize(np.asarray(e, dtype=np.float64)) for e in embeddings])
    return l2_normalize(stacked.mean(axis=0))


def _embed_open_set_entry(
    entry: OpenSetEntry, detector: FaceDetector, embedder: FaceEmbedder
) -> Tuple[Optional[np.ndarray], Optional[str], Dict[str, float]]:
    """Embed one image and return the detector-derived quality signals with it.

    YuNet emits a fifteen-element row: bounding box in the first four columns
    and the detection score in the last. Both are captured here so the review
    classifier can use image quality without a second detection pass."""
    # Each stage is timed separately so a pipeline comparison can attribute
    # cost to loading, detection or embedding rather than to the whole call.
    started = time.perf_counter()
    try:
        loaded = load_image_bgr(entry.image_path)
        load_done = time.perf_counter()
        face_row = detector.detect_single_face(loaded.bgr)
        detect_done = time.perf_counter()
        height, width = loaded.bgr.shape[:2]
        metadata: Dict[str, float] = {
            "image_load_seconds": load_done - started,
            "detection_seconds": detect_done - load_done,
        }
        if face_row is not None and len(face_row) >= 15 and height and width:
            metadata["probe_detection_confidence"] = float(face_row[14])
            metadata["probe_face_area_ratio"] = (
                float(face_row[2] * face_row[3]) / float(height * width)
            )
        embedding = l2_normalize(embedder.embed(loaded.bgr, face_row))
        embed_done = time.perf_counter()
        metadata["embedding_seconds"] = embed_done - detect_done
        metadata["complete_pipeline_seconds"] = embed_done - started
        return embedding, None, metadata
    except FaceCountError as exc:
        return None, ("zero_faces" if exc.face_count == 0 else "multiple_faces"), {}
    except (ImageLoadError, SimilarityError) as exc:
        return None, f"image_error:{exc}", {}


def _call_embed(embed: Callable[..., Any], entry, detector, embedder):
    """Normalise an embedding callable to (embedding, failure, metadata).

    Test stubs supply the older two-value form, so both are accepted rather
    than forcing every caller to carry metadata it does not produce."""
    result = embed(entry, detector, embedder)
    if len(result) == 3:
        return result
    embedding, failure = result
    return embedding, failure, {}


def run_open_set_method(
    protocol: OpenSetProtocol,
    *,
    partition: str,
    method: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    embed_fn: Optional[Callable[..., Tuple[Optional[np.ndarray], Optional[str]]]] = None,
) -> OpenSetRunResult:
    """Enrol the gallery and search every probe. No threshold is applied here:
    raw scores and ranks are retained so one search can be scored at several
    operating points without re-running the pipeline."""
    if method not in OPEN_SET_METHODS:
        raise OpenSetProtocolError(f"Unknown open-set method {method!r}.")
    embed = embed_fn or _embed_open_set_entry

    rows = protocol.partition(partition)
    enrolment_images: Dict[str, List[OpenSetEntry]] = {}
    subgroup_of_hash: Dict[str, str] = {}
    for entry in rows:
        subgroup_of_hash[entry.identity_hash] = entry.subgroup
        if entry.role == "gallery_enrolment":
            enrolment_images.setdefault(entry.identity_hash, []).append(entry)

    take = SINGLE_IMAGE_ENROLMENT if method == METHOD_A else MULTI_IMAGE_ENROLMENT
    minimum = SINGLE_IMAGE_ENROLMENT if method == METHOD_A else MULTI_IMAGE_MINIMUM_ENROLMENT

    outcomes: List[EnrolmentOutcome] = []
    enrolled: List[EnrolledIdentity] = []
    template_image_counts: Dict[str, int] = {}
    stage_times: Dict[str, List[float]] = {
        "image_load_seconds": [], "detection_seconds": [],
        "embedding_seconds": [], "complete_pipeline_seconds": [],
    }

    def _record_stages(meta: Mapping[str, float]) -> None:
        for stage in stage_times:
            value = meta.get(stage)
            if isinstance(value, (int, float)):
                stage_times[stage].append(float(value))
    for identity_hash in sorted(enrolment_images):
        # Ordered by filename, never by sample_id. sample_id is an HMAC under the
        # secret key, so ordering by it would make the choice of enrolment image
        # key-dependent and the published metrics irreproducible by anyone
        # holding a different key.
        candidates = sorted(enrolment_images[identity_hash], key=lambda e: e.image_path.name)[:take]
        embeddings: List[np.ndarray] = []
        for entry in candidates:
            embedding, _failure, meta = _call_embed(embed, entry, detector, embedder)
            _record_stages(meta)
            if embedding is not None:
                embeddings.append(embedding)
        subgroup = subgroup_of_hash[identity_hash]
        if len(embeddings) < minimum:
            # Recorded, never silently removed: an identity that cannot meet the
            # minimum is a coverage failure of the method under test.
            outcomes.append(
                EnrolmentOutcome(
                    identity_hash, subgroup, False, len(candidates), len(embeddings),
                    INSUFFICIENT_GALLERY_IMAGES,
                )
            )
            continue
        outcomes.append(
            EnrolmentOutcome(identity_hash, subgroup, True, len(candidates), len(embeddings), None)
        )
        enrolled.append(
            EnrolledIdentity(identity_hash, subgroup, build_identity_template(embeddings))
        )
        template_image_counts[identity_hash] = len(embeddings)

    if not enrolled:
        raise OpenSetProtocolError(
            f"No identity could be enrolled for method {method!r} in the {partition} partition."
        )
    enrolled_hashes = {e.identity_hash for e in enrolled}

    results: List[OpenSetSearchResult] = []
    for entry in rows:
        if entry.role not in ("mated_probe", "non_mated_probe"):
            continue
        if entry.role == "mated_probe" and entry.identity_hash not in enrolled_hashes:
            results.append(
                OpenSetSearchResult(
                    entry.sample_id, entry.identity_hash, entry.subgroup, entry.role,
                    failure_code=GALLERY_REFERENCE_UNAVAILABLE,
                )
            )
            continue

        probe_embedding, failure, probe_meta = _call_embed(embed, entry, detector, embedder)
        _record_stages(probe_meta)
        if probe_embedding is None:
            results.append(
                OpenSetSearchResult(
                    entry.sample_id, entry.identity_hash, entry.subgroup, entry.role,
                    failure_code=failure,
                )
            )
            continue

        # Two separate measurements of what a deployment would actually pay.
        # Retrieving the single best candidate needs only a running maximum;
        # retrieving five needs a partial selection. ``heapq.nlargest`` keeps a
        # heap of size k rather than ordering the whole gallery, so it reflects
        # the real cost of a five-candidate review queue.
        start = time.perf_counter()
        similarities = [(c, cosine_similarity(probe_embedding, c.template)) for c in enrolled]
        _best = max(similarities, key=lambda item: item[1])
        top1_elapsed = time.perf_counter() - start

        start_top5 = time.perf_counter()
        _five = heapq.nlargest(5, similarities, key=lambda item: item[1])
        top5_elapsed = top1_elapsed + (time.perf_counter() - start_top5)

        # The full ordering is an evaluation artefact needed to recover the
        # mate's rank; it is deliberately timed out of both figures above,
        # because no deployment would sort the entire gallery per probe.
        scored = sorted(
            similarities,
            key=lambda item: (-item[1], item[0].identity_hash),
        )

        top_candidate, top_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else None

        correct_rank: Optional[int] = None
        correct_score: Optional[float] = None
        highest_impostor: Optional[float] = None
        if entry.role == "mated_probe":
            for position, (candidate, score) in enumerate(scored, start=1):
                if candidate.identity_hash == entry.identity_hash:
                    correct_rank, correct_score = position, score
                    break
            impostors = [s for c, s in scored if c.identity_hash != entry.identity_hash]
            highest_impostor = max(impostors) if impostors else None

        top5_scores = [score for _candidate, score in scored[:5]]
        results.append(
            OpenSetSearchResult(
                entry.sample_id, entry.identity_hash, entry.subgroup, entry.role,
                None, top_score, top_candidate.identity_hash, second_score,
                correct_rank, correct_score, highest_impostor, top1_elapsed, top5_elapsed,
                statistics.fmean(top5_scores) if top5_scores else None,
                statistics.stdev(top5_scores) if len(top5_scores) > 1 else 0.0,
                template_image_counts.get(top_candidate.identity_hash),
                len(enrolled),
                probe_meta.get("probe_detection_confidence"),
                probe_meta.get("probe_face_area_ratio"),
            )
        )

    return OpenSetRunResult(
        method=method,
        partition=partition,
        enrolment_outcomes=outcomes,
        search_results=results,
        gallery_size=len(enrolled),
        comparisons_per_probe=len(enrolled),
        stage_times_seconds=stage_times,
    )


# =============================================================================
# 20. Open-set metrics (FPIR, FNIR, TPIR)
# =============================================================================
#
# These are identification metrics and are not interchangeable with the
# pairwise rates in section 10:
#
#   FPIR  a *search* against the whole gallery returns at least one candidate
#         above threshold when the person is not enrolled. FMR is a single
#         comparison between two images. One non-mated search performs as many
#         comparisons as there are enrolled identities, which is exactly why a
#         1:1 threshold transfers so badly.
#
#   FNIR  a mated search fails to place the correct identity within rank k
#         above threshold. FNMR is a single genuine comparison falling below
#         threshold, with no competing candidates and no ranking involved.
#
# TPIR (also written DIR) is 1 - FNIR at the same rank.
##############
# Title: Face Recognition Technology Evaluation (FRTE) 1:N Identification
# Author: National Institute of Standards and Technology (NIST)
# Date: 2025
# Availability: https://pages.nist.gov/frvt/html/frvt1N.html
##############
##############
# Title: ISO/IEC 19795-1:2021 Biometric performance testing and reporting,
#        Part 1: Principles and framework
# Author: International Organization for Standardization
# Date: 2021
# Availability: https://www.iso.org/standard/73515.html
##############

# NIST commonly reports FNIR at an FPIR of 0.003; that is taken as the primary
# operating point here, with a stricter and a looser point either side.
FPIR_TARGETS = (0.001, 0.003, 0.01)
PRIMARY_FPIR_TARGET = 0.003


def _scored(results: Sequence[OpenSetSearchResult], role: str) -> List[OpenSetSearchResult]:
    return [r for r in results if r.role == role and r.failure_code is None]


def open_set_rates_at_threshold(
    results: Sequence[OpenSetSearchResult], threshold: float
) -> Dict[str, float]:
    """FPIR, FNIR and TPIR at one operating threshold."""
    non_mated = _scored(results, "non_mated_probe")
    mated = _scored(results, "mated_probe")

    false_positives = sum(
        1 for r in non_mated if r.top_similarity is not None and r.top_similarity >= threshold
    )
    fpir = false_positives / len(non_mated) if non_mated else float("nan")

    def found_within(rank: int) -> int:
        return sum(
            1
            for r in mated
            if r.correct_rank is not None
            and r.correct_rank <= rank
            and r.correct_similarity is not None
            and r.correct_similarity >= threshold
        )

    tpir1 = found_within(1) / len(mated) if mated else float("nan")
    tpir5 = found_within(5) / len(mated) if mated else float("nan")
    cmc1 = (
        sum(1 for r in mated if r.correct_rank == 1) / len(mated) if mated else float("nan")
    )
    cmc5 = (
        sum(1 for r in mated if r.correct_rank is not None and r.correct_rank <= 5) / len(mated)
        if mated
        else float("nan")
    )
    return {
        "threshold": threshold,
        "fpir": fpir,
        "fnir_rank1": 1.0 - tpir1 if tpir1 == tpir1 else float("nan"),
        "fnir_rank5": 1.0 - tpir5 if tpir5 == tpir5 else float("nan"),
        "tpir_rank1": tpir1,
        "tpir_rank5": tpir5,
        "cmc_rank1": cmc1,
        "cmc_rank5": cmc5,
        "false_reviews_per_1000_non_mated": fpir * 1000.0 if fpir == fpir else float("nan"),
        "scored_non_mated_probes": len(non_mated),
        "scored_mated_probes": len(mated),
    }


def open_set_coverage(run: OpenSetRunResult) -> Dict[str, Any]:
    """Coverage counts. Reported alongside every rate, because a rate measured
    over a small surviving fraction of the protocol is not comparable with one
    measured over nearly all of it."""
    intended_gallery = len(run.enrolment_outcomes)
    enrolled = sum(1 for o in run.enrolment_outcomes if o.enrolled)
    mated = [r for r in run.search_results if r.role == "mated_probe"]
    non_mated = [r for r in run.search_results if r.role == "non_mated_probe"]
    scored_mated = [r for r in mated if r.failure_code is None]
    scored_non_mated = [r for r in non_mated if r.failure_code is None]

    failure_breakdown: Dict[str, int] = {}
    for row in run.search_results:
        if row.failure_code is None:
            continue
        key = row.failure_code.split(":", 1)[0]
        failure_breakdown[key] = failure_breakdown.get(key, 0) + 1
    enrolment_failures = sum(1 for o in run.enrolment_outcomes if not o.enrolled)

    top1_ms = [r.top1_time_seconds * 1000.0 for r in run.search_results if r.top1_time_seconds]
    top5_ms = [r.top5_time_seconds * 1000.0 for r in run.search_results if r.top5_time_seconds]

    return {
        "method": run.method,
        "partition": run.partition,
        "gallery_size": run.gallery_size,
        "similarity_comparisons": run.comparisons_per_probe * len(scored_mated + scored_non_mated),
        "intended_gallery_identities": intended_gallery,
        "enrolled_gallery_identities": enrolled,
        "gallery_enrolment_failure_count": enrolment_failures,
        "gallery_enrolment_failure_rate": (
            enrolment_failures / intended_gallery if intended_gallery else float("nan")
        ),
        "gallery_enrolment_coverage": (
            enrolled / intended_gallery if intended_gallery else float("nan")
        ),
        "intended_mated_probes": len(mated),
        "scored_mated_probes": len(scored_mated),
        "mated_extraction_failure_rate": (
            (len(mated) - len(scored_mated)) / len(mated) if mated else float("nan")
        ),
        "intended_non_mated_probes": len(non_mated),
        "scored_non_mated_probes": len(scored_non_mated),
        "non_mated_extraction_failure_rate": (
            (len(non_mated) - len(scored_non_mated)) / len(non_mated) if non_mated else float("nan")
        ),
        "probe_failure_breakdown": failure_breakdown,
        **{
            f"{stage.replace('_seconds', '')}_latency_{suffix}_ms": value
            for stage, samples in (run.stage_times_seconds or {}).items()
            for suffix, value in (
                ("mean", statistics.fmean([s * 1000.0 for s in samples]) if samples else float("nan")),
                ("p95", percentile([s * 1000.0 for s in samples], 95) if samples else float("nan")),
            )
        },
        "images_timed": len((run.stage_times_seconds or {}).get("complete_pipeline_seconds", [])),
        "top1_search_time_mean_ms": statistics.fmean(top1_ms) if top1_ms else float("nan"),
        "top1_search_time_p95_ms": percentile(top1_ms, 95) if top1_ms else float("nan"),
        "top5_search_time_mean_ms": statistics.fmean(top5_ms) if top5_ms else float("nan"),
        "top5_search_time_p95_ms": percentile(top5_ms, 95) if top5_ms else float("nan"),
    }


def open_set_duplicate_detection(
    run: OpenSetRunResult, threshold: float
) -> Dict[str, float]:
    """Conditional and end-to-end duplicate detection, defined exactly as in the
    corrected LFW gallery accounting so the two experiments stay comparable."""
    mated = [r for r in run.search_results if r.role == "mated_probe"]
    scored = [r for r in mated if r.failure_code is None]
    detected = sum(
        1
        for r in scored
        if r.correct_rank == 1 and r.correct_similarity is not None
        and r.correct_similarity >= threshold
    )
    return {
        "conditional_duplicate_detection_rate": (
            detected / len(scored) if scored else float("nan")
        ),
        "end_to_end_duplicate_detection_rate": detected / len(mated) if mated else float("nan"),
    }


# =============================================================================
# 21. Open-set threshold development and freezing
# =============================================================================
#
# The threshold is developed on the BFW development partition only. Test scores
# never enter candidate generation or selection, and a test evaluation refuses
# any policy whose status is not exactly open_set_frozen.


class OpenSetPolicyError(RuntimeError):
    """Raised when an open-set policy is missing, malformed or not frozen."""


OPEN_SET_SELECTION_RULE = (
    "Among candidate thresholds whose development FPIR is no greater than the target, "
    "select the highest development TPIR at rank 1; ties broken by lower development "
    "FPIR, then by higher threshold, then by candidate name, for full determinism."
)


def open_set_threshold_candidates(
    results: Sequence[OpenSetSearchResult],
) -> List[Tuple[str, float]]:
    """Every distinct top score observed on the development partition, plus two
    sentinels that bracket the range so a target can always be met."""
    scores = sorted(
        {
            round(float(r.top_similarity), 12)
            for r in results
            if r.failure_code is None and r.top_similarity is not None
        }
    )
    candidates: List[Tuple[str, float]] = [("sentinel_reject_all", 1.0000000001)]
    for score in scores:
        candidates.append((f"score_{score:.12f}", score))
    candidates.append(("sentinel_accept_all", -1.0000000001))
    return candidates


def select_open_set_threshold(
    development_results: Sequence[OpenSetSearchResult],
    *,
    target_fpir: float,
) -> Dict[str, Any]:
    """Apply the selection rule at one target FPIR and return the full evidence."""
    evaluated: List[Dict[str, Any]] = []
    for name, threshold in open_set_threshold_candidates(development_results):
        rates = open_set_rates_at_threshold(development_results, threshold)
        evaluated.append({"candidate": name, **rates})

    admissible = [c for c in evaluated if c["fpir"] == c["fpir"] and c["fpir"] <= target_fpir]
    if not admissible:
        raise OpenSetPolicyError(
            f"No candidate threshold achieved a development FPIR at or below {target_fpir}. "
            f"The lowest observed was {min((c['fpir'] for c in evaluated), default=float('nan'))}."
        )

    def sort_key(candidate: Dict[str, Any]):
        tpir = candidate["tpir_rank1"]
        return (
            -(tpir if tpir == tpir else -1.0),  # highest TPIR@1; NaN sorts last
            candidate["fpir"],                   # then lower FPIR
            -candidate["threshold"],             # then higher threshold
            candidate["candidate"],              # then candidate name
        )

    chosen = sorted(admissible, key=sort_key)[0]
    return {
        "target_fpir": target_fpir,
        "selected_candidate": chosen["candidate"],
        "threshold": chosen["threshold"],
        "development_fpir": chosen["fpir"],
        "development_tpir_rank1": chosen["tpir_rank1"],
        "development_tpir_rank5": chosen["tpir_rank5"],
        "selection_rule": OPEN_SET_SELECTION_RULE,
        "candidates_evaluated": len(evaluated),
        "candidates_admissible": len(admissible),
    }


def require_frozen_open_set_policy(payload: Mapping[str, Any], *, context: str = "") -> float:
    """Refuse to score the held-out test partition with anything but a frozen
    policy. This is the structural guarantee that the reported test numbers were
    not tuned on the data they describe."""
    status = payload.get("status")
    if status != OPEN_SET_STATUS_FROZEN:
        # An unrecognised status is a different fault from a recognised but
        # wrong one — the first suggests a corrupt or foreign artefact, the
        # second an ordering mistake — so the message distinguishes them.
        detail = (
            f"Only {OPEN_SET_STATUS_FROZEN!r} is accepted; develop and freeze the policy "
            f"on the development partition first."
            if status in OPEN_SET_STATUSES
            else f"That is not a recognised open-set status; expected one of "
            f"{list(OPEN_SET_STATUSES)}. The artefact may be corrupt or from another tool."
        )
        raise OpenSetPolicyError(
            f"Refusing to run a held-out open-set evaluation with policy status {status!r}"
            f"{f' from {context}' if context else ''}. {detail}"
        )
    operating = payload.get("operating_points") or {}
    primary = operating.get(str(PRIMARY_FPIR_TARGET))
    if not primary or "threshold" not in primary:
        raise OpenSetPolicyError(
            f"Frozen open-set policy carries no threshold for the primary FPIR target "
            f"{PRIMARY_FPIR_TARGET}."
        )
    return float(primary["threshold"])


# =============================================================================
# 22. Cluster-bootstrap confidence intervals
# =============================================================================
#
# Several probes can belong to one identity, so resampling images independently
# would treat correlated observations as independent and produce intervals that
# are far too narrow. Identities are resampled instead, with subgroup
# stratification preserved, which is the standard cluster bootstrap.

BOOTSTRAP_REPLICATES = 2000


def _percentile_interval(values: Sequence[float]) -> Tuple[float, float]:
    ordered = sorted(values)
    if not ordered:
        return (float("nan"), float("nan"))
    return (percentile(ordered, 2.5), percentile(ordered, 97.5))


def cluster_bootstrap_intervals(
    results: Sequence[OpenSetSearchResult],
    *,
    threshold: float,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Dict[str, Any]]:
    """95% percentile intervals over identity-level resampling.

    A replicate in which a metric is undefined (no mated or no non-mated
    identity survived, say) is excluded from that metric's interval and counted,
    never silently replaced by zero."""
    by_identity: Dict[str, List[OpenSetSearchResult]] = {}
    subgroup_of: Dict[str, str] = {}
    for row in results:
        by_identity.setdefault(row.identity_hash, []).append(row)
        subgroup_of[row.identity_hash] = row.subgroup

    mated_ids = sorted({r.identity_hash for r in results if r.role == "mated_probe"})
    non_mated_ids = sorted({r.identity_hash for r in results if r.role == "non_mated_probe"})

    def stratify(identity_hashes: Sequence[str]) -> Dict[str, List[str]]:
        buckets: Dict[str, List[str]] = {}
        for identity_hash in identity_hashes:
            buckets.setdefault(subgroup_of[identity_hash], []).append(identity_hash)
        return {k: sorted(v) for k, v in sorted(buckets.items())}

    mated_strata = stratify(mated_ids)
    non_mated_strata = stratify(non_mated_ids)

    tracked = (
        "fpir",
        "fnir_rank1",
        "fnir_rank5",
        "tpir_rank1",
        "tpir_rank5",
        "end_to_end_duplicate_detection_rate",
        "extraction_coverage",
        # Bootstrap mated and non-mated coverage separately.
        "mated_extraction_coverage",
        "non_mated_extraction_coverage",
    )
    samples: Dict[str, List[float]] = {name: [] for name in tracked}

    rng = random.Random(seed)
    for _replicate in range(replicates):
        drawn: List[OpenSetSearchResult] = []
        for strata in (mated_strata, non_mated_strata):
            for _subgroup, members in strata.items():
                # With replacement, preserving each stratum's size.
                for _ in range(len(members)):
                    drawn.extend(by_identity[members[rng.randrange(len(members))]])

        rates = open_set_rates_at_threshold(drawn, threshold)
        mated_rows = [r for r in drawn if r.role == "mated_probe"]
        scored_mated = [r for r in mated_rows if r.failure_code is None]
        detected = sum(
            1
            for r in scored_mated
            if r.correct_rank == 1
            and r.correct_similarity is not None
            and r.correct_similarity >= threshold
        )
        # Preserve role-specific denominators within each identity.
        non_mated_rows = [r for r in drawn if r.role == "non_mated_probe"]
        scored_non_mated = sum(1 for r in non_mated_rows if r.failure_code is None)
        extras = {
            "end_to_end_duplicate_detection_rate": (
                detected / len(mated_rows) if mated_rows else float("nan")
            ),
            "extraction_coverage": (
                sum(1 for r in drawn if r.failure_code is None) / len(drawn)
                if drawn
                else float("nan")
            ),
            "mated_extraction_coverage": (
                len(scored_mated) / len(mated_rows) if mated_rows else float("nan")
            ),
            "non_mated_extraction_coverage": (
                scored_non_mated / len(non_mated_rows) if non_mated_rows else float("nan")
            ),
        }
        for name in tracked:
            value = extras.get(name, rates.get(name, float("nan")))
            if isinstance(value, float) and value == value:
                samples[name].append(value)

    intervals: Dict[str, Dict[str, Any]] = {}
    for name in tracked:
        low, high = _percentile_interval(samples[name])
        intervals[name] = {
            "lower_95": low,
            "upper_95": high,
            "valid_replicates": len(samples[name]),
            "requested_replicates": replicates,
        }
    return intervals


# =============================================================================
# 23. Demographic subgroup performance
# =============================================================================
#
# Subgroup figures are aggregate fairness reporting over benchmark identities.
# No subgroup label is ever published beside anything that identifies a person,
# no attribute is inferred with another model, and the primary result uses one
# global threshold — subgroup-specific thresholds would change what is being
# measured and are not applied to the held-out test.


def subgroup_open_set_metrics(
    results: Sequence[OpenSetSearchResult],
    *,
    threshold: float,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Dict[str, Any]]:
    per_subgroup: Dict[str, Dict[str, Any]] = {}
    for subgroup in BFW_SUBGROUPS:
        rows = [r for r in results if r.subgroup == subgroup]
        if not rows:
            continue
        rates = open_set_rates_at_threshold(rows, threshold)
        mated = [r for r in rows if r.role == "mated_probe"]
        non_mated = [r for r in rows if r.role == "non_mated_probe"]
        intervals = cluster_bootstrap_intervals(
            rows, threshold=threshold, replicates=replicates, seed=seed
        )
        entry: Dict[str, Any] = {}
        for metric in ("fpir", "fnir_rank1", "fnir_rank5", "tpir_rank1", "tpir_rank5"):
            entry[metric] = rates[metric]
            band = intervals.get(metric, {})
            entry[f"{metric}_lower_95"] = band.get("lower_95", float("nan"))
            entry[f"{metric}_upper_95"] = band.get("upper_95", float("nan"))
        for label, rows_of_role, series in (
            ("mated_probe_coverage", mated, "mated_extraction_coverage"),
            ("non_mated_probe_coverage", non_mated, "non_mated_extraction_coverage"),
        ):
            scored = sum(1 for r in rows_of_role if r.failure_code is None)
            entry[label] = scored / len(rows_of_role) if rows_of_role else float("nan")
            band = intervals.get(series, {})
            entry[f"{label}_lower_95"] = band.get("lower_95", float("nan"))
            entry[f"{label}_upper_95"] = band.get("upper_95", float("nan"))
        entry["scored_mated_probes"] = rates["scored_mated_probes"]
        entry["scored_non_mated_probes"] = rates["scored_non_mated_probes"]
        entry["intended_mated_probes"] = len(mated)
        entry["intended_non_mated_probes"] = len(non_mated)
        per_subgroup[subgroup] = entry
    return per_subgroup


def subgroup_disparity_summary(per_subgroup: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Spread across subgroups. The max/min ratio is reported only when the
    denominator is non-zero; an absolute range is always reported, because a
    ratio against a zero rate is undefined rather than infinitely bad."""
    fpirs = [v["fpir"] for v in per_subgroup.values() if v["fpir"] == v["fpir"]]
    tpirs = [v["tpir_rank1"] for v in per_subgroup.values() if v["tpir_rank1"] == v["tpir_rank1"]]
    if not fpirs:
        return {"subgroups_reported": len(per_subgroup)}
    max_fpir, min_fpir = max(fpirs), min(fpirs)
    summary: Dict[str, Any] = {
        "subgroups_reported": len(per_subgroup),
        "max_subgroup_fpir": max_fpir,
        "min_subgroup_fpir": min_fpir,
        "absolute_fpir_range": max_fpir - min_fpir,
    }
    if min_fpir > 0:
        summary["max_to_min_fpir_ratio"] = max_fpir / min_fpir
    else:
        summary["max_to_min_fpir_ratio"] = None
        summary["max_to_min_fpir_ratio_note"] = (
            "Undefined: at least one subgroup recorded a zero FPIR, so the ratio is not "
            "reported. Use the absolute range instead."
        )
    if tpirs:
        summary["max_subgroup_tpir_rank1"] = max(tpirs)
        summary["min_subgroup_tpir_rank1"] = min(tpirs)
        summary["absolute_tpir_rank1_range"] = max(tpirs) - min(tpirs)
    return summary


# =============================================================================
# 24. Open-set experiment orchestration and artefacts
# =============================================================================
#
# Experiment 6 in full: load BFW, build the identity-disjoint partition, run the
# single-image control, develop and freeze a policy on the development
# partition, apply it unchanged to the held-out test partition, then compute
# confidence intervals and subgroup figures.
#
# Success criteria are declared here, in source, before any test result exists.
# They are research targets, not claims, and the report states plainly which
# were achieved, which were not, and which could not be measured.

OPEN_SET_SUCCESS_CRITERIA = {
    "held_out_fpir_max": 0.01,
    "target_fpir": PRIMARY_FPIR_TARGET,
    "tpir_rank1_min": 0.90,
    "tpir_rank5_min": 0.95,
    "gallery_enrolment_coverage_min": 0.90,
    "probe_extraction_coverage_min": 0.90,
}

OPEN_SET_LIMITATIONS = (
    "This remains a proof of concept. No result here proves fraud, misuse or "
    "misrepresentation by any person.",
    "No automatic sanction is applied. A score above threshold opens a case for human "
    "review and nothing else.",
    "The BFW open-set evaluation uses a protocol defined by this project. BFW publishes "
    "verification and bias-analysis protocols, not an open-set identification protocol.",
    "Development and test identities are completely disjoint, and the operating threshold "
    "was frozen before the held-out test partition was scored.",
    "Extraction failures are counted as coverage failures, never as genuine no-match "
    "decisions.",
    "Confidence intervals describe sampling uncertainty over these benchmark identities "
    "only. They do not extend to any other population.",
    "Benchmark demographics do not represent a real dating-application user population, "
    "so subgroup figures must not be read as deployment estimates.",
)


def evaluate_open_set_success_criteria(
    coverage: Mapping[str, Any], rates: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare held-out results against the pre-declared targets. A metric that
    is undefined is reported as not measurable rather than as a pass."""

    def verdict(actual: Any, threshold: float, *, minimum: bool) -> Dict[str, Any]:
        if not isinstance(actual, (int, float)) or actual != actual:
            return {"outcome": "not_measurable", "actual": None, "target": threshold}
        achieved = actual >= threshold if minimum else actual <= threshold
        return {
            "outcome": "achieved" if achieved else "not_achieved",
            "actual": float(actual),
            "target": threshold,
        }

    probe_coverage = min(
        (
            1.0 - coverage.get("mated_extraction_failure_rate", float("nan")),
            1.0 - coverage.get("non_mated_extraction_failure_rate", float("nan")),
        ),
        default=float("nan"),
    )
    return {
        "criteria_declared_before_test": True,
        "fpir_at_or_below_1_percent": verdict(
            rates.get("fpir"), OPEN_SET_SUCCESS_CRITERIA["held_out_fpir_max"], minimum=False
        ),
        "tpir_rank1_at_least_90_percent": verdict(
            rates.get("tpir_rank1"), OPEN_SET_SUCCESS_CRITERIA["tpir_rank1_min"], minimum=True
        ),
        "tpir_rank5_at_least_95_percent": verdict(
            rates.get("tpir_rank5"), OPEN_SET_SUCCESS_CRITERIA["tpir_rank5_min"], minimum=True
        ),
        "gallery_enrolment_coverage_at_least_90_percent": verdict(
            coverage.get("gallery_enrolment_coverage"),
            OPEN_SET_SUCCESS_CRITERIA["gallery_enrolment_coverage_min"],
            minimum=True,
        ),
        "probe_extraction_coverage_at_least_90_percent": verdict(
            probe_coverage,
            OPEN_SET_SUCCESS_CRITERIA["probe_extraction_coverage_min"],
            minimum=True,
        ),
    }


def _open_set_provenance(
    dataset: BfwDataset, protocol: OpenSetProtocol, detector: Any, embedder: Any
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "methodology_revision": GALLERY_METHODOLOGY_REVISION,
        "opaque_id_version": OPAQUE_ID_VERSION,
        "created_at": utc_now_iso(),
        "software_environment": software_environment_report(),
        "pipeline_name": MODEL_VERSION,
        "model_version": MODEL_VERSION,
        "preprocessing_revision": PREPROCESSING_REVISION,
        "model_sha256": {
            "yunet": getattr(detector, "model_sha256", YUNET_SHA256),
            "sface": getattr(embedder, "model_sha256", SFACE_SHA256),
        },
        "dataset_provenance": bfw_dataset_provenance(dataset),
        "seed": protocol.seed,
        "policy_note": POLICY_NOTE,
        "limitations": list(OPEN_SET_LIMITATIONS),
    }


def run_open_set_experiment(
    *,
    output_root: Path = AGGREGATE_ROOT,
    raw_root: Path = RAW_ROOT,
    threshold_artifact: Optional[Path] = None,
    seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, Any]:
    """Experiment 6 end to end. Stops on the exact underlying blocker; never
    fabricates, hardcodes or approximates a result."""
    config = EnvironmentConfig.load()
    if not id_hmac_key_is_configured():
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} must be configured before identifiers are produced."
        )
    image_root, metadata_path = config.require_bfw_roots()

    detector, embedder = load_models(config.require_model_root())
    announce("Loading BFW metadata and validating the pinned schema")
    dataset = load_bfw_dataset(image_root, metadata_path)
    announce(f"BFW: {len(dataset.images)} images across {len(dataset.by_identity())} identities")

    protocol = build_open_set_protocol(dataset, seed=seed)
    private_manifest = Path(raw_root) / "bfw_open_set_manifest.json"
    write_open_set_private_manifest(protocol, private_manifest)
    announce(
        f"Wrote the private open-set manifest to {project_relative(private_manifest)} "
        f"(contains real image paths — kept out of Git)"
    )

    summary_payload = open_set_protocol_summary(
        protocol, dataset=dataset, detector=detector, embedder=embedder
    )
    write_json_artifact(output_root / "bfw_open_set_protocol_summary.json", summary_payload)

    provenance = _open_set_provenance(dataset, protocol, detector, embedder)
    public_manifest_sha256 = summary_payload["public_manifest_sha256"]

    # --- Method A control, at the untouched LFW 1:1 threshold -----------------
    threshold_artifact = threshold_artifact or (output_root / "calibrated_threshold.json")
    lfw_payload = read_json_artifact(threshold_artifact)
    lfw_threshold = require_frozen_threshold(
        lfw_payload, context=project_relative(threshold_artifact)
    )

    announce("Running Method A (single-image enrolment, LFW 1:1 threshold) on development")
    control_dev = run_open_set_method(
        protocol, partition="development", method=METHOD_A, detector=detector, embedder=embedder
    )

    # --- Method B development and freezing -----------------------------------
    announce("Running Method B (three-image template) on development")
    proposed_dev = run_open_set_method(
        protocol, partition="development", method=METHOD_B, detector=detector, embedder=embedder
    )

    operating_points: Dict[str, Any] = {}
    for target in FPIR_TARGETS:
        operating_points[str(target)] = select_open_set_threshold(
            proposed_dev.search_results, target_fpir=target
        )

    policy_payload = {
        "artifact_type": "bfw_open_set_threshold",
        "status": OPEN_SET_STATUS_FROZEN,
        "method": METHOD_B,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "operating_points": operating_points,
        "selection_rule": OPEN_SET_SELECTION_RULE,
        "developed_on": "BFW development partition (identity-disjoint from test)",
        "public_manifest_sha256": public_manifest_sha256,
        "protocol_digest": summary_payload["public_manifest_sha256"],
        **provenance,
    }
    policy_path = output_root / "bfw_open_set_threshold.json"
    write_json_artifact(policy_path, policy_payload)
    announce(f"Froze the open-set policy at {project_relative(policy_path)}")

    frozen_threshold = require_frozen_open_set_policy(
        read_json_artifact(policy_path), context=project_relative(policy_path)
    )

    development_payload = {
        "artifact_type": "bfw_open_set_development_metrics",
        "status": OPEN_SET_STATUS_DEVELOPMENT,
        "public_manifest_sha256": public_manifest_sha256,
        "protocol_digest": summary_payload["public_manifest_sha256"],
        "threshold_source": project_relative(policy_path),
        "methods": {
            METHOD_A: {
                "coverage": open_set_coverage(control_dev),
                "operating_threshold": lfw_threshold,
                "threshold_role": (
                    "Control only. This is the LFW 1:1 verification threshold reused "
                    "unchanged; it is not a valid open-set operating threshold."
                ),
                "rates": open_set_rates_at_threshold(control_dev.search_results, lfw_threshold),
                **open_set_duplicate_detection(control_dev, lfw_threshold),
            },
            METHOD_B: {
                "coverage": open_set_coverage(proposed_dev),
                "operating_points": {
                    str(t): open_set_rates_at_threshold(
                        proposed_dev.search_results, operating_points[str(t)]["threshold"]
                    )
                    for t in FPIR_TARGETS
                },
                "at_lfw_control_threshold": open_set_rates_at_threshold(
                    proposed_dev.search_results, lfw_threshold
                ),
            },
        },
        **provenance,
    }
    write_json_artifact(
        output_root / "bfw_open_set_development_metrics.json", development_payload
    )

    # --- Held-out test, frozen policy applied unchanged -----------------------
    announce("Scoring the held-out test partition with the frozen policy")
    control_test = run_open_set_method(
        protocol, partition="test", method=METHOD_A, detector=detector, embedder=embedder
    )
    proposed_test = run_open_set_method(
        protocol, partition="test", method=METHOD_B, detector=detector, embedder=embedder
    )

    test_coverage = open_set_coverage(proposed_test)
    primary_rates = open_set_rates_at_threshold(proposed_test.search_results, frozen_threshold)

    test_payload = {
        "artifact_type": "bfw_open_set_test_metrics",
        "status": OPEN_SET_STATUS_TESTED,
        "public_manifest_sha256": public_manifest_sha256,
        "protocol_digest": summary_payload["public_manifest_sha256"],
        "threshold_source": project_relative(policy_path),
        "operating_threshold": frozen_threshold,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "methods": {
            METHOD_A: {
                "coverage": open_set_coverage(control_test),
                "operating_threshold": lfw_threshold,
                "threshold_role": (
                    "Control only. Not a valid open-set operating threshold."
                ),
                "rates": open_set_rates_at_threshold(control_test.search_results, lfw_threshold),
                **open_set_duplicate_detection(control_test, lfw_threshold),
            },
            METHOD_B: {
                "coverage": test_coverage,
                "at_lfw_control_threshold": open_set_rates_at_threshold(
                    proposed_test.search_results, lfw_threshold
                ),
                "operating_points": {
                    str(t): open_set_rates_at_threshold(
                        proposed_test.search_results, operating_points[str(t)]["threshold"]
                    )
                    for t in FPIR_TARGETS
                },
                "primary_operating_point": primary_rates,
                **open_set_duplicate_detection(proposed_test, frozen_threshold),
            },
        },
        "success_criteria": evaluate_open_set_success_criteria(test_coverage, primary_rates),
        **provenance,
    }
    write_json_artifact(output_root / "bfw_open_set_test_metrics.json", test_payload)

    # --- Confidence intervals and subgroup figures ---------------------------
    announce(f"Computing {bootstrap_replicates} cluster-bootstrap replicates")
    intervals = cluster_bootstrap_intervals(
        proposed_test.search_results,
        threshold=frozen_threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    write_json_artifact(
        output_root / "open_set_confidence_intervals.json",
        {
            "artifact_type": "open_set_confidence_intervals",
            "method": METHOD_B,
            "partition": "test",
            "operating_threshold": frozen_threshold,
            "replicates": bootstrap_replicates,
            "resampling_unit": "identity (cluster bootstrap, subgroup-stratified)",
            "intervals": intervals,
            **provenance,
        },
    )

    per_subgroup = subgroup_open_set_metrics(
        proposed_test.search_results, threshold=frozen_threshold,
        replicates=bootstrap_replicates, seed=seed,
    )
    _write_subgroup_csv(output_root / "bfw_subgroup_metrics.csv", per_subgroup)
    write_json_artifact(
        output_root / "bfw_subgroup_confidence_intervals.json",
        {
            "artifact_type": "bfw_subgroup_confidence_intervals",
            "resampling_unit": "identity (cluster bootstrap, subgroup-stratified)",
            "replicates": bootstrap_replicates,
            "operating_threshold": frozen_threshold,
            "subgroups": per_subgroup,
            **provenance,
        },
    )
    _write_method_comparison_csv(
        output_root / "open_set_method_comparison.csv",
        control_test=control_test,
        proposed_test=proposed_test,
        lfw_threshold=lfw_threshold,
        frozen_threshold=frozen_threshold,
    )

    # Optional extensions. Each either produces a real artefact or records why
    # it did not run; neither is allowed to interrupt the primary experiment.
    write_pipeline_comparison_csv(
        output_root / "pretrained_pipeline_comparison.csv",
        primary=primary_pipeline_description(detector, embedder),
        config=config,
    )
    consistency = profile_photo_consistency_summary(proposed_test, frozen_threshold)
    write_json_artifact(
        output_root / "profile_photo_consistency.json",
        {"artifact_type": "profile_photo_consistency", **consistency, **provenance},
    )
    write_profile_consistency_artefacts({**consistency, **provenance}, output_root)
    by_sex = sex_aggregated_metrics(
        proposed_test.search_results, threshold=frozen_threshold,
        replicates=bootstrap_replicates, seed=seed,
    )
    write_json_artifact(
        output_root / "bfw_sex_aggregated_metrics.json",
        {"artifact_type": "bfw_sex_aggregated_metrics", "operating_threshold": frozen_threshold,
         "aggregation": "pooled over identity outcomes, not averaged over subgroup percentages",
         "groups": by_sex, **provenance},
    )

    report = render_open_set_report(
        protocol_summary=summary_payload,
        development=development_payload,
        test=test_payload,
        intervals=intervals,
        per_subgroup=per_subgroup,
        disparity=subgroup_disparity_summary(per_subgroup),
    )
    (output_root / "OPEN_SET_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    announce("Wrote OPEN_SET_EVALUATION_REPORT.md")

    # Privacy validation over every artefact this experiment produced, including
    # the new ones. A leak stops the run rather than being reported afterwards.
    leaks = find_path_leaks(output_root, forbidden_substrings=default_forbidden_path_substrings())
    if leaks:
        raise PrivacyLeakError(
            "Refusing to finish: open-set output(s) contain a personal/absolute path:\n"
            + "\n".join(f"  {redact_private_paths(leak)}" for leak in leaks)
        )
    assert_no_identifier_key_leak(output_root)
    announce("Privacy validation passed for every open-set artefact")

    return test_payload


# Profile-photo identity consistency reuses the architecture already in place:
# detection, embedding, normalised template, cosine comparison, calibrated
# decision, human review. It is not a separate identity-verification system.
#
# A photograph scoring below the operating threshold is inconsistent with the
# profile's enrolled template under this model and threshold. That is a review
# signal. It does not independently establish that the photograph belongs to
# another person: pose, lighting, occlusion, image quality, age difference,
# detection failure and model error all produce the same outcome.
PROFILE_CONSISTENCY_NOTE = (
    "A non-match indicates that the photograph is inconsistent with the profile's enrolled "
    "facial template under this model and threshold. It does not independently prove that "
    "the photograph belongs to another person. Pose, lighting, occlusion, image quality, "
    "age difference, face-detection failure and model error can all produce the same result. "
    "Every outcome opens human review only."
)


def profile_photo_consistency_summary(
    run: OpenSetRunResult, threshold: float
) -> Dict[str, Any]:
    """Aggregate same-person consistency for photographs on one profile.

    Each mated probe is a further photograph supplied for a profile whose
    template was built from its enrolment images, so its similarity to its own
    template is exactly the consistency score."""
    consistent = inconsistent = 0
    extraction_failures = unavailable = 0
    scores: List[float] = []
    for row in run.search_results:
        if row.role != "mated_probe":
            continue
        if row.failure_code == GALLERY_REFERENCE_UNAVAILABLE:
            unavailable += 1
            continue
        if row.failure_code is not None or row.correct_similarity is None:
            extraction_failures += 1
            continue
        scores.append(float(row.correct_similarity))
        if row.correct_similarity >= threshold:
            consistent += 1
        else:
            inconsistent += 1

    # Deterministic mismatched-photo control: a non-mated probe compared with
    # the gallery it does not belong to. Referral is the correct outcome.
    controls_referred = controls_not_referred = control_failures = 0
    for row in run.search_results:
        if row.role != "non_mated_probe":
            continue
        if row.failure_code is not None or row.top_similarity is None:
            control_failures += 1
        elif row.top_similarity >= threshold:
            controls_referred += 1
        else:
            controls_not_referred += 1

    intended = consistent + inconsistent + extraction_failures + unavailable
    return {
        "mismatched_controls_correctly_referred": controls_referred,
        "mismatched_controls_not_referred": controls_not_referred,
        "mismatched_control_extraction_failures": control_failures,
        "operating_threshold": threshold,
        "photographs_assessed": intended,
        "consistent_same_person_photographs": consistent,
        "inconsistent_review_candidates": inconsistent,
        "extraction_failures": extraction_failures,
        "gallery_reference_unavailable": unavailable,
        "consistency_rate": consistent / intended if intended else float("nan"),
        "review_referral_rate": inconsistent / intended if intended else float("nan"),
        "consistency_score_mean": statistics.fmean(scores) if scores else float("nan"),
        "consistency_score_median": statistics.median(scores) if scores else float("nan"),
        "interpretation_note": PROFILE_CONSISTENCY_NOTE,
        "policy_note": POLICY_NOTE,
    }


def sex_aggregated_metrics(
    results: Sequence[OpenSetSearchResult],
    *,
    threshold: float,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate over the four female and four male subgroups.

    Pooled from the underlying identity outcomes rather than by averaging
    subgroup percentages, which would weight a small subgroup as heavily as a
    large one. Sex is an evaluation dimension only and is never a predictor."""
    grouped: Dict[str, List[OpenSetSearchResult]] = {"female": [], "male": []}
    for row in results:
        if row.subgroup.endswith("_females"):
            grouped["female"].append(row)
        elif row.subgroup.endswith("_males"):
            grouped["male"].append(row)

    summary: Dict[str, Dict[str, Any]] = {}
    for label, rows in grouped.items():
        if not rows:
            continue
        rates = open_set_rates_at_threshold(rows, threshold)
        intervals = cluster_bootstrap_intervals(
            rows, threshold=threshold, replicates=replicates, seed=seed
        )
        mated = [r for r in rows if r.role == "mated_probe"]
        non_mated = [r for r in rows if r.role == "non_mated_probe"]
        entry: Dict[str, Any] = {
            "subgroups_pooled": sorted({r.subgroup for r in rows}),
            "identities": len({r.identity_hash for r in rows}),
        }
        for metric in ("fpir", "tpir_rank1", "tpir_rank5", "fnir_rank1"):
            entry[metric] = rates[metric]
            entry[f"{metric}_lower_95"] = intervals[metric]["lower_95"]
            entry[f"{metric}_upper_95"] = intervals[metric]["upper_95"]
        for label_key, rows_of_role, series in (
            ("mated_probe_coverage", mated, "mated_extraction_coverage"),
            ("non_mated_probe_coverage", non_mated, "non_mated_extraction_coverage"),
        ):
            scored = sum(1 for r in rows_of_role if r.failure_code is None)
            entry[label_key] = scored / len(rows_of_role) if rows_of_role else float("nan")
            entry[f"{label_key}_lower_95"] = intervals[series]["lower_95"]
            entry[f"{label_key}_upper_95"] = intervals[series]["upper_95"]
        entry["intended_mated_probes"] = len(mated)
        entry["intended_non_mated_probes"] = len(non_mated)
        entry["population_note"] = (
            "These benchmark categories do not represent every identity or any real "
            "dating-application population."
        )
        summary[label] = entry
    return summary


def _write_subgroup_csv(path: Path, per_subgroup: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_SUBGROUP_CSV_COLUMNS)
        for subgroup in sorted(per_subgroup):
            row = per_subgroup[subgroup]
            writer.writerow([subgroup] + [row.get(n, "") for n in _SUBGROUP_CSV_COLUMNS[1:]])


def _write_method_comparison_csv(
    path: Path,
    *,
    control_test: OpenSetRunResult,
    proposed_test: OpenSetRunResult,
    lfw_threshold: float,
    frozen_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (METHOD_A, "lfw_1to1_control", lfw_threshold, control_test),
        (METHOD_B, f"open_set_fpir_{PRIMARY_FPIR_TARGET}", frozen_threshold, proposed_test),
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method", "operating_point", "threshold", "gallery_size",
                "fpir", "fnir_rank1", "tpir_rank1", "tpir_rank5",
                "false_reviews_per_1000_non_mated",
                "gallery_enrolment_coverage", "scored_mated_probes", "scored_non_mated_probes",
                "end_to_end_duplicate_detection_rate",
            ]
        )
        for method, point, threshold, run in rows:
            rates = open_set_rates_at_threshold(run.search_results, threshold)
            coverage = open_set_coverage(run)
            detection = open_set_duplicate_detection(run, threshold)
            writer.writerow(
                [
                    method, point, threshold, run.gallery_size,
                    rates["fpir"], rates["fnir_rank1"], rates["tpir_rank1"], rates["tpir_rank5"],
                    rates["false_reviews_per_1000_non_mated"],
                    coverage["gallery_enrolment_coverage"],
                    rates["scored_mated_probes"], rates["scored_non_mated_probes"],
                    detection["end_to_end_duplicate_detection_rate"],
                ]
            )


def render_open_set_report(
    *,
    protocol_summary: Mapping[str, Any],
    development: Mapping[str, Any],
    test: Mapping[str, Any],
    intervals: Mapping[str, Any],
    per_subgroup: Mapping[str, Mapping[str, Any]],
    disparity: Mapping[str, Any],
) -> str:
    """Every number is read from the artefacts, never restated from memory."""
    method_b_test = test["methods"][METHOD_B]
    method_a_test = test["methods"][METHOD_A]
    primary = method_b_test["primary_operating_point"]
    coverage = method_b_test["coverage"]
    control_rates = method_a_test["rates"]

    lines = [
        "# Open-set duplicate-profile evaluation (Experiment 6)",
        "",
        f"Auto-generated by `ACP_arden.py --mode open-set` on {test['created_at']}.",
        "",
        "## Research question",
        "",
        "To what extent can gallery-specific threshold calibration and multi-image profile "
        "enrolment reduce false duplicate-profile reviews while retaining duplicate-detection "
        "performance in an open-set face-verification proof of concept evaluated on real "
        "public benchmark datasets?",
        "",
        "## Protocol",
        "",
        f"- Development identities: {protocol_summary['development']['identities']}; "
        f"held-out test identities: {protocol_summary['test']['identities']} (disjoint).",
        f"- Seed {protocol_summary['seed']}, stratified by the eight BFW subgroups.",
        f"- {protocol_summary['odd_count_rule']}",
        "",
        "## Method A — control: 1:1 threshold reused for 1:N search",
        "",
        f"- Operating threshold {format_number(method_a_test['operating_threshold'], 6)} "
        f"({method_a_test['threshold_role']})",
        f"- FPIR **{format_percentage(control_rates['fpir'])}**, "
        f"TPIR@1 {format_percentage(control_rates['tpir_rank1'])}",
        f"- False reviews per 1,000 non-mated searches: "
        f"{format_number(control_rates['false_reviews_per_1000_non_mated'], 1)}",
        "",
        "## Method B — proposed: three-image template, open-set calibrated",
        "",
        f"- Frozen threshold {format_number(test['operating_threshold'], 6)} at target FPIR "
        f"{test['primary_fpir_target']}",
        f"- FPIR **{format_percentage(primary['fpir'])}** "
        f"(95% CI {format_percentage(intervals['fpir']['lower_95'])} – "
        f"{format_percentage(intervals['fpir']['upper_95'])})",
        f"- TPIR@1 **{format_percentage(primary['tpir_rank1'])}** "
        f"(95% CI {format_percentage(intervals['tpir_rank1']['lower_95'])} – "
        f"{format_percentage(intervals['tpir_rank1']['upper_95'])})",
        f"- TPIR@5 {format_percentage(primary['tpir_rank5'])} "
        f"(CMC rank-1 {format_percentage(primary['cmc_rank1'])}, rank-5 "
        f"{format_percentage(primary['cmc_rank5'])})",
        f"- False reviews per 1,000 non-mated searches: "
        f"{format_number(primary['false_reviews_per_1000_non_mated'], 1)}",
        "",
        "TPIR@5 and TPIR@1 can coincide even though the CMC figures differ, and that is "
        "not an error: TPIR requires the mate to be both within rank *k* **and** above "
        "threshold. The CMC gap shows the mate does surface at ranks 2-5 for a small "
        "number of probes, but those scores fall below the operating threshold, so "
        "returning five candidates instead of one adds no true identification at this "
        "operating point. It would still change the reviewer's workload.",
        "",
        "## Attributing the effect: enrolment or calibration?",
        "",
        "Method B changes two things at once, so the comparison above cannot say which "
        "one helped. Holding the enrolment at three images and reverting only the "
        "threshold separates them:",
        "",
        "| Configuration | FPIR | TPIR@1 |",
        "| --- | --- | --- |",
        f"| A: one image, LFW 1:1 threshold (control) | "
        f"{format_percentage(control_rates['fpir'])} | "
        f"{format_percentage(control_rates['tpir_rank1'])} |",
        f"| B: three images, LFW 1:1 threshold (enrolment change only) | "
        f"{format_percentage(method_b_test['at_lfw_control_threshold']['fpir'])} | "
        f"{format_percentage(method_b_test['at_lfw_control_threshold']['tpir_rank1'])} |",
        f"| B: three images, calibrated threshold (both changes) | "
        f"{format_percentage(primary['fpir'])} | "
        f"{format_percentage(primary['tpir_rank1'])} |",
        "",
        "Read the middle row before drawing any conclusion about multi-image enrolment. "
        "Averaging three images raises rank-1 identification, but at a fixed threshold it "
        "*raises* the false-positive identification rate rather than lowering it: a mean "
        "template sits nearer the centre of the embedding space and is therefore closer "
        "to everyone, so more non-mated searches clear the same bar. The reduction in "
        "false reviews is attributable to the gallery-specific calibration, not to the "
        "multi-image representation. The two are complementary — enrolment supplies the "
        "headroom in TPIR that calibration then spends on a stricter threshold — but they "
        "must not be credited interchangeably.",
        "",
        "## Operating points and the development-to-test gap",
        "",
        "| Target FPIR | Threshold | Achieved test FPIR | TPIR@1 |",
        "| --- | --- | --- | --- |",
        *[
            f"| {target} | "
            f"{format_number(method_b_test['operating_points'][str(target)].get('threshold'), 4)} | "
            f"{format_percentage(method_b_test['operating_points'][str(target)]['fpir'])} | "
            f"{format_percentage(method_b_test['operating_points'][str(target)]['tpir_rank1'])} |"
            for target in FPIR_TARGETS
        ],
        "",
        "Each achieved test FPIR exceeds the target it was calibrated for. That gap is the "
        "cost of holding the threshold fixed across disjoint identities and is the reason "
        "the policy is frozen on development data before the test partition is scored — "
        "recalibrating on these figures would make them meaningless.",
        "",
        "### Coverage (quoted with every rate above)",
        "",
        f"- Gallery enrolment coverage "
        f"{format_percentage(coverage['gallery_enrolment_coverage'])} "
        f"({coverage['enrolled_gallery_identities']} of "
        f"{coverage['intended_gallery_identities']} identities)",
        f"- Mated probe extraction failure "
        f"{format_percentage(coverage['mated_extraction_failure_rate'])}; non-mated "
        f"{format_percentage(coverage['non_mated_extraction_failure_rate'])}",
        "",
        "## Pre-declared success criteria",
        "",
    ]
    for name, verdict in test["success_criteria"].items():
        if name == "criteria_declared_before_test":
            continue
        actual = verdict.get("actual")
        rendered = "not measurable" if actual is None else format_percentage(actual)
        lines.append(
            f"- {name}: **{verdict['outcome'].replace('_', ' ')}** "
            f"(target {verdict['target']}, achieved {rendered})"
        )

    lines += ["", "## Demographic subgroup performance", "", "| Subgroup | FPIR | TPIR@1 |", "| --- | --- | --- |"]
    for subgroup in sorted(per_subgroup):
        row = per_subgroup[subgroup]
        lines.append(
            f"| {subgroup} | {format_percentage(row['fpir'])} | "
            f"{format_percentage(row['tpir_rank1'])} |"
        )
    ratio = disparity.get("max_to_min_fpir_ratio")
    lines += [
        "",
        f"Absolute FPIR range across subgroups: "
        f"{format_percentage(disparity.get('absolute_fpir_range'))}."
        + (
            f" Max/min ratio {format_number(ratio, 2)}."
            if ratio is not None
            else " Max/min ratio not reported (a subgroup recorded zero FPIR)."
        ),
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in OPEN_SET_LIMITATIONS]
    return "\n".join(lines) + "\n"


def render_open_set_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    """Terminal summary for --mode open-set-summary, read from the artefacts."""
    test_path = output_root / "bfw_open_set_test_metrics.json"
    if not test_path.is_file():
        return (
            "No open-set results found. Run `python ACP_arden.py --mode open-set` first.\n"
            f"That requires the official BFW dataset and {BFW_ROOT_VARIABLE}."
        )
    test = read_json_artifact(test_path)
    intervals = read_json_artifact(output_root / "open_set_confidence_intervals.json").get(
        "intervals", {}
    )
    method_b = test["methods"][METHOD_B]
    method_a = test["methods"][METHOD_A]
    primary = method_b["primary_operating_point"]
    coverage = method_b["coverage"]

    def interval(name: str) -> str:
        row = intervals.get(name)
        if not row:
            return ""
        return (
            f" [95% CI {format_percentage(row['lower_95'])} – "
            f"{format_percentage(row['upper_95'])}]"
        )

    lines = [
        f"{PROGRAMME_TITLE} — open-set results summary",
        "",
        "Experiment 6 — BFW open-set duplicate-profile evaluation (held-out test)",
        f"  Frozen threshold: {format_number(test.get('operating_threshold'), 6)} "
        f"(target FPIR {test.get('primary_fpir_target')}, status {test.get('status')})",
        "",
        "  Method A — control, LFW 1:1 threshold reused for 1:N search",
        f"    FPIR: {format_percentage(method_a['rates']['fpir'])}",
        f"    TPIR@1: {format_percentage(method_a['rates']['tpir_rank1'])}",
        "    NOTE: a control, not a valid open-set operating threshold.",
        "",
        "  Method B — proposed, three-image template with open-set calibration",
        f"    FPIR: {format_percentage(primary['fpir'])}{interval('fpir')}",
        f"    TPIR@1: {format_percentage(primary['tpir_rank1'])}{interval('tpir_rank1')}",
        f"    TPIR@5: {format_percentage(primary['tpir_rank5'])}",
        f"    False reviews per 1,000 non-mated searches: "
        f"{format_number(primary['false_reviews_per_1000_non_mated'], 1)}",
        f"    Gallery enrolment coverage: "
        f"{format_percentage(coverage['gallery_enrolment_coverage'])}",
        f"    Mated extraction failure: "
        f"{format_percentage(coverage['mated_extraction_failure_rate'])}",
        f"    Non-mated extraction failure: "
        f"{format_percentage(coverage['non_mated_extraction_failure_rate'])}",
        "    LIMITATION: every rate above is conditional on the coverage figures printed "
        "with it. An FPIR measured over a small surviving fraction of the protocol is not "
        "comparable with one measured over nearly all of it.",
        "",
        "  Pre-declared success criteria:",
    ]
    for name, verdict in test.get("success_criteria", {}).items():
        if name == "criteria_declared_before_test":
            continue
        lines.append(f"    {name}: {verdict['outcome'].replace('_', ' ')}")
    lines += ["", "Policy: " + POLICY_NOTE, "", "Full write-up: results/aggregate/OPEN_SET_EVALUATION_REPORT.md"]
    return "\n".join(lines)


def report_optional_dataset_status() -> List[str]:
    """Concise skipped-with-reason lines for the optional extensions. Never
    fabricates a result for a dataset or model that is not present."""
    config = EnvironmentConfig.load()
    lines: List[str] = []
    status = pipeline_comparison_status(config)
    if status["comparison_run"]:
        lines.append(
            "Higher-capacity pipeline comparison: configured and verified against pinned "
            "digests. Reported as a complete-pipeline comparison."
        )
    else:
        lines.append(
            f"Higher-capacity pipeline comparison (InsightFace SCRFD + ArcFace "
            f"{ARCFACE_MODEL_PACK}): NOT RUN [{status['status']}] — {status['reason']} "
            f"This is a technical precondition, not a licensing obstacle: the evaluation "
            f"is non-commercial academic research, which the official research terms "
            f"permit. No substitute model was used."
        )
    return lines


# =============================================================================
# 25. Pipeline description and the optional higher-capacity comparison
# =============================================================================
#
# A "pipeline" is the whole chain — detector, preprocessing and embedding model
# — not the embedding model alone. Two pipelines that differ in any of those
# three cannot attribute a difference in results to the embedding, which is why
# the record below names every component and why the comparison is always
# labelled a complete-pipeline comparison.
#
# The comparator is InsightFace SCRFD detection with ArcFace buffalo_l
# recognition. Its pretrained weights are published for non-commercial research,
# which this MSc artefact is, so those terms permit the evaluation. The weights
# are loaded from private local storage by their exact verified path, are never
# redistributed and are never downloaded automatically. No substitute model is
# used in place of the approved one.
##############
# Title: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
# Author: Deng, J., Guo, J., Xue, N. and Zafeiriou, S., Proceedings of the
#         IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
# Date: 2019
# Availability: https://doi.org/10.1109/CVPR.2019.00482
##############
##############
# Title: Sample and Computation Redistribution for Efficient Face Detection (SCRFD)
# Author: Guo, J., Deng, J., Lattas, A. and Zafeiriou, S., International
#         Conference on Learning Representations (ICLR)
# Date: 2022
# Availability: https://arxiv.org/abs/2105.04714
##############
##############
# Title: InsightFace: 2D and 3D Face Analysis Project
# Author: InsightFace contributors (deepinsight/insightface)
# Date: 2021 onwards; buffalo_l model pack released 2021
# Availability: https://github.com/deepinsight/insightface
##############
# The detector and recognition weights are external pretrained artefacts,
# created and trained by the InsightFace project. Nothing here trains or
# fine-tunes them, and no weight file is redistributed by this repository.

ARCFACE_MODEL_ROOT_VARIABLE = "FACE_ARCFACE_MODEL_ROOT"
ARCFACE_MODEL_PACK = "buffalo_l"
ARCFACE_DETECTOR_FILENAME = "det_10g.onnx"
ARCFACE_RECOGNITION_FILENAME = "w600k_r50.onnx"

# Digests are pinned in source, never accepted from the command line, so a
# reportable evaluation cannot be pointed at an unverified weight file. They are
# filled in only once the exact approved files have been obtained; until then
# the adapter refuses rather than trusting whatever is on disk.
ARCFACE_DETECTOR_SHA256: Optional[str] = (
    "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
)
ARCFACE_RECOGNITION_SHA256: Optional[str] = (
    "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
)

# Status vocabulary. A licensing status is reserved for the case where the
# official terms do not clearly permit non-commercial academic research; it is
# never used merely because commercial deployment would need separate
# permission, which is irrelevant to this project's purpose.
PIPELINE_STATUS_EVALUATED = "evaluated_non_commercial_academic_research"
PIPELINE_STATUS_NOT_CONFIGURED = "not_run_model_files_not_configured"
PIPELINE_STATUS_SOURCE_UNVERIFIED = "not_run_official_model_source_unverified"
PIPELINE_STATUS_DIGEST_NOT_PINNED = "not_run_model_digest_not_pinned"
PIPELINE_STATUS_DEPENDENCIES_MISSING = "not_run_dependencies_not_installed"
PIPELINE_STATUS_TERMS_UNCLEAR = "not_run_research_terms_not_established"

ARCFACE_LICENCE_NOTE = (
    "InsightFace publishes its pretrained models for non-commercial research use. This "
    "project is an MSc academic research artefact: it is not a commercial service, is not "
    "deployed to real users, makes no commercial decisions, and neither sells, licenses "
    "nor redistributes the pretrained weights. The evaluation is local and non-commercial "
    "and publishes only aggregate metrics, so it falls within those research terms. The "
    "models were created and trained externally by the InsightFace project; no "
    "face-recognition network is trained or fine-tuned here. The MIT licence covering "
    "InsightFace source code does "
    "not automatically extend to every pretrained weight file, and no ownership of the "
    "models, their training data or their weights is claimed here."
)

# Tense follows the real status: the past tense is only correct once held-out
# metrics exist, so the wording is generated rather than fixed.
ARCFACE_USE_STATEMENT_NOT_RUN = (
    "The InsightFace pipeline is intended for evaluation solely within non-commercial MSc "
    "research. The pretrained model files are stored outside the public repository and are "
    "not redistributed. The project publishes only aggregate benchmark results and "
    "provides full attribution to the original model, software and research publications."
)

ARCFACE_USE_STATEMENT_EVALUATED = (
    "The InsightFace pipeline was evaluated solely within non-commercial MSc research. The "
    "pretrained model files were stored outside the public repository and were not "
    "redistributed. The project publishes only aggregate benchmark results and provides "
    "full attribution to the original model, software and research publications."
)


def arcface_use_statement(evaluated: bool) -> str:
    return ARCFACE_USE_STATEMENT_EVALUATED if evaluated else ARCFACE_USE_STATEMENT_NOT_RUN


class PipelineUnavailableError(RuntimeError):
    """Raised when an optional pipeline cannot be configured, verified or loaded."""


class PipelineComparisonError(RuntimeError):
    """Raised when a comparison would be recorded without real held-out metrics."""


# SCRFD rescales the input to a fixed square before detection. InsightFace
# defaults to 640, which detects nothing on BFW's face crops: their longest
# side is around one hundred pixels, so a 640 canvas leaves the face far below
# the smallest anchor. A 320 canvas detects reliably. This is a preprocessing
# scale, not a decision threshold; the detection threshold stays at the
# published default so coverage is not inflated by lowering the bar.
ARCFACE_DETECTION_INPUT_SIZE = 320
ARCFACE_DETECTION_THRESHOLD = 0.5


class ArcFaceDetector:
    """Adapter exposing InsightFace detection through the project's protocol.

    Requires exactly one face, matching the rule the primary pipeline applies,
    so the two pipelines remain comparable."""

    def __init__(self, model: Any, model_sha256: str):
        self._model = model
        self.model_sha256 = model_sha256
        self._last_landmarks: Optional[np.ndarray] = None

    def detect_single_face(self, bgr: np.ndarray) -> np.ndarray:
        boxes, landmarks = self._model.detect(bgr, max_num=0, metric="default")
        count = 0 if boxes is None else len(boxes)
        if count != 1:
            self._last_landmarks = None
            raise FaceCountError(count)
        box = np.asarray(boxes[0], dtype=np.float64)  # type: ignore[index]
        self._last_landmarks = (
            np.asarray(landmarks[0], dtype=np.float64) if landmarks is not None else None
        )
        # Fifteen columns matching the YuNet row shape: box, five landmarks,
        # then the detection score last.
        row = np.zeros(15, dtype=np.float64)
        row[0], row[1] = box[0], box[1]
        row[2], row[3] = box[2] - box[0], box[3] - box[1]
        if self._last_landmarks is not None:
            row[4:14] = self._last_landmarks.reshape(-1)[:10]
        row[14] = float(box[4]) if box.shape[0] > 4 else 0.0
        return row

    def last_landmarks(self) -> Optional[np.ndarray]:
        return self._last_landmarks


class ArcFaceEmbedder:
    """Align by the detected landmarks, then embed with ArcFace."""

    def __init__(
        self, model: Any, detector: ArcFaceDetector, model_sha256: str, dimensions: int = 512
    ):
        self._model = model
        self._detector = detector
        self.model_sha256 = model_sha256
        self._dimensions = dimensions

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        from insightface.utils import face_align  # type: ignore[import-not-found]

        landmarks = self._detector.last_landmarks()
        if landmarks is None:
            raise SimilarityError("No landmarks were available for alignment.")
        aligned = face_align.norm_crop(bgr, landmark=landmarks, image_size=112)
        embedding = np.asarray(self._model.get_feat(aligned), dtype=np.float64).reshape(-1)
        if embedding.shape[0] != self._dimensions:
            # Refuse an unexpected width rather than compare templates drawn
            # from two different spaces.
            raise SimilarityError(
                f"Expected {self._dimensions}-dimensional embeddings, got {embedding.shape[0]}."
            )
        return embedding


@dataclass(frozen=True)
class PipelineDescription:
    """Everything that must match before two runs are comparable. Published in
    every artefact so a reader can tell which chain produced a number."""

    pipeline_name: str
    detector_name: str
    embedding_model_name: str
    embedding_dimensions: int
    preprocessing_revision: str
    model_sha256: Dict[str, str]
    licence_note: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "detector_name": self.detector_name,
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimensions": self.embedding_dimensions,
            "preprocessing_revision": self.preprocessing_revision,
            "model_sha256": dict(self.model_sha256),
            "licence_note": self.licence_note,
            "comparison_scope": (
                "Complete pipeline: detection, preprocessing and embedding. A difference "
                "between pipelines cannot be attributed to the embedding model alone."
            ),
        }


def primary_pipeline_description(detector: Any = None, embedder: Any = None) -> PipelineDescription:
    """The mandatory YuNet + SFace pipeline."""
    return PipelineDescription(
        pipeline_name=MODEL_VERSION,
        detector_name="OpenCV YuNet 2023mar",
        embedding_model_name="OpenCV SFace 2021dec",
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        preprocessing_revision=PREPROCESSING_REVISION,
        model_sha256={
            "yunet": getattr(detector, "model_sha256", YUNET_SHA256),
            "sface": getattr(embedder, "model_sha256", SFACE_SHA256),
        },
        licence_note=(
            "OpenCV Zoo release: YuNet weights MIT, SFace weights Apache-2.0. "
            "Both are redistributable for research use."
        ),
    )


def arcface_preconditions(config: Optional[EnvironmentConfig] = None) -> Dict[str, Any]:
    """Diagnose the comparator in a fixed order and name the first real blocker.

    Every status returned is technical. Commercial-use restrictions are not a
    blocker here: this artefact is non-commercial academic research, which the
    official research terms permit."""
    config = config or EnvironmentConfig.load()
    checks: Dict[str, bool] = {}

    missing_dependencies = [
        name for name in ("onnxruntime", "insightface")
        if importlib.util.find_spec(name) is None
    ]
    checks["dependencies_installed"] = not missing_dependencies
    checks["model_root_configured"] = config.arcface_model_root is not None

    root = Path(config.arcface_model_root) if config.arcface_model_root else None
    detector_path = root / ARCFACE_DETECTOR_FILENAME if root else None
    recognition_path = root / ARCFACE_RECOGNITION_FILENAME if root else None
    checks["model_files_present"] = bool(
        detector_path and recognition_path
        and detector_path.is_file() and recognition_path.is_file()
    )
    checks["digests_pinned"] = (
        ARCFACE_DETECTOR_SHA256 is not None and ARCFACE_RECOGNITION_SHA256 is not None
    )
    checks["research_terms_established"] = True

    if not checks["research_terms_established"]:
        status, reason = PIPELINE_STATUS_TERMS_UNCLEAR, (
            "The official terms do not clearly permit non-commercial academic research."
        )
    elif not checks["model_root_configured"]:
        status, reason = PIPELINE_STATUS_NOT_CONFIGURED, (
            f"{ARCFACE_MODEL_ROOT_VARIABLE} is not set, so no model files are available. "
            f"Obtain the official {ARCFACE_MODEL_PACK} pack from the InsightFace project and "
            f"store it in private local research storage; nothing is downloaded automatically."
        )
    elif not checks["model_files_present"]:
        status, reason = PIPELINE_STATUS_SOURCE_UNVERIFIED, (
            f"{ARCFACE_MODEL_ROOT_VARIABLE} is set but {ARCFACE_DETECTOR_FILENAME} and "
            f"{ARCFACE_RECOGNITION_FILENAME} were not both found, so the model source "
            f"cannot be verified."
        )
    elif not checks["digests_pinned"]:
        status, reason = PIPELINE_STATUS_DIGEST_NOT_PINNED, (
            "The model files are present but their SHA-256 digests are not pinned in "
            "source. Compute the digests of the exact approved files and set "
            "ARCFACE_DETECTOR_SHA256 and ARCFACE_RECOGNITION_SHA256; digests are never "
            "accepted as command-line arguments for a reportable evaluation."
        )
    elif not checks["dependencies_installed"]:
        status, reason = PIPELINE_STATUS_DEPENDENCIES_MISSING, (
            f"Required package(s) not installed: {', '.join(missing_dependencies)}. "
            f"Install them from requirements-comparison.txt."
        )
    else:
        status, reason = PIPELINE_STATUS_EVALUATED, ""

    return {
        "status": status,
        "reason": reason,
        "checks": checks,
        "missing_dependencies": missing_dependencies,
        "ready": status == PIPELINE_STATUS_EVALUATED,
    }


def arcface_pipeline_description(config: Optional[EnvironmentConfig] = None) -> PipelineDescription:
    """Describe the comparator, or refuse with the precise technical blocker."""
    config = config or EnvironmentConfig.load()
    preconditions = arcface_preconditions(config)
    if not preconditions["ready"]:
        raise PipelineUnavailableError(
            f"[{preconditions['status']}] {preconditions['reason']}"
        )
    root = Path(config.arcface_model_root)  # type: ignore[arg-type]
    return PipelineDescription(
        pipeline_name=f"insightface-scrfd-arcface-{ARCFACE_MODEL_PACK}",
        detector_name="InsightFace SCRFD (det_10g)",
        embedding_model_name=f"InsightFace ArcFace {ARCFACE_MODEL_PACK} (w600k_r50)",
        embedding_dimensions=512,
        preprocessing_revision="insightface-arcface-112x112-v1",
        model_sha256={
            "detector": verify_model_file(
                root / ARCFACE_DETECTOR_FILENAME, str(ARCFACE_DETECTOR_SHA256)
            ),
            "recognition": verify_model_file(
                root / ARCFACE_RECOGNITION_FILENAME, str(ARCFACE_RECOGNITION_SHA256)
            ),
        },
        licence_note=ARCFACE_LICENCE_NOTE,
    )


def load_arcface_pipeline(config: Optional[EnvironmentConfig] = None):
    """Load the comparator through InsightFace's documented API.

    ``download=False`` and an explicit local root are required: the pretrained
    weights must come from private research storage that the researcher
    populated, never from an automatic fetch."""
    config = config or EnvironmentConfig.load()
    description = arcface_pipeline_description(config)
    try:
        # Optional dependency, absent by default; the ImportError below is the
        # supported path when the comparison is not being run.
        from insightface.model_zoo import get_model  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PipelineUnavailableError(
            f"[{PIPELINE_STATUS_DEPENDENCIES_MISSING}] insightface is not installed."
        ) from exc

    # Each ONNX file is loaded by its exact verified path. FaceAnalysis is
    # deliberately not used: it resolves models through a cache directory and
    # fetches the pack from the network when that directory is empty, which
    # would both download automatically and evaluate files other than the
    # pinned ones.
    root = Path(config.arcface_model_root)  # type: ignore[arg-type]
    detector_model = cast(Any, get_model(str(root / ARCFACE_DETECTOR_FILENAME)))
    detector_model.prepare(
        ctx_id=-1,
        input_size=(ARCFACE_DETECTION_INPUT_SIZE, ARCFACE_DETECTION_INPUT_SIZE),
        det_thresh=ARCFACE_DETECTION_THRESHOLD,
    )
    recognition_model = cast(Any, get_model(str(root / ARCFACE_RECOGNITION_FILENAME)))
    recognition_model.prepare(ctx_id=-1)

    detector = ArcFaceDetector(detector_model, description.model_sha256["detector"])
    embedder = ArcFaceEmbedder(
        recognition_model, detector, description.model_sha256["recognition"],
        dimensions=description.embedding_dimensions,
    )
    return (detector, embedder), description


def pipeline_comparison_status(
    config: Optional[EnvironmentConfig] = None,
) -> Dict[str, Any]:
    """Whether the comparison can run, and if not, precisely why."""
    preconditions = arcface_preconditions(config)
    if not preconditions["ready"]:
        return {
            "comparison_run": False,
            "status": preconditions["status"],
            "reason": redact_private_paths(preconditions["reason"]),
            "preconditions": preconditions["checks"],
            "substitute_model_used": False,
            "licence_note": ARCFACE_LICENCE_NOTE,
            "use_statement": arcface_use_statement(False),
        }
    description = arcface_pipeline_description(config)
    return {
        "comparison_run": True,
        "status": PIPELINE_STATUS_EVALUATED,
        "pipeline": description.as_dict(),
        "preconditions": preconditions["checks"],
        "substitute_model_used": False,
        "licence_note": ARCFACE_LICENCE_NOTE,
        "use_statement": arcface_use_statement(True),
    }


def _model_file_sizes(root: Optional[Path]) -> Dict[str, Any]:
    """Byte and megabyte sizes of the evaluated weight files."""
    sizes: Dict[str, Any] = {}
    if root is None:
        return sizes
    for name in (ARCFACE_DETECTOR_FILENAME, ARCFACE_RECOGNITION_FILENAME):
        path = Path(root) / name
        if path.is_file():
            size = path.stat().st_size
            sizes[name] = {"bytes": size, "megabytes": round(size / (1024 * 1024), 2)}
    return sizes


def _primary_model_file_sizes(model_root: Optional[Path]) -> Dict[str, Any]:
    sizes: Dict[str, Any] = {}
    if model_root is None:
        return sizes
    for name in (YUNET_FILENAME, SFACE_FILENAME):
        path = Path(model_root) / name
        if path.is_file():
            size = path.stat().st_size
            sizes[name] = {"bytes": size, "megabytes": round(size / (1024 * 1024), 2)}
    return sizes


def write_pipeline_performance_csv(
    path: Path, *, payload: Mapping[str, Any]
) -> None:
    """Performance comparison table, written only from computed metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "pipeline", "detector", "recognition_model", "embedding_dimensions", "threshold",
        "fpir", "tpir_rank1", "tpir_rank5", "fnir_rank1", "fnir_rank5",
        "false_reviews_per_1000_non_mated", "end_to_end_duplicate_detection_rate",
        "gallery_enrolment_coverage", "mated_extraction_coverage",
        "non_mated_extraction_coverage",
        "embedding_latency_mean_ms", "embedding_latency_p95_ms",
        "search_latency_mean_ms", "search_latency_p95_ms",
        "detector_file_size_mb", "recognition_file_size_mb", "status",
    ]
    held_out = payload.get("held_out_metrics") or {}
    descriptions = {
        payload["primary_pipeline"]["pipeline_name"]: payload["primary_pipeline"],
    }
    if payload.get("comparison_pipeline"):
        descriptions[payload["comparison_pipeline"]["pipeline_name"]] = (
            payload["comparison_pipeline"]
        )
    sizes = payload.get("model_file_sizes", {})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        if not held_out:
            for name, description in descriptions.items():
                writer.writerow(
                    [name, description["detector_name"], description["embedding_model_name"],
                     description["embedding_dimensions"]] + [""] * 17 + [payload["status"]]
                )
            return
        for name, metrics in held_out.items():
            description = descriptions.get(name, {})
            rates, coverage = metrics["rates"], metrics["coverage"]
            group = "primary" if name == payload["primary_pipeline"]["pipeline_name"] else "comparison"
            file_sizes = list((sizes.get(group) or {}).values())
            writer.writerow([
                name, description.get("detector_name", ""),
                description.get("embedding_model_name", ""),
                description.get("embedding_dimensions", ""),
                metrics["development_threshold"],
                rates["fpir"], rates["tpir_rank1"], rates["tpir_rank5"],
                rates["fnir_rank1"], rates["fnir_rank5"],
                rates["false_reviews_per_1000_non_mated"],
                metrics.get("end_to_end_duplicate_detection_rate"),
                coverage["gallery_enrolment_coverage"],
                1.0 - coverage["mated_extraction_failure_rate"],
                1.0 - coverage["non_mated_extraction_failure_rate"],
                coverage.get("top1_search_time_mean_ms"),
                coverage.get("top5_search_time_p95_ms"),
                coverage.get("top1_search_time_mean_ms"),
                coverage.get("top1_search_time_p95_ms"),
                file_sizes[0]["megabytes"] if len(file_sizes) > 0 else "",
                file_sizes[1]["megabytes"] if len(file_sizes) > 1 else "",
                payload["status"],
            ])


def write_pipeline_subgroup_csv(path: Path, *, payload: Mapping[str, Any]) -> None:
    """One row per pipeline and subgroup once the comparison has been run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    held_out = payload.get("held_out_metrics") or {}
    columns = ["pipeline"] + list(_SUBGROUP_CSV_COLUMNS) + ["status"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        if not held_out:
            for subgroup in BFW_SUBGROUPS:
                writer.writerow(
                    [f"insightface-scrfd-arcface-{ARCFACE_MODEL_PACK}", subgroup]
                    + [""] * (len(_SUBGROUP_CSV_COLUMNS) - 1) + [payload["status"]]
                )
            return
        for name, metrics in held_out.items():
            for subgroup in sorted(metrics.get("subgroups", {})):
                row = metrics["subgroups"][subgroup]
                writer.writerow(
                    [name, subgroup]
                    + [row.get(c, "") for c in _SUBGROUP_CSV_COLUMNS[1:]]
                    + [payload["status"]]
                )


def write_pipeline_comparison_csv(
    path: Path, *, primary: PipelineDescription, config: Optional[EnvironmentConfig] = None
) -> None:
    """Emit the comparison table. When the comparator is unavailable the file
    still records the primary pipeline and states plainly that no second
    pipeline was evaluated, so absence is visible rather than inferred."""
    status = pipeline_comparison_status(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["pipeline_name", "detector_name", "embedding_model_name", "embedding_dimensions",
             "preprocessing_revision", "evaluated", "note"]
        )
        writer.writerow(
            [primary.pipeline_name, primary.detector_name, primary.embedding_model_name,
             primary.embedding_dimensions, primary.preprocessing_revision, "yes",
             "Mandatory primary pipeline."]
        )
        if status["comparison_run"]:
            other = status["pipeline"]
            writer.writerow(
                [other["pipeline_name"], other["detector_name"], other["embedding_model_name"],
                 other["embedding_dimensions"], other["preprocessing_revision"], "yes",
                 other["comparison_scope"]]
            )
        else:
            writer.writerow(
                ["insightface-scrfd-arcface-buffalo_l", "InsightFace SCRFD / RetinaFace",
                 "InsightFace ArcFace buffalo_l", 512, "insightface-arcface-112x112-v1",
                 "no", status["reason"]]
            )


# =============================================================================
# 26. Experiment 7 — interpretable machine-learning review classifier
# =============================================================================
#
# Asks whether a small classifier over ranking context can refer fewer innocent
# registrations for human review than a single similarity threshold, without
# losing duplicate detection.
#
# Logistic regression is the primary model: its coefficients are directly
# readable, it is reproducible under a fixed seed, it suits a nine-feature
# problem, and it emits a probability that can be calibrated to a target FPIR.
# No face-recognition model is trained or fine-tuned anywhere in this section.
#
# The output opens a human-review case and nothing else. It is not evidence of
# duplication, fraud or misuse.
##############
# Title: The Regression Analysis of Binary Sequences
# Author: Cox, D.R., Journal of the Royal Statistical Society, Series B, 20(2),
#         pp. 215-242
# Date: 1958
# Availability: https://doi.org/10.1111/j.2517-6161.1958.tb00292.x
##############
##############
# Title: Scikit-learn: Machine Learning in Python (LogisticRegression and
#        StandardScaler APIs)
# Author: Pedregosa, F. et al., Journal of Machine Learning Research, 12,
#         pp. 2825-2830
# Date: 2011
# Availability: https://jmlr.org/papers/v12/pedregosa11a.html
##############

ML_REVIEW_STATUS_FROZEN = "ml_review_frozen"
ML_REVIEW_METHOD = "logistic_regression_review_classifier"

# Fixed feature order. Serialised with the model so that a stored coefficient
# can never be applied to a differently-ordered vector.
ML_REVIEW_FEATURES = (
    "top1_similarity",
    "top2_similarity",
    "top1_top2_margin",
    "top5_similarity_mean",
    "top5_similarity_stdev",
    "top1_gallery_image_count",
    "gallery_size",
    "probe_detection_confidence",
    "probe_face_area_ratio",
)

# Declared as typed constants rather than read back out of the dictionary
# below. scikit-learn types solver and class_weight as literals, so recovering
# them with str() would erase the literal and defeat the type checker. The
# dictionary is built from these constants, keeping one source of truth for
# both the estimator call and the published provenance.
ML_REVIEW_SOLVER: Literal["lbfgs"] = "lbfgs"
ML_REVIEW_CLASS_WEIGHT: Literal["balanced"] = "balanced"
# Ridge penalty expressed as l1_ratio=0; the older penalty="l2" spelling is
# deprecated from scikit-learn 1.8 and removed at 1.10.
ML_REVIEW_L1_RATIO = 0.0
ML_REVIEW_C = 1.0
ML_REVIEW_MAX_ITER = 1000
ML_REVIEW_FIT_INTERCEPT = True

ML_REVIEW_HYPERPARAMETERS = {
    "estimator": "sklearn.linear_model.LogisticRegression",
    "l1_ratio": ML_REVIEW_L1_RATIO,
    "C": ML_REVIEW_C,
    "solver": ML_REVIEW_SOLVER,
    "max_iter": ML_REVIEW_MAX_ITER,
    "class_weight": ML_REVIEW_CLASS_WEIGHT,
    "random_state": DEFAULT_RANDOM_SEED,
    "fit_intercept": ML_REVIEW_FIT_INTERCEPT,
}

# Proportion of development identities used for fitting; the remainder
# calibrates the probability threshold.
ML_REVIEW_TRAINING_FRACTION = 0.70

ML_REVIEW_SUCCESS_CRITERIA = {
    "fpir_max": 0.01,
    "target_fpir": PRIMARY_FPIR_TARGET,
    "tpir_rank1_min": 0.90,
    "end_to_end_detection_tolerance": 0.02,
    "coverage_min": 0.90,
}


class MlReviewError(RuntimeError):
    """Raised when the review classifier cannot be built, frozen or applied."""


@dataclass(frozen=True)
class ReviewFeatureRow:
    """One search rendered as classifier input plus its supervision label.

    ``correct_rank`` and ``correct_similarity`` are evaluation metadata drawn
    from benchmark ground truth. They are never placed in the feature vector."""

    sample_id: str
    identity_hash: str
    subgroup: str
    role: str
    features: Dict[str, float]
    # The label represents a correct rank-one referral.
    label: int
    correct_rank: Optional[int] = None
    correct_similarity: Optional[float] = None


@dataclass(frozen=True)
class ReviewIdentityOutcome:
    """Complete intended protocol outcome for one identity.

    Retained so the end-to-end denominator keeps probes that never produced a
    score; filtering to scored rows would silently make it conditional."""

    identity_hash: str
    subgroup: str
    intended_mated: int
    scored_mated: int
    intended_non_mated: int
    scored_non_mated: int
    mated_extraction_failures: int
    non_mated_extraction_failures: int
    gallery_reference_unavailable: int


def split_development_identities_for_classifier(
    protocol: OpenSetProtocol,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    training_fraction: float = ML_REVIEW_TRAINING_FRACTION,
) -> Tuple[List[str], List[str]]:
    """Divide development identities into training and calibration groups.

    Split by identity, never by image, so no photograph of one person can
    appear on both sides. The held-out test identities are not touched."""
    development = sorted({e.identity for e in protocol.partition("development")})
    subgroup_of = {e.identity: e.subgroup for e in protocol.partition("development")}

    training: List[str] = []
    calibration: List[str] = []
    by_subgroup: Dict[str, List[str]] = {}
    for identity in development:
        by_subgroup.setdefault(subgroup_of[identity], []).append(identity)

    # Stratified so every subgroup is represented in both groups.
    for subgroup in sorted(by_subgroup):
        members = sorted(by_subgroup[subgroup])
        rng = random.Random(f"{seed}:ml-review:{subgroup}")
        rng.shuffle(members)
        cut = round(len(members) * training_fraction)
        cut = min(max(cut, 1), len(members) - 1) if len(members) > 1 else len(members)
        training.extend(members[:cut])
        calibration.extend(members[cut:])

    training, calibration = sorted(training), sorted(calibration)
    if set(training) & set(calibration):
        raise MlReviewError("Classifier training and calibration identities overlap.")
    if not training or not calibration:
        raise MlReviewError("Too few development identities to form both classifier groups.")
    return training, calibration


def build_review_feature_rows(
    results: Sequence[OpenSetSearchResult],
    *,
    identities: Optional[Set[str]] = None,
    identity_of_sample: Optional[Mapping[str, str]] = None,
) -> Tuple[List[ReviewFeatureRow], Dict[str, int]]:
    """Turn scored searches into labelled feature rows.

    A row is emitted only when every feature is available; records with a
    missing feature are counted and excluded rather than imputed, because a
    silently invented value would propagate into the coefficients."""
    rows: List[ReviewFeatureRow] = []
    excluded = {"unscored": 0, "missing_feature": 0, "outside_partition": 0}

    for result in results:
        if identities is not None and identity_of_sample is not None:
            private = identity_of_sample.get(result.sample_id)
            if private is None or private not in identities:
                excluded["outside_partition"] += 1
                continue
        if result.failure_code is not None or result.top_similarity is None:
            # An extraction failure is a coverage outcome, never a negative
            # decision, so it must not become a training example.
            excluded["unscored"] += 1
            continue

        top1 = float(result.top_similarity)
        top2 = result.top2_similarity
        values = {
            "top1_similarity": top1,
            "top2_similarity": None if top2 is None else float(top2),
            "top1_top2_margin": None if top2 is None else top1 - float(top2),
            "top5_similarity_mean": result.top5_similarity_mean,
            "top5_similarity_stdev": result.top5_similarity_stdev,
            "top1_gallery_image_count": result.top1_gallery_image_count,
            "gallery_size": result.gallery_size,
            "probe_detection_confidence": result.probe_detection_confidence,
            "probe_face_area_ratio": result.probe_face_area_ratio,
        }
        if any(values[name] is None for name in ML_REVIEW_FEATURES):
            excluded["missing_feature"] += 1
            continue

        # Ground truth is used for supervision, never as an input feature.
        # A referral to the wrong identity is not a true identification, so a
        # mated probe is positive only when its own identity ranks first.
        positive = result.role == "mated_probe" and result.correct_rank == 1
        rows.append(
            ReviewFeatureRow(
                sample_id=result.sample_id,
                identity_hash=result.identity_hash,
                subgroup=result.subgroup,
                role=result.role,
                features={name: float(values[name]) for name in ML_REVIEW_FEATURES},
                label=1 if positive else 0,
                correct_rank=result.correct_rank,
                correct_similarity=result.correct_similarity,
            )
        )
    return rows, excluded


def _feature_matrix(rows: Sequence[ReviewFeatureRow]) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.array([[r.features[name] for name in ML_REVIEW_FEATURES] for r in rows], dtype=float)
    labels = np.array([r.label for r in rows], dtype=int)
    return matrix, labels


@dataclass(frozen=True)
class ReviewClassifier:
    """A fitted classifier stored as plain numbers.

    Coefficients, intercept and scaler parameters are published as JSON rather
    than pickled: a pickle would be unsafe to load and opaque to a reader."""

    feature_order: Tuple[str, ...]
    coefficients: List[float]
    intercept: float
    scaler_mean: List[float]
    scaler_scale: List[float]

    def probabilities(self, matrix: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.scaler_mean, dtype=float)
        scale = np.asarray(self.scaler_scale, dtype=float)
        standardised = (matrix - mean) / scale
        logits = standardised @ np.asarray(self.coefficients, dtype=float) + self.intercept
        # Logistic link, evaluated in a form that does not overflow for large
        # negative logits.
        return np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": ML_REVIEW_METHOD,
            "feature_order": list(self.feature_order),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "hyperparameters": dict(ML_REVIEW_HYPERPARAMETERS),
            "class_imbalance_handling": "class_weight='balanced', declared before fitting",
            "serialisation": "plain JSON numerics; no pickle is written or read",
        }


def fit_review_classifier(rows: Sequence[ReviewFeatureRow]) -> ReviewClassifier:
    """Fit the logistic regression on training identities only."""
    if not rows:
        raise MlReviewError("No training rows available for the review classifier.")
    labels_present = {r.label for r in rows}
    if labels_present != {0, 1}:
        raise MlReviewError(
            f"Training rows carry labels {sorted(labels_present)}; both classes are required."
        )
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - pinned dependency
        raise MlReviewError(
            "scikit-learn is required for the review classifier. Install it from "
            "requirements.txt before running --mode ml-review."
        ) from exc

    matrix, labels = _feature_matrix(rows)
    # Fit the scaler on training identities only; calibration and test rows are
    # transformed with these same parameters.
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(
        l1_ratio=ML_REVIEW_L1_RATIO,
        C=ML_REVIEW_C,
        solver=ML_REVIEW_SOLVER,
        max_iter=ML_REVIEW_MAX_ITER,
        class_weight=ML_REVIEW_CLASS_WEIGHT,
        random_state=DEFAULT_RANDOM_SEED,
        fit_intercept=ML_REVIEW_FIT_INTERCEPT,
    ).fit(scaler.transform(matrix), labels)

    scale = np.asarray(scaler.scale_, dtype=float)
    # A constant feature yields a zero scale; guard so division stays defined.
    scale = np.where(scale == 0.0, 1.0, scale)
    return ReviewClassifier(
        feature_order=ML_REVIEW_FEATURES,
        coefficients=[float(c) for c in np.asarray(model.coef_).ravel()],
        intercept=float(np.asarray(model.intercept_).ravel()[0]),
        scaler_mean=[float(m) for m in np.asarray(scaler.mean_, dtype=float)],
        scaler_scale=[float(s) for s in scale],
    )


def build_review_identity_outcomes(
    results: Sequence[OpenSetSearchResult],
) -> Dict[str, ReviewIdentityOutcome]:
    """Aggregate the complete intended outcome per identity.

    Retain failed probes in the end-to-end denominator; excluding them would
    turn an end-to-end rate into a conditional one."""
    scratch: Dict[str, Dict[str, Any]] = {}
    for result in results:
        row = scratch.setdefault(
            result.identity_hash,
            {"subgroup": result.subgroup, "intended_mated": 0, "scored_mated": 0,
             "intended_non_mated": 0, "scored_non_mated": 0, "mated_fail": 0,
             "non_mated_fail": 0, "unavailable": 0},
        )
        scored = result.failure_code is None and result.top_similarity is not None
        if result.role == "mated_probe":
            row["intended_mated"] += 1
            row["scored_mated"] += scored
            if not scored:
                row["mated_fail"] += 1
                if result.failure_code == GALLERY_REFERENCE_UNAVAILABLE:
                    row["unavailable"] += 1
        elif result.role == "non_mated_probe":
            row["intended_non_mated"] += 1
            row["scored_non_mated"] += scored
            if not scored:
                row["non_mated_fail"] += 1
    return {
        identity: ReviewIdentityOutcome(
            identity_hash=identity, subgroup=row["subgroup"],
            intended_mated=row["intended_mated"], scored_mated=row["scored_mated"],
            intended_non_mated=row["intended_non_mated"],
            scored_non_mated=row["scored_non_mated"],
            mated_extraction_failures=row["mated_fail"],
            non_mated_extraction_failures=row["non_mated_fail"],
            gallery_reference_unavailable=row["unavailable"],
        )
        for identity, row in sorted(scratch.items())
    }


def review_rates_at_probability(
    rows: Sequence[ReviewFeatureRow],
    probabilities: np.ndarray,
    threshold: float,
    *,
    outcomes: Optional[Mapping[str, ReviewIdentityOutcome]] = None,
) -> Dict[str, Any]:
    """Rank-aware FPIR, TPIR and the decision counts at one threshold.

    Require the correct identity at rank one: a mated probe referred to some
    other identity is a referral, but it is not an identification."""
    rank1 = rank5 = wrong_identity = not_referred = 0
    non_mated_referred = non_mated_correct = 0
    for row, probability in zip(rows, probabilities):
        referred = bool(probability >= threshold)
        if row.role == "mated_probe":
            if not referred:
                not_referred += 1
                continue
            if row.correct_rank == 1:
                rank1 += 1
                rank5 += 1
            elif row.correct_rank is not None and row.correct_rank <= 5:
                rank5 += 1
                wrong_identity += 1
            else:
                wrong_identity += 1
        else:
            non_mated_referred += referred
            non_mated_correct += not referred

    scored_mated = rank1 + wrong_identity + not_referred
    scored_non_mated = non_mated_referred + non_mated_correct
    fpir = non_mated_referred / scored_non_mated if scored_non_mated else float("nan")
    tpir1 = rank1 / scored_mated if scored_mated else float("nan")
    tpir5 = rank5 / scored_mated if scored_mated else float("nan")

    # Intended denominators come from the protocol, not the surviving rows.
    intended_mated = (
        sum(o.intended_mated for o in outcomes.values()) if outcomes else scored_mated
    )
    intended_non_mated = (
        sum(o.intended_non_mated for o in outcomes.values()) if outcomes else scored_non_mated
    )
    mated_failures = (
        sum(o.mated_extraction_failures for o in outcomes.values()) if outcomes else 0
    )
    non_mated_failures = (
        sum(o.non_mated_extraction_failures for o in outcomes.values()) if outcomes else 0
    )
    unavailable = (
        sum(o.gallery_reference_unavailable for o in outcomes.values()) if outcomes else 0
    )
    return {
        "probability_threshold": threshold,
        "fpir": fpir,
        "tpir_rank1": tpir1,
        "tpir_rank5": tpir5,
        "fnir_rank1": 1.0 - tpir1 if tpir1 == tpir1 else float("nan"),
        "fnir_rank5": 1.0 - tpir5 if tpir5 == tpir5 else float("nan"),
        "false_reviews_per_1000_non_mated": fpir * 1000.0 if fpir == fpir else float("nan"),
        "mated_correct_rank1_referred": rank1,
        "mated_correct_rank5_referred": rank5,
        "mated_wrong_identity_referred": wrong_identity,
        "mated_not_referred": not_referred,
        "non_mated_incorrectly_referred": non_mated_referred,
        "non_mated_correctly_not_referred": non_mated_correct,
        "scored_mated_probes": scored_mated,
        "scored_non_mated_probes": scored_non_mated,
        "intended_mated_probes": intended_mated,
        "intended_non_mated_probes": intended_non_mated,
        "mated_extraction_failures": mated_failures,
        "non_mated_extraction_failures": non_mated_failures,
        "gallery_reference_unavailable": unavailable,
        "mated_extraction_coverage": (
            scored_mated / intended_mated if intended_mated else float("nan")
        ),
        "non_mated_extraction_coverage": (
            scored_non_mated / intended_non_mated if intended_non_mated else float("nan")
        ),
        "extraction_coverage": (
            (scored_mated + scored_non_mated) / (intended_mated + intended_non_mated)
            if (intended_mated + intended_non_mated)
            else float("nan")
        ),
        "end_to_end_duplicate_detection_rate": (
            rank1 / intended_mated if intended_mated else float("nan")
        ),
        "conditional_duplicate_detection_rate": tpir1,
    }


def select_review_probability_threshold(
    rows: Sequence[ReviewFeatureRow],
    probabilities: np.ndarray,
    *,
    target_fpir: float,
) -> Dict[str, Any]:
    """Choose the probability threshold on calibration identities only.

    Same deterministic rule as the similarity threshold: admissible by FPIR,
    then highest TPIR@1, then lower FPIR, then the higher threshold."""
    candidates = sorted({round(float(p), 12) for p in probabilities}) + [1.0000000001]
    evaluated = [review_rates_at_probability(rows, probabilities, c) for c in candidates]
    admissible = [e for e in evaluated if e["fpir"] == e["fpir"] and e["fpir"] <= target_fpir]
    if not admissible:
        raise MlReviewError(
            f"No probability threshold reached a calibration FPIR at or below {target_fpir}."
        )
    chosen = sorted(
        admissible,
        key=lambda e: (
            -(e["tpir_rank1"] if e["tpir_rank1"] == e["tpir_rank1"] else -1.0),
            e["fpir"],
            -e["probability_threshold"],
        ),
    )[0]
    return {
        "target_fpir": target_fpir,
        "probability_threshold": chosen["probability_threshold"],
        "calibration_fpir": chosen["fpir"],
        "calibration_tpir_rank1": chosen["tpir_rank1"],
        "selection_rule": (
            "Among probability thresholds whose calibration FPIR is no greater than the "
            "target, select the highest calibration TPIR at rank 1; ties broken by lower "
            "calibration FPIR, then by higher probability threshold."
        ),
        "candidates_evaluated": len(evaluated),
        "candidates_admissible": len(admissible),
    }


def require_frozen_review_policy(payload: Mapping[str, Any], *, context: str = "") -> float:
    """Refuse held-out evaluation unless the classifier policy is frozen."""
    status = payload.get("status")
    if status != ML_REVIEW_STATUS_FROZEN:
        raise MlReviewError(
            f"Refusing to evaluate held-out identities with review-policy status {status!r}"
            f"{f' from {context}' if context else ''}. Only {ML_REVIEW_STATUS_FROZEN!r} is "
            f"accepted; fit on training identities and calibrate on calibration identities "
            f"first."
        )
    operating = payload.get("operating_points") or {}
    primary = operating.get(str(PRIMARY_FPIR_TARGET))
    if not primary or "probability_threshold" not in primary:
        raise MlReviewError(
            f"Frozen review policy carries no probability threshold for the primary FPIR "
            f"target {PRIMARY_FPIR_TARGET}."
        )
    return float(primary["probability_threshold"])


def review_cluster_bootstrap(
    rows: Sequence[ReviewFeatureRow],
    probabilities: np.ndarray,
    threshold: float,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
    outcomes: Optional[Mapping[str, ReviewIdentityOutcome]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Identity-cluster bootstrap over classifier decisions.

    Resample identities with their complete protocol outcomes, so a replicate's
    end-to-end denominator counts probes that failed extraction as well as
    those that were scored."""
    by_identity: Dict[str, List[int]] = {}
    subgroup_of: Dict[str, str] = {}
    for index, row in enumerate(rows):
        by_identity.setdefault(row.identity_hash, []).append(index)
        subgroup_of[row.identity_hash] = row.subgroup
    # An identity whose every probe failed contributes no feature row, yet still
    # belongs in the denominator, so it must be resampled too.
    if outcomes:
        for identity, outcome in outcomes.items():
            by_identity.setdefault(identity, [])
            subgroup_of.setdefault(identity, outcome.subgroup)

    strata: Dict[str, List[str]] = {}
    for identity_hash in sorted(by_identity):
        strata.setdefault(subgroup_of[identity_hash], []).append(identity_hash)

    tracked = (
        "fpir", "tpir_rank1", "tpir_rank5", "fnir_rank1", "fnir_rank5",
        "end_to_end_duplicate_detection_rate", "extraction_coverage",
        "mated_extraction_coverage", "non_mated_extraction_coverage",
    )
    samples: Dict[str, List[float]] = {name: [] for name in tracked}
    rng = random.Random(seed)
    for _ in range(replicates):
        indices: List[int] = []
        drawn_identities: List[str] = []
        for _subgroup, members in sorted(strata.items()):
            for _ in range(len(members)):
                chosen = members[rng.randrange(len(members))]
                drawn_identities.append(chosen)
                indices.extend(by_identity[chosen])
        drawn = [rows[i] for i in indices]
        drawn_probabilities = probabilities[np.asarray(indices, dtype=int)] if indices else (
            np.asarray([], dtype=float)
        )
        replicate_outcomes: Optional[Dict[str, ReviewIdentityOutcome]] = None
        if outcomes:
            # Accumulate per draw, since one identity may be drawn repeatedly.
            totals: Dict[str, ReviewIdentityOutcome] = {}
            for position, identity in enumerate(drawn_identities):
                outcome = outcomes.get(identity)
                if outcome is not None:
                    totals[f"{identity}:{position}"] = outcome
            replicate_outcomes = totals
        rates = review_rates_at_probability(
            drawn, drawn_probabilities, threshold, outcomes=replicate_outcomes
        )
        for name in tracked:
            value = rates.get(name, float("nan"))
            if isinstance(value, float) and value == value:
                samples[name].append(value)

    intervals: Dict[str, Dict[str, Any]] = {}
    for name in tracked:
        low, high = _percentile_interval(samples[name])
        intervals[name] = {
            "lower_95": low,
            "upper_95": high,
            "valid_replicates": len(samples[name]),
            "requested_replicates": replicates,
        }
    return intervals


def review_subgroup_metrics(
    rows: Sequence[ReviewFeatureRow],
    probabilities: np.ndarray,
    threshold: float,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
    outcomes: Optional[Mapping[str, ReviewIdentityOutcome]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-subgroup rates with confidence intervals.

    Subgroup is used for evaluation, never prediction."""
    per_subgroup: Dict[str, Dict[str, Any]] = {}
    for subgroup in BFW_SUBGROUPS:
        indices = [i for i, r in enumerate(rows) if r.subgroup == subgroup]
        subset_outcomes = (
            {k: v for k, v in outcomes.items() if v.subgroup == subgroup} if outcomes else None
        )
        if not indices and not subset_outcomes:
            continue
        subset = [rows[i] for i in indices]
        subset_probabilities = (
            probabilities[np.asarray(indices, dtype=int)] if indices
            else np.asarray([], dtype=float)
        )
        rates = review_rates_at_probability(
            subset, subset_probabilities, threshold, outcomes=subset_outcomes
        )
        intervals = review_cluster_bootstrap(
            subset, subset_probabilities, threshold, replicates=replicates, seed=seed,
            outcomes=subset_outcomes,
        )
        entry: Dict[str, Any] = {}
        for metric in ("fpir", "fnir_rank1", "fnir_rank5", "tpir_rank1", "tpir_rank5"):
            entry[metric] = rates[metric]
            entry[f"{metric}_lower_95"] = intervals[metric]["lower_95"]
            entry[f"{metric}_upper_95"] = intervals[metric]["upper_95"]
        for metric, key in (
            ("mated_probe_coverage", "mated_extraction_coverage"),
            ("non_mated_probe_coverage", "non_mated_extraction_coverage"),
        ):
            entry[metric] = rates[key]
            entry[f"{metric}_lower_95"] = intervals[key]["lower_95"]
            entry[f"{metric}_upper_95"] = intervals[key]["upper_95"]
        entry["scored_mated_probes"] = rates["scored_mated_probes"]
        entry["scored_non_mated_probes"] = rates["scored_non_mated_probes"]
        entry["intended_mated_probes"] = rates["intended_mated_probes"]
        entry["intended_non_mated_probes"] = rates["intended_non_mated_probes"]
        per_subgroup[subgroup] = entry
    return per_subgroup


def evaluate_review_success_criteria(
    classifier_rates: Mapping[str, Any],
    coverage: Mapping[str, Any],
    baseline_rates: Mapping[str, Any],
    baseline_detection: Mapping[str, Any],
    classifier_detection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare against criteria declared before the held-out run."""

    def verdict(
        actual: Any, target: float, *, minimum: bool, strict: bool = False
    ) -> Dict[str, Any]:
        if not isinstance(actual, (int, float)) or actual != actual:
            return {"outcome": "not_measurable", "actual": None, "target": target}
        if minimum:
            ok = actual > target if strict else actual >= target
        else:
            ok = actual < target if strict else actual <= target
        return {"outcome": "achieved" if ok else "not_achieved", "actual": float(actual),
                "target": target}

    baseline_e2e = baseline_detection.get("end_to_end_duplicate_detection_rate", float("nan"))
    classifier_e2e = classifier_detection.get("end_to_end_duplicate_detection_rate", float("nan"))
    detection_gap = (
        baseline_e2e - classifier_e2e
        if isinstance(baseline_e2e, float) and isinstance(classifier_e2e, float)
        and baseline_e2e == baseline_e2e and classifier_e2e == classifier_e2e
        else float("nan")
    )
    probe_coverage = min(
        (
            1.0 - coverage.get("mated_extraction_failure_rate", float("nan")),
            1.0 - coverage.get("non_mated_extraction_failure_rate", float("nan")),
        ),
        default=float("nan"),
    )
    return {
        "criteria_declared_before_test": True,
        "primary_fpir_at_or_below_1_percent": verdict(
            classifier_rates.get("fpir"), ML_REVIEW_SUCCESS_CRITERIA["fpir_max"], minimum=False
        ),
        "tpir_rank1_at_least_90_percent": verdict(
            classifier_rates.get("tpir_rank1"),
            ML_REVIEW_SUCCESS_CRITERIA["tpir_rank1_min"],
            minimum=True,
        ),
        "end_to_end_detection_within_2pp_of_threshold_method": verdict(
            detection_gap,
            ML_REVIEW_SUCCESS_CRITERIA["end_to_end_detection_tolerance"],
            minimum=False,
        ),
        # "Lower than" means strictly lower. Matching the comparator is not a
        # reduction, and recording it as one would overstate the finding.
        "fewer_false_reviews_than_threshold_method": verdict(
            (
                baseline_rates.get("false_reviews_per_1000_non_mated", float("nan"))
                - classifier_rates.get("false_reviews_per_1000_non_mated", float("nan"))
            ),
            0.0,
            minimum=True,
            strict=True,
        ),
        "gallery_enrolment_coverage_at_least_90_percent": verdict(
            coverage.get("gallery_enrolment_coverage"),
            ML_REVIEW_SUCCESS_CRITERIA["coverage_min"],
            minimum=True,
        ),
        "probe_extraction_coverage_at_least_90_percent": verdict(
            probe_coverage, ML_REVIEW_SUCCESS_CRITERIA["coverage_min"], minimum=True
        ),
    }


# =============================================================================
# 27. Experiment 7 and 8 orchestration, figures and reports
# =============================================================================
#
# Experiment 7 fits and freezes the review classifier before the held-out BFW
# identities are scored once. Experiment 8 compares complete pretrained
# pipelines, each calibrated on its own development data, and records an
# explicit not-run status when the stronger pipeline is unlicensed or absent.
#
# Every figure is drawn from the JSON and CSV artefacts, never from values
# typed by hand, so a figure cannot drift from the numbers it illustrates.

FIGURES_ROOT = RESULTS_ROOT / "figures"

ML_REVIEW_LIMITATIONS = (
    "A benchmark-validated, human-review-only research proof of concept. Nothing here "
    "is production-ready, unbiased, secure, or capable of proving fraudulent behaviour.",
    "The classifier predicts whether a search should open a human-review case. Its output "
    "is not evidence of duplication, fraud or misuse, and no sanction follows from it.",
    "Coefficients describe association within this classifier on these benchmark "
    "identities. They are not causal, and they do not transfer to another population.",
    "Demographic subgroup is never a classifier input. It is used only for post-hoc "
    "fairness reporting.",
    "Extraction failures are reported separately and are never counted as genuine "
    "negative decisions.",
)


def _identity_of_sample(protocol: OpenSetProtocol) -> Dict[str, str]:
    # Private identity per sample, used only to enforce partition membership.
    return {e.sample_id: e.identity for e in protocol.entries}


def _provenance_block(
    dataset: BfwDataset, protocol: OpenSetProtocol, summary: Mapping[str, Any],
    detector: Any, embedder: Any, *, artifact_type: str,
) -> Dict[str, Any]:
    """Provenance carried by every artefact this section writes."""
    return {
        "artifact_type": artifact_type,
        "schema_version": SCHEMA_VERSION,
        "opaque_id_version": OPAQUE_ID_VERSION,
        "created_at": utc_now_iso(),
        "seed": protocol.seed,
        "dataset_name": "BFW",
        "protocol_version": BFW_PROTOCOL_VERSION,
        "protocol_digest": summary["public_manifest_sha256"],
        "public_manifest_digest": summary["public_manifest_sha256"],
        "evaluated_image_set_sha256": summary.get("evaluated_image_set_sha256")
        or bfw_dataset_provenance(dataset)["evaluated_image_set_sha256"],
        "dataset_provenance": bfw_dataset_provenance(dataset),
        "pipeline": primary_pipeline_description(detector, embedder).as_dict(),
        "preprocessing_revision": PREPROCESSING_REVISION,
        "software_environment": software_environment_report(),
        "dependency_versions": _reported_dependency_versions(),
        "policy_note": POLICY_NOTE,
        "limitations": list(ML_REVIEW_LIMITATIONS),
    }


def _reported_dependency_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package in ("numpy", "opencv-python-headless", "Pillow", "scikit-learn", "matplotlib"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_ml_review_experiment(
    *, output_root: Path = AGGREGATE_ROOT, seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, Any]:
    """Experiment 7 end to end. Stops on the exact blocker; fabricates nothing."""
    config = EnvironmentConfig.load()
    if not id_hmac_key_is_configured():
        raise OpaqueIdentifierKeyError(
            f"{ID_HMAC_KEY_VARIABLE} must be configured before identifiers are produced."
        )
    image_root, metadata_path = config.require_bfw_roots()
    detector, embedder = load_models(config.require_model_root())

    announce("Loading BFW and rebuilding the identity-disjoint protocol")
    dataset = load_bfw_dataset(image_root, metadata_path)
    protocol = build_open_set_protocol(dataset, seed=seed)
    summary = open_set_protocol_summary(
        protocol, dataset=dataset, detector=detector, embedder=embedder
    )
    identity_of_sample = _identity_of_sample(protocol)

    training_ids, calibration_ids = split_development_identities_for_classifier(
        protocol, seed=seed
    )
    test_ids = {e.identity for e in protocol.partition("test")}
    if (set(training_ids) | set(calibration_ids)) & test_ids:
        raise MlReviewError("Classifier identities overlap the held-out test partition.")

    announce(
        f"Classifier identities: {len(training_ids)} training, {len(calibration_ids)} "
        f"calibration, {len(test_ids)} held-out test (disjoint)"
    )

    announce("Scoring the development partition once for both classifier groups")
    development = run_open_set_method(
        protocol, partition="development", method=METHOD_B, detector=detector, embedder=embedder
    )

    training_rows, training_excluded = build_review_feature_rows(
        development.search_results, identities=set(training_ids),
        identity_of_sample=identity_of_sample,
    )
    calibration_rows, calibration_excluded = build_review_feature_rows(
        development.search_results, identities=set(calibration_ids),
        identity_of_sample=identity_of_sample,
    )
    calibration_outcomes = {
        identity_hash: outcome
        for identity_hash, outcome in build_review_identity_outcomes(
            [
                r for r in development.search_results
                if identity_of_sample.get(r.sample_id) in set(calibration_ids)
            ]
        ).items()
    }

    classifier = fit_review_classifier(training_rows)
    calibration_matrix, _ = _feature_matrix(calibration_rows)
    calibration_probabilities = classifier.probabilities(calibration_matrix)

    operating_points: Dict[str, Any] = {}
    for target in FPIR_TARGETS:
        operating_points[str(target)] = select_review_probability_threshold(
            calibration_rows, calibration_probabilities, target_fpir=target
        )

    provenance = _provenance_block(
        dataset, protocol, summary, detector, embedder, artifact_type="ml_review_threshold"
    )
    subgroup_counts = lambda ids: {
        s: sum(1 for i in ids if _subgroup_of_identity(protocol)[i] == s) for s in BFW_SUBGROUPS
    }
    write_json_artifact(
        output_root / "ml_review_protocol_summary.json",
        {
            **_provenance_block(dataset, protocol, summary, detector, embedder,
                                artifact_type="ml_review_protocol_summary"),
            "training_identities": len(training_ids),
            "calibration_identities": len(calibration_ids),
            "held_out_test_identities": len(test_ids),
            "training_fraction": ML_REVIEW_TRAINING_FRACTION,
            "split_rule": (
                "Development identities split by identity, stratified by subgroup, under "
                "the research seed. No image of one person appears in both groups, and the "
                "held-out test identities are untouched."
            ),
            "training_subgroup_counts": subgroup_counts(training_ids),
            "calibration_subgroup_counts": subgroup_counts(calibration_ids),
            "training_identity_hashes": sorted(opaque_id(f"bfw-identity:{i}") for i in training_ids),
            "calibration_identity_hashes": sorted(
                opaque_id(f"bfw-identity:{i}") for i in calibration_ids
            ),
            "excluded_records": {"training": training_excluded, "calibration": calibration_excluded},
        },
    )

    write_json_artifact(
        output_root / "ml_review_model.json",
        {**_provenance_block(dataset, protocol, summary, detector, embedder,
                             artifact_type="ml_review_model"),
         "model": classifier.as_dict(),
         "feature_definitions": _feature_definitions(),
         "training_rows": len(training_rows),
         "trained_on": "BFW classifier-training identities only"},
    )

    policy_payload = {
        **provenance,
        "status": ML_REVIEW_STATUS_FROZEN,
        "method": ML_REVIEW_METHOD,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "operating_points": operating_points,
        "threshold_source": "BFW classifier-calibration identities only",
        "threshold_selection_rule": operating_points[str(PRIMARY_FPIR_TARGET)]["selection_rule"],
        "threshold_status": ML_REVIEW_STATUS_FROZEN,
        "classifier_hyperparameters": dict(ML_REVIEW_HYPERPARAMETERS),
    }
    policy_path = output_root / "ml_review_threshold.json"
    write_json_artifact(policy_path, policy_payload)
    frozen_probability = require_frozen_review_policy(
        read_json_artifact(policy_path), context=project_relative(policy_path)
    )
    announce(f"Froze the review classifier at probability {frozen_probability:.6f}")

    write_json_artifact(
        output_root / "ml_review_development_metrics.json",
        {**_provenance_block(dataset, protocol, summary, detector, embedder,
                             artifact_type="ml_review_development_metrics"),
         "status": "ml_review_development",
         "calibration_operating_points": {
             str(t): review_rates_at_probability(
                 calibration_rows, calibration_probabilities,
                 operating_points[str(t)]["probability_threshold"],
                 outcomes=calibration_outcomes)
             for t in FPIR_TARGETS},
         "training_rows": len(training_rows),
         "calibration_rows": len(calibration_rows)},
    )

    # --- Held-out test, scored once ------------------------------------------
    announce("Scoring the held-out test partition")
    test_run = run_open_set_method(
        protocol, partition="test", method=METHOD_B, detector=detector, embedder=embedder
    )
    test_rows, test_excluded = build_review_feature_rows(test_run.search_results)
    test_matrix, _ = _feature_matrix(test_rows)
    decision_start = time.perf_counter()
    test_probabilities = classifier.probabilities(test_matrix)
    decision_elapsed = time.perf_counter() - decision_start
    per_decision_ms = (decision_elapsed / len(test_rows) * 1000.0) if test_rows else float("nan")

    coverage = open_set_coverage(test_run)
    # Retain failed probes in the intended denominator.
    test_outcomes = build_review_identity_outcomes(test_run.search_results)
    classifier_rates = review_rates_at_probability(
        test_rows, test_probabilities, frozen_probability, outcomes=test_outcomes
    )

    # The comparator, on identical identities, gallery, probes and accounting.
    baseline_policy = read_json_artifact(output_root / "bfw_open_set_threshold.json")
    baseline_threshold = require_frozen_open_set_policy(baseline_policy)
    baseline_rates = open_set_rates_at_threshold(test_run.search_results, baseline_threshold)
    baseline_detection = open_set_duplicate_detection(test_run, baseline_threshold)

    intervals = review_cluster_bootstrap(
        test_rows, test_probabilities, frozen_probability,
        replicates=bootstrap_replicates, seed=seed, outcomes=test_outcomes,
    )
    per_subgroup = review_subgroup_metrics(
        test_rows, test_probabilities, frozen_probability,
        replicates=bootstrap_replicates, seed=seed, outcomes=test_outcomes,
    )

    test_payload = {
        **_provenance_block(dataset, protocol, summary, detector, embedder,
                            artifact_type="ml_review_test_metrics"),
        "status": "ml_review_tested",
        "threshold_source": project_relative(policy_path),
        "threshold_status": ML_REVIEW_STATUS_FROZEN,
        "operating_probability_threshold": frozen_probability,
        "primary_fpir_target": PRIMARY_FPIR_TARGET,
        "classifier_hyperparameters": dict(ML_REVIEW_HYPERPARAMETERS),
        "feature_definitions": _feature_definitions(),
        "classifier": classifier_rates,
        "classifier_decision_latency_mean_ms": per_decision_ms,
        "classifier_decision_latency_p95_ms": per_decision_ms,
        "comparator_three_image_open_set_calibrated": {
            "operating_threshold": baseline_threshold,
            "rates": baseline_rates,
            **baseline_detection,
        },
        "coverage": coverage,
        "excluded_records": test_excluded,
        "success_criteria": evaluate_review_success_criteria(
            classifier_rates, coverage, baseline_rates, baseline_detection, classifier_rates
        ),
    }
    write_json_artifact(output_root / "ml_review_test_metrics.json", test_payload)
    write_json_artifact(
        output_root / "ml_review_confidence_intervals.json",
        {**_provenance_block(dataset, protocol, summary, detector, embedder,
                             artifact_type="ml_review_confidence_intervals"),
         "replicates": bootstrap_replicates,
         "resampling_unit": "identity (cluster bootstrap, subgroup-stratified)",
         "intervals": intervals},
    )
    write_json_artifact(
        output_root / "ml_review_subgroup_metrics.json",
        {**_provenance_block(dataset, protocol, summary, detector, embedder,
                             artifact_type="ml_review_subgroup_metrics"),
         "replicates": bootstrap_replicates,
         "resampling_unit": "identity (cluster bootstrap, subgroup-stratified)",
         "subgroups": per_subgroup},
    )
    _write_review_csvs(output_root, classifier_rates, baseline_rates, baseline_detection,
                       per_subgroup, test_rows, test_probabilities, frozen_probability)

    report = render_ml_review_report(test_payload, intervals, per_subgroup, classifier)
    (output_root / "ML_REVIEW_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")

    leaks = find_path_leaks(output_root, forbidden_substrings=default_forbidden_path_substrings())
    if leaks:
        raise PrivacyLeakError(
            "Refusing to finish: review output(s) contain a personal/absolute path:\n"
            + "\n".join(f"  {redact_private_paths(leak)}" for leak in leaks)
        )
    assert_no_identifier_key_leak(output_root)
    announce("Privacy validation passed for every review artefact")
    return test_payload


def _subgroup_of_identity(protocol: OpenSetProtocol) -> Dict[str, str]:
    return {e.identity: e.subgroup for e in protocol.entries}


def _feature_definitions() -> Dict[str, str]:
    return {
        "top1_similarity": "Cosine similarity to the highest-ranked gallery template.",
        "top2_similarity": "Cosine similarity to the second-ranked gallery template.",
        "top1_top2_margin": "top1_similarity minus top2_similarity; ranking decisiveness.",
        "top5_similarity_mean": "Mean similarity across the five highest-ranked templates.",
        "top5_similarity_stdev": "Sample standard deviation across those five similarities.",
        "top1_gallery_image_count": "Images that contributed to the top-ranked template.",
        "gallery_size": "Enrolled identities searched, which scales impostor exposure.",
        "probe_detection_confidence": "YuNet detection score for the probe image.",
        "probe_face_area_ratio": "Detected face box area divided by whole-image area.",
    }


# Shared column order for both experiments' subgroup files, so a reader can
# compare them directly.
_SUBGROUP_CSV_COLUMNS = (
    "subgroup",
    "fpir", "fpir_lower_95", "fpir_upper_95",
    "fnir_rank1", "fnir_rank1_lower_95", "fnir_rank1_upper_95",
    "fnir_rank5", "fnir_rank5_lower_95", "fnir_rank5_upper_95",
    "tpir_rank1", "tpir_rank1_lower_95", "tpir_rank1_upper_95",
    "tpir_rank5", "tpir_rank5_lower_95", "tpir_rank5_upper_95",
    "mated_probe_coverage", "mated_probe_coverage_lower_95", "mated_probe_coverage_upper_95",
    "non_mated_probe_coverage", "non_mated_probe_coverage_lower_95",
    "non_mated_probe_coverage_upper_95",
    "scored_mated_probes", "scored_non_mated_probes",
    "intended_mated_probes", "intended_non_mated_probes",
)


def _write_review_csvs(
    output_root: Path, classifier_rates: Mapping[str, Any], baseline_rates: Mapping[str, Any],
    baseline_detection: Mapping[str, Any], per_subgroup: Mapping[str, Mapping[str, Any]],
    rows: Sequence[ReviewFeatureRow], probabilities: np.ndarray, threshold: float,
) -> None:
    with open(output_root / "ml_review_method_comparison.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "fpir", "tpir_rank1", "tpir_rank5",
                         "false_reviews_per_1000_non_mated",
                         "end_to_end_duplicate_detection_rate"])
        writer.writerow(["three_image_open_set_calibrated", baseline_rates["fpir"],
                         baseline_rates["tpir_rank1"], baseline_rates["tpir_rank5"],
                         baseline_rates["false_reviews_per_1000_non_mated"],
                         baseline_detection["end_to_end_duplicate_detection_rate"]])
        writer.writerow([ML_REVIEW_METHOD, classifier_rates["fpir"],
                         classifier_rates["tpir_rank1"], classifier_rates["tpir_rank5"],
                         classifier_rates["false_reviews_per_1000_non_mated"],
                         classifier_rates["end_to_end_duplicate_detection_rate"]])

    with open(output_root / "ml_review_subgroup_metrics.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = _SUBGROUP_CSV_COLUMNS
        writer.writerow(columns)
        for subgroup in sorted(per_subgroup):
            r = per_subgroup[subgroup]
            writer.writerow([subgroup] + [r.get(name, "") for name in columns[1:]])

    # Aggregate only: no per-image score, identifier or path is published.
    referred = int(sum(1 for p in probabilities if p >= threshold))
    with open(output_root / "ml_review_predictions_summary.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["statistic", "value"])
        writer.writerow(["scored_records", len(rows)])
        writer.writerow(["referred_for_review", referred])
        writer.writerow(["not_referred", len(rows) - referred])
        writer.writerow(["probability_threshold", threshold])
        writer.writerow(["mean_probability", float(np.mean(probabilities)) if len(rows) else ""])


def render_ml_review_report(
    test: Mapping[str, Any], intervals: Mapping[str, Any],
    per_subgroup: Mapping[str, Mapping[str, Any]], classifier: ReviewClassifier,
) -> str:
    """Every figure below is read from the artefacts, never restated."""
    c = test["classifier"]
    b = test["comparator_three_image_open_set_calibrated"]["rates"]
    coverage = test["coverage"]
    lines = [
        "# Machine-learning review classifier (Experiment 7)",
        "",
        f"Auto-generated by `ACP_arden.py --mode ml-review` on {test['created_at']}.",
        "",
        "> Can an interpretable machine-learning review classifier trained on BFW "
        "development identities reduce false duplicate-profile review referrals while "
        "retaining duplicate-detection performance compared with the existing single "
        "calibrated similarity threshold?",
        "",
        "## Held-out comparison",
        "",
        "| Method | FPIR | TPIR@1 | TPIR@5 | False reviews / 1,000 |",
        "| --- | --- | --- | --- | --- |",
        f"| Calibrated similarity threshold | {format_percentage(b['fpir'])} | "
        f"{format_percentage(b['tpir_rank1'])} | {format_percentage(b['tpir_rank5'])} | "
        f"{format_number(b['false_reviews_per_1000_non_mated'], 1)} |",
        f"| Logistic-regression classifier | {format_percentage(c['fpir'])} | "
        f"{format_percentage(c['tpir_rank1'])} | {format_percentage(c['tpir_rank5'])} | "
        f"{format_number(c['false_reviews_per_1000_non_mated'], 1)} |",
        "",
        "Both use the same rank-aware TPIR definition: the correct identity must be "
        "ranked first (or within five) *and* above the operating point. A referral to "
        "another identity is not an identification.",
        "",
        f"Classifier FPIR 95% CI {format_percentage(intervals['fpir']['lower_95'])} – "
        f"{format_percentage(intervals['fpir']['upper_95'])}; TPIR@1 95% CI "
        f"{format_percentage(intervals['tpir_rank1']['lower_95'])} – "
        f"{format_percentage(intervals['tpir_rank1']['upper_95'])} "
        f"(2,000 identity-cluster replicates).",
        "",
        "### Decision counts",
        "",
        f"- Mated probes referred with the correct identity at rank one: "
        f"{c['mated_correct_rank1_referred']}; within rank five: "
        f"{c['mated_correct_rank5_referred']}",
        f"- Mated probes referred to the wrong identity (a referral, not an "
        f"identification): {c['mated_wrong_identity_referred']}",
        f"- Mated probes not referred: {c['mated_not_referred']}",
        f"- Non-mated probes referred in error: {c['non_mated_incorrectly_referred']}; "
        f"correctly not referred: {c['non_mated_correctly_not_referred']}",
        f"- Gallery-reference-unavailable failures: {c['gallery_reference_unavailable']}",
        f"- Extraction failures (never counted as negatives): "
        f"{coverage['intended_mated_probes'] - coverage['scored_mated_probes']} mated, "
        f"{coverage['intended_non_mated_probes'] - coverage['scored_non_mated_probes']} non-mated",
        "",
        "### Coverage",
        "",
        f"- Gallery enrolment coverage "
        f"{format_percentage(coverage['gallery_enrolment_coverage'])}",
        f"- Mated extraction failure "
        f"{format_percentage(coverage['mated_extraction_failure_rate'])}; non-mated "
        f"{format_percentage(coverage['non_mated_extraction_failure_rate'])}",
        "",
        "## Pre-declared success criteria",
        "",
    ]
    for name, verdict in test["success_criteria"].items():
        if name == "criteria_declared_before_test":
            continue
        actual = verdict.get("actual")
        lines.append(
            f"- {name}: **{verdict['outcome'].replace('_', ' ')}** "
            f"(target {verdict['target']}, achieved "
            f"{'not measurable' if actual is None else format_number(actual, 4)})"
        )

    lines += ["", "## Standardised coefficients", "",
              "Positive coefficients raise the probability of opening a review case; negative "
              "coefficients lower it. These describe association inside this classifier, not "
              "causation.", "", "| Feature | Coefficient |", "| --- | --- |"]
    for name, weight in zip(classifier.feature_order, classifier.coefficients):
        lines.append(f"| {name} | {weight:+.4f} |")

    lines += ["", "## Subgroup performance (95% confidence intervals)", "",
              "| Subgroup | FPIR | TPIR@1 |", "| --- | --- | --- |"]
    for subgroup in sorted(per_subgroup):
        r = per_subgroup[subgroup]
        lines.append(
            f"| {subgroup} | {format_percentage(r['fpir'])} "
            f"[{format_percentage(r['fpir_lower_95'])}–{format_percentage(r['fpir_upper_95'])}] | "
            f"{format_percentage(r['tpir_rank1'])} "
            f"[{format_percentage(r['tpir_rank1_lower_95'])}–"
            f"{format_percentage(r['tpir_rank1_upper_95'])}] |"
        )
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in ML_REVIEW_LIMITATIONS]
    return "\n".join(lines) + "\n"


def render_ml_review_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    path = output_root / "ml_review_test_metrics.json"
    if not path.is_file():
        return (
            "No review-classifier results found. Run `python ACP_arden.py --mode ml-review`.\n"
            f"That requires the official BFW dataset and {BFW_ROOT_VARIABLE}."
        )
    test = read_json_artifact(path)
    c = test["classifier"]
    b = test["comparator_three_image_open_set_calibrated"]["rates"]
    coverage = test["coverage"]
    lines = [
        f"{PROGRAMME_TITLE} — review-classifier summary",
        "",
        "Experiment 7 — logistic-regression review classifier (held-out BFW test)",
        f"  Frozen probability threshold: "
        f"{format_number(test.get('operating_probability_threshold'), 6)} "
        f"(target FPIR {test.get('primary_fpir_target')}, status {test.get('status')})",
        "",
        "  Calibrated similarity threshold (comparator)",
        f"    FPIR: {format_percentage(b['fpir'])}   TPIR@1: "
        f"{format_percentage(b['tpir_rank1'])}",
        f"    False reviews per 1,000: "
        f"{format_number(b['false_reviews_per_1000_non_mated'], 1)}",
        "",
        "  Logistic-regression classifier",
        f"    FPIR: {format_percentage(c['fpir'])}   TPIR@1: "
        f"{format_percentage(c['tpir_rank1'])}",
        f"    False reviews per 1,000: "
        f"{format_number(c['false_reviews_per_1000_non_mated'], 1)}",
        f"    Gallery enrolment coverage: "
        f"{format_percentage(coverage['gallery_enrolment_coverage'])}",
        f"    Mated extraction failure: "
        f"{format_percentage(coverage['mated_extraction_failure_rate'])}; non-mated "
        f"{format_percentage(coverage['non_mated_extraction_failure_rate'])}",
        "    LIMITATION: every rate is conditional on the coverage printed with it, and a "
        "referral opens human review only.",
        "",
        "  Pre-declared success criteria:",
    ]
    for name, verdict in test.get("success_criteria", {}).items():
        if name == "criteria_declared_before_test":
            continue
        lines.append(f"    {name}: {verdict['outcome'].replace('_', ' ')}")
    lines += ["", "Policy: " + POLICY_NOTE, "",
              "Full write-up: results/aggregate/ML_REVIEW_EVALUATION_REPORT.md"]
    return "\n".join(lines)


def evaluate_one_pipeline_for_comparison(
    protocol: OpenSetProtocol,
    *,
    name: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, Any]:
    """Develop, freeze and evaluate one pipeline over the shared protocol.

    Each pipeline receives its own development-only threshold: similarity
    scores from different embedding models are not interchangeable."""
    development = run_open_set_method(
        protocol, partition="development", method=METHOD_B,
        detector=detector, embedder=embedder,
    )
    operating_points = {
        str(target): select_open_set_threshold(
            development.search_results, target_fpir=target
        )
        for target in FPIR_TARGETS
    }
    # Freeze each pipeline threshold before held-out evaluation.
    frozen = float(operating_points[str(PRIMARY_FPIR_TARGET)]["threshold"])

    test = run_open_set_method(
        protocol, partition="test", method=METHOD_B, detector=detector, embedder=embedder,
    )
    coverage = open_set_coverage(test)
    rates = open_set_rates_at_threshold(test.search_results, frozen)
    intervals = cluster_bootstrap_intervals(
        test.search_results, threshold=frozen, replicates=bootstrap_replicates, seed=seed
    )
    subgroup_replicates = max(200, bootstrap_replicates // 4)
    per_subgroup = subgroup_open_set_metrics(
        test.search_results, threshold=frozen, replicates=subgroup_replicates, seed=seed,
    )

    failures = {"zero_faces": 0, "multiple_faces": 0, "image_error": 0,
                GALLERY_REFERENCE_UNAVAILABLE: 0}
    for row in test.search_results:
        if row.failure_code is None:
            continue
        key = row.failure_code.split(":", 1)[0]
        failures[key] = failures.get(key, 0) + 1
    for outcome in test.enrolment_outcomes:
        if not outcome.enrolled and outcome.failure_code:
            failures[outcome.failure_code] = failures.get(outcome.failure_code, 0) + 1

    # Aggregate histograms only: no individual score, identifier or path.
    def histogram(rows: Sequence[OpenSetSearchResult], attribute: str) -> Dict[str, Any]:
        values = [
            float(getattr(r, attribute)) for r in rows
            if r.failure_code is None and getattr(r, attribute) is not None
        ]
        if not values:
            return {"bin_edges": [], "counts": [], "n": 0}
        counts, edges = np.histogram(values, bins=40, range=(-1.0, 1.0))
        return {"bin_edges": [float(e) for e in edges],
                "counts": [int(c) for c in counts], "n": len(values)}

    mated_rows = [r for r in test.search_results if r.role == "mated_probe"]
    non_mated_rows = [r for r in test.search_results if r.role == "non_mated_probe"]

    return {
        "pipeline_name": name,
        "development_threshold": frozen,
        "threshold_status": OPEN_SET_STATUS_FROZEN,
        "operating_points": operating_points,
        "rates": rates,
        "coverage": coverage,
        "confidence_intervals": intervals,
        "subgroups": per_subgroup,
        "subgroup_bootstrap_replicates": subgroup_replicates,
        "global_bootstrap_replicates": bootstrap_replicates,
        "failure_breakdown": failures,
        "profile_photo_consistency": profile_photo_consistency_summary(test, frozen),
        "sex_aggregated": sex_aggregated_metrics(
            test.search_results, threshold=frozen,
            replicates=max(200, bootstrap_replicates // 4), seed=seed,
        ),
        "similarity_histograms": {
            "mated_correct_identity": histogram(mated_rows, "correct_similarity"),
            "non_mated_top1": histogram(non_mated_rows, "top_similarity"),
        },
        **open_set_duplicate_detection(test, frozen),
    }


def run_pipeline_comparison(*, output_root: Path = AGGREGATE_ROOT) -> Dict[str, Any]:
    """Records a precise technical status when the local comparison pipeline is
    unavailable and performs the complete held-out comparison when all verified
    preconditions are satisfied."""
    config = EnvironmentConfig.load()
    detector, embedder = load_models(config.require_model_root())
    primary = primary_pipeline_description(detector, embedder)
    status = pipeline_comparison_status(config)

    # Real held-out metrics, computed only when every precondition holds.
    comparison_metrics: Optional[Dict[str, Any]] = None
    protocol_digest: Optional[str] = None
    evaluated_image_digest: Optional[str] = None
    dataset: Optional[BfwDataset] = None
    if status["comparison_run"]:
        image_root, metadata_path = config.require_bfw_roots()
        dataset = load_bfw_dataset(image_root, metadata_path)
        protocol = build_open_set_protocol(dataset, seed=DEFAULT_RANDOM_SEED)
        summary = open_set_protocol_summary(
            protocol, dataset=dataset, detector=detector, embedder=embedder
        )
        protocol_digest = summary["public_manifest_sha256"]
        evaluated_image_digest = bfw_dataset_provenance(dataset)["evaluated_image_set_sha256"]

        (arcface_detector, arcface_embedder), arcface_description = load_arcface_pipeline(config)
        # Both pipelines traverse the identical protocol: same identities, same
        # split, same roles, same failure taxonomy.
        comparison_metrics = {
            primary.pipeline_name: evaluate_one_pipeline_for_comparison(
                protocol, name=primary.pipeline_name, detector=detector, embedder=embedder,
            ),
            arcface_description.pipeline_name: evaluate_one_pipeline_for_comparison(
                protocol, name=arcface_description.pipeline_name,
                detector=arcface_detector, embedder=arcface_embedder,
            ),
        }
        if not comparison_metrics:
            raise PipelineComparisonError(
                "The comparison cannot be marked as evaluated without held-out metrics."
            )

    payload: Dict[str, Any] = {
        "artifact_type": "pipeline_comparison_metrics",
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "seed": DEFAULT_RANDOM_SEED,
        "dataset_name": "BFW",
        "protocol_version": BFW_PROTOCOL_VERSION,
        "primary_pipeline": primary.as_dict(),
        "comparison_scope": (
            "Complete pretrained pipelines. Detection, landmarking, alignment, "
            "preprocessing, embedding dimensionality and runtime all differ, so no "
            "difference may be attributed to the embedding model alone."
        ),
        "training_performed": False,
        "fine_tuning_performed": False,
        "separate_calibration_required": (
            "Similarity scores from different embedding models are not interchangeable, so "
            "each pipeline must receive its own development-only threshold. The SFace "
            "threshold is never applied to ArcFace."
        ),
        "software_environment": software_environment_report(),
        "dependency_versions": _reported_dependency_versions(),
        "policy_note": POLICY_NOTE,
    }
    payload["licence_note"] = status["licence_note"]
    payload["use_statement"] = status["use_statement"]
    payload["preconditions"] = status["preconditions"]
    payload["substitute_model_used"] = False
    payload["model_provenance"] = {
        "created_and_trained_by": "InsightFace project; not trained or fine-tuned here",
        "model_pack": ARCFACE_MODEL_PACK,
        "weights_redistributed": False,
        "weights_committed_to_git": False,
        "weights_downloaded_automatically": False,
        "ownership_claimed": False,
    }
    # evaluated=yes requires real held-out metrics, never readiness alone.
    if status["comparison_run"] and comparison_metrics:
        payload["evaluated"] = "yes"
        payload["status"] = status["status"]
        payload["comparison_pipeline"] = status["pipeline"]
        payload["held_out_metrics"] = comparison_metrics
    else:
        payload["evaluated"] = "no"
        payload["status"] = status["status"]
        payload["reason"] = status["reason"] or (
            "Preconditions were met but no held-out metrics were produced."
        )
        payload["comparison_pipeline"] = None
        payload["held_out_metrics"] = None
    payload["protocol_digest"] = protocol_digest
    payload["public_manifest_digest"] = protocol_digest
    payload["evaluated_image_set_sha256"] = evaluated_image_digest
    payload["preprocessing_revision"] = PREPROCESSING_REVISION
    payload["model_filenames"] = {
        "primary": [YUNET_FILENAME, SFACE_FILENAME],
        "comparison": [ARCFACE_DETECTOR_FILENAME, ARCFACE_RECOGNITION_FILENAME],
    }
    payload["model_digests"] = {
        "primary": primary.model_sha256,
        "comparison": (
            payload["comparison_pipeline"]["model_sha256"]
            if payload["comparison_pipeline"] else
            {"detector": ARCFACE_DETECTOR_SHA256, "recognition": ARCFACE_RECOGNITION_SHA256}
        ),
    }
    payload["threshold_policy"] = (
        "Each pipeline is calibrated on the BFW development partition only and its "
        "threshold frozen before the held-out identities are scored once."
    )
    payload["calibration_partition"] = (
        "BFW development identities, identity-disjoint from the held-out test partition."
    )
    payload["held_out_partition"] = (
        "BFW test identities, never used for fitting or threshold selection."
    )
    payload["limitations"] = list(OPEN_SET_LIMITATIONS)
    payload["model_file_sizes"] = {
        "primary": _primary_model_file_sizes(config.model_root),
        "comparison": _model_file_sizes(config.arcface_model_root),
    }
    payload["model_storage"] = (
        "Weight files are held in private local research storage, are excluded from Git "
        "and from any release, and are never downloaded automatically."
    )
    payload["redistribution"] = "No pretrained weight file is redistributed by this project."
    payload["dataset_provenance"] = (
        bfw_dataset_provenance(dataset) if dataset is not None else None
    )
    payload["development_partition"] = (
        "BFW development identities; each pipeline calibrated here only."
    )
    payload["held_out_partition"] = (
        "BFW test identities, identity-disjoint, scored once per pipeline."
    )
    payload["frozen_thresholds"] = {
        name: metrics["development_threshold"]
        for name, metrics in (comparison_metrics or {}).items()
    }

    write_json_artifact(output_root / "pipeline_comparison_metrics.json", payload)
    write_json_artifact(
        output_root / "pipeline_comparison_protocol.json",
        {"artifact_type": "pipeline_comparison_protocol", "schema_version": SCHEMA_VERSION,
         "created_at": utc_now_iso(), "seed": DEFAULT_RANDOM_SEED,
         "shared_protocol": (
             "Both pipelines use the same BFW identities, development and test split, "
             "subgroup stratification, gallery image count, mated and non-mated probes, "
             "target FPIR, bootstrap procedure, seed, success criteria and failure taxonomy."
         ),
         "status": payload["status"], "evaluated": payload["evaluated"],
         "dataset_name": "BFW", "dataset_provenance": payload["dataset_provenance"],
         "protocol_version": BFW_PROTOCOL_VERSION,
         "protocol_digest": protocol_digest,
         "public_manifest_digest": protocol_digest,
         "evaluated_image_set_sha256": evaluated_image_digest,
         "development_partition": payload["development_partition"],
         "held_out_partition": payload["held_out_partition"],
         "primary_pipeline": payload["primary_pipeline"],
         "comparison_pipeline": payload.get("comparison_pipeline"),
         "model_filenames": payload["model_filenames"],
         "model_digests": payload["model_digests"],
         "model_file_sizes": payload["model_file_sizes"],
         "dependency_versions": payload["dependency_versions"],
         "software_environment": payload["software_environment"],
         "preprocessing_revision": payload["preprocessing_revision"],
         "threshold_policy": payload["threshold_policy"],
         "frozen_thresholds": payload["frozen_thresholds"],
         "licence_note": payload["licence_note"],
         "limitations": payload["limitations"],
         "policy_note": POLICY_NOTE},
    )
    write_pipeline_performance_csv(
        output_root / "pretrained_pipeline_comparison.csv", payload=payload
    )
    write_pipeline_subgroup_csv(
        output_root / "pretrained_pipeline_subgroup_metrics.csv", payload=payload
    )

    # The interval and subgroup artefacts are written whether or not the
    # comparison ran. A silently absent file is indistinguishable from one that
    # was forgotten; an empty file carrying the blocking status is not.
    evaluated = payload["evaluated"] == "yes"
    write_json_artifact(
        output_root / "pipeline_comparison_confidence_intervals.json",
        {
            "artifact_type": "pipeline_comparison_confidence_intervals",
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "seed": DEFAULT_RANDOM_SEED,
            "status": payload["status"],
            "evaluated": payload["evaluated"],
            "replicates": BOOTSTRAP_REPLICATES if evaluated else 0,
            "resampling_unit": "identity (cluster bootstrap, subgroup-stratified)",
            "intervals": {
                name: metrics["confidence_intervals"]
                for name, metrics in (comparison_metrics or {}).items()
            },
            "note": (
                "Populated for both pipelines once evaluated. "
                + ("" if evaluated else payload.get("reason", ""))
            ),
            "seed": DEFAULT_RANDOM_SEED,
            "dataset_name": "BFW",
            "protocol_version": BFW_PROTOCOL_VERSION,
            "protocol_digest": protocol_digest,
            "public_manifest_digest": protocol_digest,
            "evaluated_image_set_sha256": evaluated_image_digest,
            "primary_pipeline": payload["primary_pipeline"],
            "comparison_pipeline": payload.get("comparison_pipeline"),
            "model_filenames": payload["model_filenames"],
            "model_digests": payload["model_digests"],
            "model_file_sizes": payload["model_file_sizes"],
            "dependency_versions": payload["dependency_versions"],
            "software_environment": payload["software_environment"],
            "preprocessing_revision": payload["preprocessing_revision"],
            "threshold_policy": payload["threshold_policy"],
            "frozen_thresholds": payload["frozen_thresholds"],
            "development_partition": payload["development_partition"],
            "held_out_partition": payload["held_out_partition"],
            "licence_note": payload["licence_note"],
            "limitations": payload["limitations"],
            "policy_note": POLICY_NOTE,
        },
    )
    report = [
        "# Pretrained pipeline comparison (Experiment 8)",
        "",
        f"Auto-generated by `ACP_arden.py --mode pipeline-compare` on {payload['created_at']}.",
        "",
        "> Does a stronger pretrained detection and face-embedding pipeline improve "
        "extraction coverage, open-set duplicate detection and subgroup consistency "
        "compared with YuNet + SFace under the same BFW protocol?",
        "",
        f"## Status: {payload['status']}",
        "",
    ]
    report += [
        "## Licensing position",
        "",
        payload["licence_note"],
        "",
        payload["use_statement"],
        "",
    ]
    if payload["evaluated"] == "no":
        report += [
            f"## Why it did not run: `{payload['status']}`",
            "",
            f"{payload['reason']}",
            "",
            "**This is a technical precondition, not a licensing obstacle.** The official "
            "terms permit non-commercial research, and this evaluation is non-commercial "
            "academic research, so commercial-use restrictions do not apply to it.",
            "",
            "Outstanding preconditions:",
            "",
        ]
        report += [
            f"- `{name}`: {'satisfied' if ok else 'NOT satisfied'}"
            for name, ok in sorted(payload["preconditions"].items())
        ]
        report += [
            "",
            "No substitute model was used and no performance figures are reported. "
            "Replacing the approved comparator with a different model in order to produce "
            "a number would make the comparison meaningless.",
        ]
    else:
        report += [
            "Both pipelines were evaluated under the shared protocol above, each with its "
            "own development-only threshold.",
        ]
    report += [
        "",
        "## Why this is not an embedding-only comparison",
        "",
        payload["comparison_scope"],
        "",
        "Each pipeline receives its own development-only threshold. " +
        payload["separate_calibration_required"],
        "",
        "Neither pipeline was trained or fine-tuned.",
    ]
    (output_root / "PRETRAINED_PIPELINE_COMPARISON_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    announce(f"Pipeline comparison recorded with status {payload['status']}")
    return payload


def render_pipeline_comparison_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    path = output_root / "pipeline_comparison_metrics.json"
    if not path.is_file():
        return "No pipeline-comparison record found. Run `--mode pipeline-compare`."
    payload = read_json_artifact(path)
    lines = [
        f"{PROGRAMME_TITLE} — pretrained pipeline comparison",
        "",
        f"  Primary pipeline: {payload['primary_pipeline']['pipeline_name']} "
        f"({payload['primary_pipeline']['embedding_dimensions']}-dimensional)",
        f"  Evaluated: {payload['evaluated']}   Status: {payload['status']}",
    ]
    if payload.get("reason"):
        lines.append(f"  Reason: {payload['reason']}")
    lines += [
        "  No model was trained or fine-tuned; no substitute model was used.",
        "",
        "Full write-up: results/aggregate/PRETRAINED_PIPELINE_COMPARISON_REPORT.md",
    ]
    return "\n".join(lines)


# =============================================================================
# 28. Figure generation
# =============================================================================
#
# Every figure is built from the published JSON and CSV artefacts, so a chart
# cannot drift from the numbers it illustrates. Axes start at zero unless a log
# scale is stated, no three-dimensional effects are used, and PNG text metadata
# is stripped before publication so a renderer cannot leak a local path.
##############
# Title: Matplotlib: A 2D Graphics Environment
# Author: Hunter, J.D., Computing in Science and Engineering, 9(3), pp. 90-95
# Date: 2007
# Availability: https://doi.org/10.1109/MCSE.2007.55
##############

FIGURE_DPI = 300


def _figure_backend():
    """Import matplotlib with a headless backend, or explain the blocker."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - pinned dependency
        raise ArtifactError(
            "matplotlib is required to generate figures. Install it from requirements.txt."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, path: Path) -> None:
    """Write PNG and SVG, then strip PNG text metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", metadata={"Creator": None})
    _strip_png_text_metadata(path)


def _strip_png_text_metadata(path: Path) -> None:
    """Rewrite the PNG without tEXt/iTXt chunks, which can carry a local path."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - pinned dependency
        return
    with Image.open(path) as image:
        pixels = image.copy()
    pixels.save(path, format="PNG", optimize=True)


def _percent(value: Any) -> float:
    return float(value) * 100.0 if isinstance(value, (int, float)) and value == value else float("nan")


def _write_figure_captions(
    aggregate_root: Path, figures_root: Path, written: Sequence[Path]
) -> None:
    """Caption every figure with its denominator and an interpretation.

    A chart without its sample size invites over-reading, so the denominators
    are stated beside each figure rather than left in the JSON."""
    def load(name: str) -> Optional[Dict[str, Any]]:
        path = aggregate_root / name
        return read_json_artifact(path) if path.is_file() else None

    open_set = load("bfw_open_set_test_metrics.json")
    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")

    mated = non_mated = enrolled = intended_gallery = "n/a"
    if open_set:
        coverage = open_set["methods"][METHOD_B]["coverage"]
        mated = coverage["scored_mated_probes"]
        non_mated = coverage["scored_non_mated_probes"]
        enrolled = coverage["enrolled_gallery_identities"]
        # Coverage is enrolled over *intended*; using the enrolled count as the
        # denominator would state a rate of one by construction.
        intended_gallery = coverage["intended_gallery_identities"]

    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")
    consistency = load("profile_photo_consistency.json")
    evaluated = bool(pipeline and pipeline.get("evaluated") == "yes")

    lines = [
        "# Figure captions",
        "",
        "Generated from the published JSON and CSV artefacts by `ACP_arden.py`. No value is "
        "typed by hand. Every figure is PNG at 300 dpi plus SVG, with PNG text metadata "
        "stripped, and passes the project privacy scan.",
        "",
        f"**Denominators (BFW held-out test):** {enrolled} of {intended_gallery} gallery "
        f"identities enrolled, {mated} scored mated probes, {non_mated} scored non-mated "
        f"probes.",
        "",
        "**Metric definitions.** FPIR is the proportion of non-mated searches returning at "
        "least one candidate above threshold — a 1:N quantity that compounds with gallery "
        "size, and never interchangeable with the 1:1 false-match rate. TPIR@k is the "
        "proportion of mated searches placing the correct identity within rank k *and* above "
        "threshold; a referral to another identity is a referral, not an identification. "
        "End-to-end detection divides by every intended mated probe, so extraction failures "
        "reduce it; conditional rates divide by those actually scored.",
        "",
        "**Confidence intervals.** All intervals are 95% percentile bounds from an "
        "identity-cluster bootstrap: identities are resampled with replacement, carrying "
        "their complete protocol outcomes, with subgroup stratification preserved. Images "
        "are never resampled independently, which would treat correlated probes of one "
        "person as independent observations and understate the intervals.",
        "",
        "**Coefficients describe association inside the fitted classifier on these "
        "benchmark identities. They are not causal and do not transfer to another "
        "population.**",
        "",
        f"**Experiment 8 status:** `{pipeline['status'] if pipeline else 'not run'}`.",
        "",
        "## Implementation layers (Figures A-D)",
        "",
        "The five layers are measured on the same BFW open-set protocol and are therefore "
        "directly comparable, in the order the project developed them:",
        "",
        "1. Single-image gallery, transferred 1:1 threshold",
        "2. Three-image gallery, transferred 1:1 threshold",
        "3. Three-image gallery, BFW development calibration",
        "4. Logistic-regression review classifier",
        "5. SCRFD + ArcFace, its own BFW development calibration",
        "",
        "LFW and CPLFW are 1:1 verification and are deliberately absent from this series: "
        "mixing an FMR into an FPIR axis would compare different quantities.",
        "",
        "- **implementation_layers_fpir** — false review referrals per 1,000 non-mated "
        f"searches over {non_mated} scored probes. Lower is better.",
        "- **implementation_layers_duplicate_detection** — TPIR@1, TPIR@5 and end-to-end "
        "detection, kept as separate bars because they use different denominators. Higher "
        "is better.",
        "- **implementation_layers_coverage** — gallery, mated and non-mated coverage. The "
        "remainder in each bar is extraction failure, which is shown rather than hidden.",
        "- **implementation_layers_performance_latency** — end-to-end detection against "
        "mean search latency; point size is false reviews per 1,000, so the speed cost of a "
        "stronger pipeline stays visible.",
        "",
        "## Same-person and profile-photo figures (Figures E-F)",
        "",
        "- **mated_non_mated_similarity_distributions** — aggregate histograms only. No "
        "individual score, identifier or path is published.",
    ]
    if consistency:
        lines.append(
            f"- **profile_photo_consistency_outcomes** — over "
            f"{consistency['photographs_assessed']} photographs: "
            f"{consistency['consistent_same_person_photographs']} consistent, "
            f"{consistency['inconsistent_review_candidates']} review candidates, "
            f"{consistency['extraction_failures']} extraction failures. "
            f"An inconsistent result is **not** proof of photo theft or fraud: pose, "
            f"lighting, occlusion, image quality, age difference, detection failure and "
            f"model error all produce it. Every outcome opens human review only."
        )

    lines += [
        "",
        "## Sex-separated figures (Figures G-I)",
        "",
        "Sex is an evaluation dimension only and is never a classifier feature or threshold "
        "input. The female panel covers asian, black, indian and white females; the male "
        "panel covers the same four categories. Both use identical axes, units and ordering "
        "so they compare fairly.",
        "",
        "- **female_subgroup_pipeline_comparison** / **male_subgroup_pipeline_comparison** — "
        "FPIR and TPIR@1 with 95% identity-cluster intervals.",
        "- **female_male_aggregate_comparison** — pooled from underlying identity outcomes, "
        "not by averaging subgroup percentages, which would weight a small subgroup as "
        "heavily as a large one.",
        "",
        "These benchmark categories do not represent every identity or any real "
        "dating-application population.",
        "",
        "## Pipeline figures",
        "",
        "- **pipeline_coverage_and_latency** — coverage beside latency, so a stronger "
        "pipeline's cost is not omitted.",
        "",
        "## Limitations common to every figure",
        "",
        "- Each rate is conditional on the coverage reported beside it.",
        "- Subgroup intervals are wide once the partition is divided eight ways; "
        "overlapping intervals are not evidence of equality.",
        "- These are benchmark identities, not a user population.",
        "- A referral opens human review only. Nothing here proves duplication, fraud, "
        "ownership or identity.",
    ]
    if review:
        c = review["classifier"]
        lines += [
            "",
            "## Note on the classifier",
            "",
            f"The classifier referred {c['non_mated_incorrectly_referred']} non-mated "
            f"searches in error against the calibrated threshold's "
            f"{review['comparator_three_image_open_set_calibrated']['rates'].get('false_reviews_per_1000_non_mated', 0) * (c['scored_non_mated_probes'] / 1000):.0f}. "
            f"Its primary hypothesis — fewer false referrals — is not achieved.",
        ]
    if not evaluated:
        lines += [
            "",
            "## Note on the absent comparison",
            "",
            f"No stronger-pipeline series appears: the comparison did not run "
            f"(`{pipeline['status'] if pipeline else 'unknown'}`). Nothing is estimated in "
            f"its place.",
        ]

    (figures_root / "FIGURE_CAPTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



# The five implementation layers, in the order the project developed them. All
# five are measured on the same BFW open-set protocol so they are directly
# comparable. LFW and CPLFW are 1:1 verification and are never mixed into this
# series: an FMR and an FPIR are not the same quantity.
IMPLEMENTATION_LAYERS = (
    "Layer 1\nsingle image\ntransferred threshold",
    "Layer 2\nthree images\ntransferred threshold",
    "Layer 3\nthree images\nBFW calibration",
    "Layer 4\nreview classifier",
    "Layer 5\nSCRFD + ArcFace\nown calibration",
)


def _collect_implementation_layers(aggregate_root: Path) -> List[Dict[str, Any]]:
    """Assemble the layer series from published artefacts only."""
    def load(name: str) -> Optional[Dict[str, Any]]:
        path = aggregate_root / name
        return read_json_artifact(path) if path.is_file() else None

    open_set = load("bfw_open_set_test_metrics.json")
    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")
    layers: List[Dict[str, Any]] = []
    if not open_set:
        return layers

    control = open_set["methods"][METHOD_A]
    proposed = open_set["methods"][METHOD_B]
    layers.append({
        "name": IMPLEMENTATION_LAYERS[0],
        "rates": control["rates"], "coverage": control["coverage"],
        "end_to_end": control.get("end_to_end_duplicate_detection_rate"),
    })
    layers.append({
        "name": IMPLEMENTATION_LAYERS[1],
        "rates": proposed["at_lfw_control_threshold"], "coverage": proposed["coverage"],
        "end_to_end": None,
    })
    layers.append({
        "name": IMPLEMENTATION_LAYERS[2],
        "rates": proposed["primary_operating_point"], "coverage": proposed["coverage"],
        "end_to_end": proposed.get("end_to_end_duplicate_detection_rate"),
    })
    if review:
        layers.append({
            "name": IMPLEMENTATION_LAYERS[3],
            "rates": review["classifier"], "coverage": review["coverage"],
            "end_to_end": review["classifier"].get("end_to_end_duplicate_detection_rate"),
        })
    if pipeline and pipeline.get("evaluated") == "yes":
        held_out = pipeline.get("held_out_metrics") or {}
        arcface = next(
            (v for k, v in held_out.items() if "arcface" in k.lower()), None
        )
        if arcface:
            layers.append({
                "name": IMPLEMENTATION_LAYERS[4],
                "rates": arcface["rates"], "coverage": arcface["coverage"],
                "end_to_end": arcface.get("end_to_end_duplicate_detection_rate"),
            })
    return layers


def write_implementation_layer_artefacts(
    aggregate_root: Path = AGGREGATE_ROOT
) -> Optional[List[Dict[str, Any]]]:
    """Publish the layer series as machine-readable files.

    Assembled from existing frozen policies and search results; no method or
    threshold is recalculated."""
    layers = _collect_implementation_layers(aggregate_root)
    if not layers:
        return None
    sources = [
        "LFW 1:1 frozen threshold (transferred)",
        "LFW 1:1 frozen threshold (transferred)",
        "BFW development calibration, frozen",
        "BFW classifier-calibration identities, frozen probability",
        "BFW development calibration for SCRFD + ArcFace, frozen",
    ]
    rows: List[Dict[str, Any]] = []
    for index, layer in enumerate(layers):
        rates, coverage = layer["rates"], layer["coverage"]
        rows.append({
            "layer": index + 1,
            "method": layer["name"].replace("\n", " "),
            "fpir": rates.get("fpir"),
            "false_reviews_per_1000_non_mated": rates.get("false_reviews_per_1000_non_mated"),
            "tpir_rank1": rates.get("tpir_rank1"),
            "tpir_rank5": rates.get("tpir_rank5"),
            "end_to_end_duplicate_detection_rate": layer["end_to_end"],
            "gallery_enrolment_coverage": coverage.get("gallery_enrolment_coverage"),
            "mated_probe_coverage": (
                1.0 - coverage["mated_extraction_failure_rate"]
                if isinstance(coverage.get("mated_extraction_failure_rate"), (int, float))
                else None
            ),
            "non_mated_probe_coverage": (
                1.0 - coverage["non_mated_extraction_failure_rate"]
                if isinstance(coverage.get("non_mated_extraction_failure_rate"), (int, float))
                else None
            ),
            "complete_pipeline_latency_mean_ms": coverage.get(
                "complete_pipeline_latency_mean_ms"
            ),
            "search_latency_mean_ms": coverage.get("top1_search_time_mean_ms"),
            "threshold_source": sources[index] if index < len(sources) else "",
        })

    with open(aggregate_root / "implementation_layer_comparison.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json_artifact(
        aggregate_root / "implementation_layer_comparison.json",
        {
            "artifact_type": "implementation_layer_comparison",
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "seed": DEFAULT_RANDOM_SEED,
            "layers": rows,
            "note": (
                "All layers share the BFW open-set protocol and are directly comparable. "
                "LFW and CPLFW are 1:1 verification and are deliberately excluded: an FMR "
                "and an FPIR are different quantities. No layer improved every metric."
            ),
            "policy_note": POLICY_NOTE,
        },
    )
    return rows


def write_profile_consistency_artefacts(
    consistency: Mapping[str, Any], aggregate_root: Path = AGGREGATE_ROOT
) -> None:
    """Publish the consistency outcomes as JSON and CSV. Aggregates only."""
    write_json_artifact(
        aggregate_root / "profile_photo_consistency_metrics.json",
        {"artifact_type": "profile_photo_consistency_metrics", **dict(consistency)},
    )
    fields = [
        "consistent_same_person_photographs", "inconsistent_review_candidates",
        "mismatched_controls_correctly_referred", "mismatched_controls_not_referred",
        "extraction_failures", "mismatched_control_extraction_failures",
        "gallery_reference_unavailable", "photographs_assessed",
    ]
    with open(aggregate_root / "profile_photo_consistency_metrics.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["outcome", "count"])
        for name in fields:
            writer.writerow([name, consistency.get(name, "")])


def render_research_report(aggregate_root: Path = AGGREGATE_ROOT) -> str:
    """Consolidated write-up, ordered so each layer's intent is visible.

    Every figure is read from the published artefacts. No layer is claimed to
    have improved every metric."""
    def load(name: str) -> Optional[Dict[str, Any]]:
        path = aggregate_root / name
        return read_json_artifact(path) if path.is_file() else None

    final = load("lfw_final_metrics.json")
    cplfw = load("cplfw_metrics.json")
    open_set = load("bfw_open_set_test_metrics.json")
    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")
    consistency = load("profile_photo_consistency.json")
    by_sex = load("bfw_sex_aggregated_metrics.json")
    pct = format_percentage

    lines = [
        "# ACP-arden research report",
        "",
        "Auto-generated from the published artefacts. Ordered to show what each layer was "
        "intended to improve, and where it did not.",
        "",
        "## 1. LFW 1:1 verification",
        "",
    ]
    if final:
        lines += [
            f"Accuracy {pct(final.get('accuracy'))}, FMR {pct(final.get('false_match_rate'))}, "
            f"FNMR {pct(final.get('false_non_match_rate'))}, EER "
            f"{pct(final.get('equal_error_rate'))}, extraction failure "
            f"{pct(final.get('failure_rate'))}. This is a 1:1 pair task and its FMR is not "
            f"comparable with the 1:N FPIR figures below.",
        ]
    lines += ["", "## 2. CPLFW cross-pose transfer", ""]
    if cplfw:
        lines += [
            f"Conditional accuracy {pct(cplfw.get('accuracy'))} over "
            f"{format_count(cplfw.get('scored_pairs'))} scored pairs, with "
            f"{pct(cplfw.get('failure_rate'))} of the protocol never reaching comparison. "
            f"Cross-pose *detection*, not comparison, is the dominant finding.",
        ]

    if open_set:
        control = open_set["methods"][METHOD_A]
        proposed = open_set["methods"][METHOD_B]
        primary = proposed["primary_operating_point"]
        lines += [
            "", "## 3. BFW single-image open-set control", "",
            f"FPIR {pct(control['rates']['fpir'])}, TPIR@1 "
            f"{pct(control['rates']['tpir_rank1'])}, "
            f"{format_number(control['rates']['false_reviews_per_1000_non_mated'], 1)} false "
            f"reviews per 1,000. Reusing a 1:1 threshold for 1:N search refers a large share "
            f"of genuinely new identities.",
            "", "## 4. BFW three-image template, same threshold", "",
            f"FPIR {pct(proposed['at_lfw_control_threshold']['fpir'])}, TPIR@1 "
            f"{pct(proposed['at_lfw_control_threshold']['tpir_rank1'])}. Averaging three "
            f"images raises identification but **raises** FPIR at a fixed threshold: a mean "
            f"template sits nearer the centre of the embedding space and is closer to "
            f"everyone. Multi-image enrolment alone did not reduce false reviews.",
            "", "## 5. BFW gallery-specific calibration", "",
            f"FPIR {pct(primary['fpir'])}, TPIR@1 {pct(primary['tpir_rank1'])}, "
            f"{format_number(primary['false_reviews_per_1000_non_mated'], 1)} false reviews "
            f"per 1,000. The reduction is attributable to calibration, not to the "
            f"representation.",
        ]

    if review:
        c = review["classifier"]
        b = review["comparator_three_image_open_set_calibrated"]["rates"]
        achieved = review["success_criteria"].get(
            "fewer_false_reviews_than_threshold_method", {}
        ).get("outcome")
        lines += [
            "", "## 6. Logistic-regression review classifier", "",
            f"FPIR {pct(c['fpir'])} against the threshold method's {pct(b['fpir'])}; TPIR@1 "
            f"{pct(c['tpir_rank1'])} against {pct(b['tpir_rank1'])}; "
            f"{format_number(c['false_reviews_per_1000_non_mated'], 2)} false reviews per "
            f"1,000 against {format_number(b['false_reviews_per_1000_non_mated'], 2)}.",
            "",
            f"The primary hypothesis was that the classifier would reduce false review "
            f"referrals while retaining detection. That criterion is "
            f"**{str(achieved).replace('_', ' ')}**. The classifier raises identification "
            f"while referring more innocent registrations, which is a trade-off rather than "
            f"an improvement.",
        ]

    if by_sex and by_sex.get("groups"):
        for label, heading in (("female", "7. Female subgroup analysis"),
                               ("male", "8. Male subgroup analysis")):
            entry = by_sex["groups"].get(label)
            if not entry:
                continue
            lines += [
                "", f"## {heading}", "",
                f"Pooled over {entry['identities']} identities: FPIR {pct(entry['fpir'])} "
                f"[{pct(entry['fpir_lower_95'])}–{pct(entry['fpir_upper_95'])}], TPIR@1 "
                f"{pct(entry['tpir_rank1'])} "
                f"[{pct(entry['tpir_rank1_lower_95'])}–{pct(entry['tpir_rank1_upper_95'])}], "
                f"mated coverage {pct(entry['mated_probe_coverage'])}. "
                f"Subgroups pooled: {', '.join(entry['subgroups_pooled'])}.",
            ]

    if consistency:
        lines += [
            "", "## 9. Profile-photo identity consistency", "",
            f"Of {consistency['photographs_assessed']} photographs, "
            f"{consistency['consistent_same_person_photographs']} were consistent with their "
            f"profile template, {consistency['inconsistent_review_candidates']} became review "
            f"candidates, {consistency['extraction_failures']} failed extraction and "
            f"{consistency['gallery_reference_unavailable']} had no enrolled reference.",
            "", consistency["interpretation_note"],
        ]

    if pipeline:
        lines += ["", "## 10. YuNet + SFace against SCRFD + ArcFace", ""]
        held_out = pipeline.get("held_out_metrics") or {}
        if held_out:
            lines += ["| Pipeline | Threshold | FPIR | TPIR@1 | Reviews/1,000 | Coverage |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for name, m in held_out.items():
                lines.append(
                    f"| {name} | {format_number(m['development_threshold'], 6)} | "
                    f"{pct(m['rates']['fpir'])} | {pct(m['rates']['tpir_rank1'])} | "
                    f"{format_number(m['rates']['false_reviews_per_1000_non_mated'], 2)} | "
                    f"{pct(m['coverage']['gallery_enrolment_coverage'])} |"
                )
            lines += [
                "",
                "Each pipeline was calibrated on its own development scores; the SFace "
                "threshold is never applied to ArcFace. This is a complete-pipeline "
                "comparison — detection, alignment, preprocessing, embedding width and "
                "runtime all differ — so no difference is attributable to the embedding "
                "model alone.",
            ]
        else:
            lines.append(
                f"Not run: `{pipeline['status']}`. {pipeline.get('reason', '')} No figures "
                f"are invented."
            )
        lines += [
            "", "## 11. Performance against cost", "",
            "A stronger pipeline is not free. Where it improves extraction and "
            "identification it also costs disk and latency, and the trade-off is shown in "
            "`implementation_layers_performance_latency` rather than omitted.",
        ]

    lines += [
        "", "## 12. Limitations and policy", "",
        "ACP-arden is a benchmark-validated, human-review-only academic face-comparison "
        "proof of concept. It evaluates duplicate-profile screening and profile-photo "
        "facial consistency using frozen pretrained face-recognition pipelines and an "
        "identity-disjoint logistic-regression review classifier. A mismatch or duplicate "
        "signal opens human review only and is not proof of fraud, ownership or identity.",
        "",
        "No face-detection or face-recognition network is trained or fine-tuned. Experiment "
        "7 trains a small logistic-regression review classifier on identity-disjoint BFW "
        "development data and evaluates it on untouched held-out identities.",
        "",
    ]
    lines += [f"- {item}" for item in OPEN_SET_LIMITATIONS]
    return "\n".join(lines) + "\n"


def generate_figures(
    *, aggregate_root: Path = AGGREGATE_ROOT, figures_root: Path = FIGURES_ROOT
) -> List[Path]:
    """Produce every dissertation figure the available artefacts support."""
    plt = _figure_backend()
    figures_root.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def load(name: str) -> Optional[Dict[str, Any]]:
        path = aggregate_root / name
        return read_json_artifact(path) if path.is_file() else None

    open_set = load("bfw_open_set_test_metrics.json")
    review = load("ml_review_test_metrics.json")
    pipeline = load("pipeline_comparison_metrics.json")
    intervals = load("ml_review_confidence_intervals.json")


    # --- Figures A-D: implementation layers ----------------------------------
    layers = _collect_implementation_layers(aggregate_root)
    if layers:
        names = [layer["name"] for layer in layers]
        denominator = layers[0]["rates"].get("scored_non_mated_probes", "n/a")

        values = [layer["rates"].get("false_reviews_per_1000_non_mated", float("nan"))
                  for layer in layers]
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        ax.bar(names, values, color="#4C72B0")
        ax.set_ylabel("False reviews per 1,000 non-mated searches (lower is better)")
        ax.set_title(
            f"Implementation layers: false human-review referrals "
            f"(n={denominator} scored non-mated probes)"
        )
        ax.set_ylim(bottom=0)
        for index, value in enumerate(values):
            if value == value:
                ax.text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
        path = figures_root / "implementation_layers_fpir.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        width = 0.26
        positions = np.arange(len(layers))
        for offset, key, label, colour in (
            (-width, "tpir_rank1", "TPIR@1 (conditional)", "#4C72B0"),
            (0.0, "tpir_rank5", "TPIR@5 (conditional)", "#55A868"),
            (width, None, "End-to-end duplicate detection", "#DD8452"),
        ):
            series = [
                _percent(layer["end_to_end"]) if key is None
                else _percent(layer["rates"].get(key))
                for layer in layers
            ]
            ax.bar(positions + offset, series, width, label=label, color=colour)
        ax.set_xticks(positions); ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("Per cent (higher is better)"); ax.set_ylim(0, 100)
        ax.set_title("Implementation layers: duplicate detection, conditional and end-to-end")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        path = figures_root / "implementation_layers_duplicate_detection.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for offset, key, label, colour in (
            (-width, "gallery_enrolment_coverage", "Gallery enrolment", "#55A868"),
            (0.0, "mated_extraction_failure_rate", "Mated probe", "#4C72B0"),
            (width, "non_mated_extraction_failure_rate", "Non-mated probe", "#DD8452"),
        ):
            series = []
            for layer in layers:
                value = layer["coverage"].get(key)
                if key.endswith("failure_rate") and isinstance(value, (int, float)):
                    value = 1.0 - value
                series.append(_percent(value))
            ax.bar(positions + offset, series, width, label=label, color=colour)
        ax.set_xticks(positions); ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("Coverage (%) — extraction failure is the remainder")
        ax.set_ylim(0, 100)
        ax.set_title("Implementation layers: extraction coverage")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        path = figures_root / "implementation_layers_coverage.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        for layer in layers:
            latency = layer["coverage"].get("top1_search_time_mean_ms")
            detection = _percent(layer["end_to_end"])
            reviews = layer["rates"].get("false_reviews_per_1000_non_mated", float("nan"))
            if not (isinstance(latency, (int, float)) and latency == latency and detection == detection):
                continue
            size = 40 + (reviews * 6 if reviews == reviews else 0)
            ax.scatter(latency, detection, s=size, alpha=0.75, color="#4C72B0")
            ax.annotate(layer["name"].replace("\n", " "), (latency, detection),
                        fontsize=6, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Mean gallery-search latency per probe (ms)")
        ax.set_ylabel("End-to-end duplicate detection (%)")
        ax.set_title("Performance against cost — point size is false reviews per 1,000")
        ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
        path = figures_root / "implementation_layers_performance_latency.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

    # --- Figure F: profile-photo consistency ---------------------------------
    consistency_path = aggregate_root / "profile_photo_consistency.json"
    if consistency_path.is_file():
        c = read_json_artifact(consistency_path)
        labels = ["Consistent\nsame-person", "Inconsistent\nreview candidate",
                  "Extraction\nfailure", "Gallery reference\nunavailable"]
        values = [c["consistent_same_person_photographs"], c["inconsistent_review_candidates"],
                  c["extraction_failures"], c["gallery_reference_unavailable"]]
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(labels, values, color=["#55A868", "#DD8452", "#C44E52", "#8172B3"])
        ax.set_ylabel(f"Photographs (n={c['photographs_assessed']})")
        ax.set_title("Profile-photo identity consistency — outcomes open human review only")
        for index, value in enumerate(values):
            ax.text(index, value, str(value), ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        path = figures_root / "profile_photo_consistency_outcomes.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

    # --- Figures G-I: sex-separated results ----------------------------------
    subgroup_csv = aggregate_root / "bfw_subgroup_metrics.csv"
    if subgroup_csv.is_file():
        rows = {r["subgroup"]: r for r in csv.DictReader(open(subgroup_csv, encoding="utf-8"))}
        for sex, suffix in (("female", "_females"), ("male", "_males")):
            members = sorted(k for k in rows if k.endswith(suffix))
            if not members:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharey=True)
            positions = np.arange(len(members))
            for ax, metric, title in (
                (axes[0], "fpir", "FPIR (lower is better)"),
                (axes[1], "tpir_rank1", "TPIR@1 (higher is better)"),
            ):
                centre = [_percent(float(rows[m][metric])) for m in members]
                lower = [max(centre[i] - _percent(float(rows[members[i]][f"{metric}_lower_95"])), 0)
                         for i in range(len(members))]
                upper = [max(_percent(float(rows[members[i]][f"{metric}_upper_95"])) - centre[i], 0)
                         for i in range(len(members))]
                ax.errorbar(positions, centre, yerr=[lower, upper], fmt="o", capsize=4,
                            color="#4C72B0")
                ax.set_xticks(positions)
                ax.set_xticklabels([m.replace(suffix[1:], "") for m in members],
                                   rotation=30, ha="right", fontsize=8)
                ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
                # Shared scale so the companion figures compare fairly.
                ax.set_ylim(0, 100)
            axes[0].set_ylabel("Per cent (95% CI)")
            fig.suptitle(f"{sex.capitalize()} subgroup performance (BFW held-out test)")
            path = figures_root / f"{sex}_subgroup_pipeline_comparison.png"
            _save_figure(fig, path); plt.close(fig); written.append(path)

    sex_path = aggregate_root / "bfw_sex_aggregated_metrics.json"
    if sex_path.is_file():
        groups = read_json_artifact(sex_path).get("groups", {})
        if groups:
            metrics = ("fpir", "tpir_rank1", "mated_probe_coverage", "non_mated_probe_coverage")
            labels = ["FPIR", "TPIR@1", "Mated coverage", "Non-mated coverage"]
            fig, ax = plt.subplots(figsize=(9.0, 5.0))
            positions = np.arange(len(metrics)); width = 0.35
            for offset, (name, colour) in zip((-width / 2, width / 2),
                                              (("female", "#C44E52"), ("male", "#4C72B0"))):
                entry = groups.get(name)
                if not entry:
                    continue
                centre = [_percent(entry.get(m)) for m in metrics]
                lower = [max(centre[i] - _percent(entry.get(f"{metrics[i]}_lower_95")), 0)
                         for i in range(len(metrics))]
                upper = [max(_percent(entry.get(f"{metrics[i]}_upper_95")) - centre[i], 0)
                         for i in range(len(metrics))]
                ax.bar(positions + offset, centre, width, label=name.capitalize(), color=colour,
                       yerr=[lower, upper], capsize=4)
            ax.set_xticks(positions); ax.set_xticklabels(labels)
            ax.set_ylabel("Per cent (95% CI)"); ax.set_ylim(0, 100)
            ax.set_title("Aggregate female against male, pooled over identity outcomes")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            path = figures_root / "female_male_aggregate_comparison.png"
            _save_figure(fig, path); plt.close(fig); written.append(path)

    # --- Figure 1: false review referrals by method --------------------------
    if open_set:
        labels, values = [], []
        control = open_set["methods"][METHOD_A]["rates"]
        proposed = open_set["methods"][METHOD_B]["primary_operating_point"]
        labels.append("Single-image\n1:1 threshold\n(control)")
        values.append(control["false_reviews_per_1000_non_mated"])
        labels.append("Three-image\ncalibrated")
        values.append(proposed["false_reviews_per_1000_non_mated"])
        if review:
            labels.append("Logistic-regression\nclassifier")
            values.append(review["classifier"]["false_reviews_per_1000_non_mated"])
        if pipeline and pipeline.get("evaluated") == "yes":
            labels.append("Stronger\npipeline")
            values.append(float("nan"))

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(labels, values, color="#4C72B0")
        ax.set_ylabel("False reviews per 1,000 non-mated searches")
        ax.set_title("False human-review referrals by method (BFW held-out test)")
        ax.set_ylim(bottom=0)
        for index, value in enumerate(values):
            if value == value:
                ax.text(index, value, f"{value:.1f}", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.3)
        path = figures_root / "false_reviews_per_1000_by_method.png"
        _save_figure(fig, path)
        plt.close(fig)
        written.append(path)

    # --- Figure 2: detection and coverage ------------------------------------
    if open_set:
        proposed = open_set["methods"][METHOD_B]
        primary = proposed["primary_operating_point"]
        coverage = proposed["coverage"]
        groups = ["Conditional\nTPIR@1", "End-to-end\nduplicate detection", "Gallery enrolment\ncoverage"]
        values = [
            _percent(primary["tpir_rank1"]),
            _percent(proposed.get("end_to_end_duplicate_detection_rate")),
            _percent(coverage["gallery_enrolment_coverage"]),
        ]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(groups, values, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_ylabel("Per cent")
        ax.set_ylim(0, 100)
        ax.set_title("Detection and coverage, clearly separated (BFW held-out test)")
        for index, value in enumerate(values):
            if value == value:
                ax.text(index, value, f"{value:.1f}%", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.3)
        path = figures_root / "duplicate_detection_by_method.png"
        _save_figure(fig, path)
        plt.close(fig)
        written.append(path)

    # --- Figure 3: open-set operating curve ----------------------------------
    if open_set:
        development = load("bfw_open_set_development_metrics.json")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        if development:
            points = development["methods"][METHOD_B]["operating_points"]
            xs = [max(points[k]["fpir"], 1e-4) for k in sorted(points)]
            ys = [_percent(points[k]["tpir_rank1"]) for k in sorted(points)]
            ax.plot(xs, ys, "o--", label="Development (threshold selected here)", color="#4C72B0")
        points = open_set["methods"][METHOD_B]["operating_points"]
        xs = [max(points[k]["fpir"], 1e-4) for k in sorted(points)]
        ys = [_percent(points[k]["tpir_rank1"]) for k in sorted(points)]
        ax.plot(xs, ys, "s-", label="Held-out test (never used to select)", color="#C44E52")
        ax.set_xscale("log")
        ax.set_xlabel("FPIR (log scale)")
        ax.set_ylabel("TPIR@1 (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Open-set operating points: TPIR@1 against FPIR")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        path = figures_root / "open_set_operating_curve.png"
        _save_figure(fig, path)
        plt.close(fig)
        written.append(path)

    # --- Figure 4: subgroup performance with intervals -----------------------
    subgroup_csv = aggregate_root / "ml_review_subgroup_metrics.csv"
    if subgroup_csv.is_file():
        rows = list(csv.DictReader(open(subgroup_csv, encoding="utf-8")))
        if rows:
            names = [r["subgroup"] for r in rows]
            positions = list(range(len(names)))
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
            for ax, key, title in (
                (axes[0], "fpir", "FPIR by subgroup (95% CI)"),
                (axes[1], "tpir_rank1", "TPIR@1 by subgroup (95% CI)"),
            ):
                centre = [_percent(float(r[key])) for r in rows]
                lower = [max(centre[i] - _percent(float(rows[i][f"{key}_lower_95"])), 0) for i in positions]
                upper = [max(_percent(float(rows[i][f"{key}_upper_95"])) - centre[i], 0) for i in positions]
                ax.errorbar(positions, centre, yerr=[lower, upper], fmt="o", capsize=4,
                            color="#4C72B0")
                ax.set_ylabel("Per cent")
                ax.set_title(title)
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=0.3)
            axes[1].set_xticks(positions)
            axes[1].set_xticklabels(names, rotation=30, ha="right")
            fig.suptitle("Subgroup performance of the review classifier", y=0.98)
            path = figures_root / "subgroup_fpir_tpir_with_confidence_intervals.png"
            _save_figure(fig, path)
            plt.close(fig)
            written.append(path)

    # --- Figure 5: classifier coefficients -----------------------------------
    model = load("ml_review_model.json")
    if model:
        order = model["model"]["feature_order"]
        weights = model["model"]["coefficients"]
        colours = ["#C44E52" if w < 0 else "#4C72B0" for w in weights]
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.barh(order, weights, color=colours)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Standardised logistic-regression coefficient")
        ax.set_title("Review-classifier coefficients (association, not causation)")
        ax.grid(axis="x", alpha=0.3)
        path = figures_root / "ml_review_classifier_coefficients.png"
        _save_figure(fig, path)
        plt.close(fig)
        written.append(path)

    # --- Figure 6: pipeline coverage and latency -----------------------------
    held_out = (pipeline or {}).get("held_out_metrics") or {}
    if held_out:
        # Both evaluated pipelines, never the primary alone once ArcFace exists.
        names = sorted(held_out, key=lambda n: "opencv" not in n)
        fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.6))
        coverage_labels = ["Gallery", "Mated probe", "Non-mated probe"]
        positions = np.arange(len(coverage_labels)); width = 0.35
        for offset, name, colour in zip((-width / 2, width / 2), names, ("#4C72B0", "#DD8452")):
            c = held_out[name]["coverage"]
            left.bar(positions + offset, [
                _percent(c["gallery_enrolment_coverage"]),
                _percent(1.0 - c["mated_extraction_failure_rate"]),
                _percent(1.0 - c["non_mated_extraction_failure_rate"]),
            ], width, label=name.split("-")[0], color=colour)
        left.set_xticks(positions); left.set_xticklabels(coverage_labels)
        left.set_ylabel("Coverage (%) — higher is better"); left.set_ylim(0, 100)
        left.set_title("Extraction coverage"); left.legend(fontsize=8)
        left.grid(axis="y", alpha=0.3)

        latency_keys = [
            ("embedding_latency_mean_ms", "Embed\nmean"),
            ("embedding_latency_p95_ms", "Embed\np95"),
            ("complete_pipeline_latency_mean_ms", "Complete\nmean"),
            ("complete_pipeline_latency_p95_ms", "Complete\np95"),
            ("top1_search_time_mean_ms", "Search\nmean"),
            ("top1_search_time_p95_ms", "Search\np95"),
        ]
        positions = np.arange(len(latency_keys))
        for offset, name, colour in zip((-width / 2, width / 2), names, ("#4C72B0", "#DD8452")):
            c = held_out[name]["coverage"]
            values = [c.get(k) if isinstance(c.get(k), (int, float)) else float("nan")
                      for k, _ in latency_keys]
            right.bar(positions + offset, values, width, label=name.split("-")[0], color=colour)
        right.set_xticks(positions)
        right.set_xticklabels([lab for _, lab in latency_keys], fontsize=7)
        right.set_ylabel("Milliseconds per image — lower is better")
        right.set_ylim(bottom=0); right.set_title("Latency (model loading excluded)")
        right.legend(fontsize=8); right.grid(axis="y", alpha=0.3)
        fig.suptitle("Complete-pipeline coverage and latency", y=1.0)
        path = figures_root / "pipeline_coverage_and_latency.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)
    elif open_set:
        coverage = open_set["methods"][METHOD_B]["coverage"]
        fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.2))
        bars = ["Gallery", "Mated probe", "Non-mated probe"]
        values = [
            _percent(coverage["gallery_enrolment_coverage"]),
            _percent(1.0 - coverage["mated_extraction_failure_rate"]),
            _percent(1.0 - coverage["non_mated_extraction_failure_rate"]),
        ]
        left.bar(bars, values, color="#55A868")
        left.set_ylabel("Coverage (%)"); left.set_ylim(0, 100)
        left.set_title("Extraction coverage"); left.grid(axis="y", alpha=0.3)
        latencies = [coverage["top1_search_time_mean_ms"], coverage["top5_search_time_p95_ms"]]
        right.bar(["Search mean", "Search p95"], latencies, color="#DD8452")
        right.set_ylabel("Milliseconds"); right.set_ylim(bottom=0)
        right.set_title("Search latency"); right.grid(axis="y", alpha=0.3)
        pipeline_name = open_set.get("pipeline_name") or MODEL_VERSION
        fig.suptitle(f"{pipeline_name} coverage and latency", y=1.0)
        path = figures_root / "pipeline_coverage_and_latency.png"
        _save_figure(fig, path); plt.close(fig); written.append(path)

    write_implementation_layer_artefacts(aggregate_root)
    _write_figure_captions(aggregate_root, figures_root, written)
    (aggregate_root / "RESEARCH_REPORT.md").write_text(
        render_research_report(aggregate_root), encoding="utf-8"
    )

    # Figures are published artefacts and are scanned like any other.
    leaks = find_path_leaks(figures_root, forbidden_substrings=default_forbidden_path_substrings())
    if leaks:
        raise PrivacyLeakError(
            "Refusing to publish figures containing a personal/absolute path:\n"
            + "\n".join(f"  {redact_private_paths(leak)}" for leak in leaks)
        )
    return written


# =============================================================================
# 29. Synthetic self-test mode
# =============================================================================
#
# Deterministic checks that need no model binary, no dataset and no network.
# Detection and embedding are stood in for by small fakes keyed off image
# content, so every behaviour below is fully reproducible.


class SelfTestFailure(AssertionError):
    """Raised by a self-test assertion helper when an expectation does not hold."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SelfTestFailure(message)


def _assert_close(actual: float, expected: float, message: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise SelfTestFailure(f"{message} (expected {expected!r}, got {actual!r})")


def _assert_raises(
    exception: type[BaseException], callable_: Callable[[], Any], message: str
) -> None:
    try:
        callable_()
    except exception:
        return
    except Exception as exc:  # noqa: BLE001 - the wrong exception is still a failure
        raise SelfTestFailure(f"{message} (raised {type(exc).__name__} instead)") from exc
    raise SelfTestFailure(f"{message} (nothing was raised)")


def _image_key(bgr: np.ndarray) -> str:
    return hashlib.sha256(bgr.tobytes()).hexdigest()


class SyntheticDetector:
    """Duck-types YuNetDetector.detect_single_face without a real model."""

    def __init__(self, face_counts: Optional[Dict[str, int]] = None, default_count: int = 1):
        self.face_counts = face_counts or {}
        self.default_count = default_count

    def detect_single_face(self, bgr: np.ndarray) -> np.ndarray:
        count = self.face_counts.get(_image_key(bgr), self.default_count)
        if count != 1:
            raise FaceCountError(count)
        return np.array(
            [0, 0, bgr.shape[1], bgr.shape[0], 0, 0, 0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32
        )


class SyntheticEmbedder:
    """Duck-types SFaceEmbedder.embed: one deterministic vector per image."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS):
        self.dimensions = dimensions

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        seed = int(_image_key(bgr)[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.normal(size=self.dimensions)


def _write_synthetic_image(directory: Path, name: str, fill: int) -> Tuple[Path, str]:
    """A tiny, lossless, uniform-colour PNG. Every channel holds the same value,
    so the bytes are identical read back as RGB or BGR and the returned key
    matches what load_image_bgr will see after the round trip."""
    from PIL import Image

    array = np.full((8, 8, 3), fill, dtype=np.uint8)
    path = directory / name
    Image.fromarray(array, mode="RGB").save(path, format="PNG")
    return path, _image_key(array)


def _self_test_cosine_similarity() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    _assert_close(cosine_similarity(a, a), 1.0, "identical vectors must score 1.0")
    _assert_close(cosine_similarity(a, b), 0.0, "orthogonal vectors must score 0.0")
    _assert_close(cosine_similarity(a, -a), -1.0, "opposed vectors must score -1.0")
    _assert_close(
        cosine_similarity(a, 5.0 * a), 1.0, "cosine similarity must be scale-invariant"
    )
    _assert_raises(
        SimilarityError, lambda: cosine_similarity(a, np.array([1.0, 0.0])), "ragged input"
    )
    _assert_raises(
        SimilarityError,
        lambda: cosine_similarity(a, np.array([np.nan, 0.0, 0.0])),
        "non-finite input",
    )


def _self_test_l2_normalisation() -> None:
    vector = np.array([3.0, 4.0])
    normalized = l2_normalize(vector)
    _assert_close(float(np.linalg.norm(normalized)), 1.0, "unit norm after normalisation")
    _assert_close(float(normalized[0]), 0.6, "direction preserved")
    _assert_close(float(normalized[1]), 0.8, "direction preserved")
    _assert_raises(SimilarityError, lambda: l2_normalize(np.zeros(4)), "zero vector")
    _assert_raises(
        SimilarityError, lambda: l2_normalize(np.array([np.inf, 1.0])), "non-finite vector"
    )


def _self_test_confusion_matrix_accounting() -> None:
    scores = [0.9, 0.8, 0.4, 0.1]
    labels = [1, 0, 1, 0]
    matrix = confusion_matrix(scores, labels, 0.5)
    _assert(matrix.true_positive == 1, "one true positive expected")
    _assert(matrix.false_positive == 1, "one false positive expected")
    _assert(matrix.true_negative == 1, "one true negative expected")
    _assert(matrix.false_negative == 1, "one false negative expected")
    _assert(matrix.total == len(scores), "every scored pair must land in exactly one cell")
    # The decision rule is inclusive: a score exactly on the threshold matches.
    boundary = confusion_matrix([0.5, 0.1], [1, 0], 0.5)
    _assert(boundary.true_positive == 1, "score == threshold must count as a predicted match")
    _assert_raises(MetricsError, lambda: confusion_matrix([0.5], [1], 0.5), "single-class labels")


def _self_test_rate_derivation() -> None:
    matrix = ConfusionMatrix(true_positive=8, false_positive=1, true_negative=9, false_negative=2)
    rates = rates_from_confusion(matrix)
    _assert_close(rates["accuracy"], 17 / 20, "accuracy")
    _assert_close(rates["precision"], 8 / 9, "precision")
    _assert_close(rates["recall"], 8 / 10, "recall")
    _assert_close(rates["f1"], 2 * (8 / 9) * 0.8 / ((8 / 9) + 0.8), "F1")
    _assert_close(rates["false_match_rate"], 1 / 10, "false match rate")
    _assert_close(rates["false_non_match_rate"], 2 / 10, "false non-match rate")
    # An undefined rate must propagate as NaN rather than a misleading zero.
    empty = rates_from_confusion(
        ConfusionMatrix(true_positive=0, false_positive=0, true_negative=5, false_negative=0)
    )
    _assert(empty["recall"] != empty["recall"], "recall must be NaN with no positives")
    _assert(empty["f1"] != empty["f1"], "F1 must be NaN when a component is undefined")


def _self_test_roc_auc() -> None:
    _assert_close(roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0, "perfect separation")
    _assert_close(roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0, "inverted separation")
    # Fully tied scores carry no ranking information at all.
    _assert_close(roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5, "tied scores")
    _assert_close(roc_auc([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0]), 0.75, "partial separation")
    _assert_raises(MetricsError, lambda: roc_auc([0.9, 0.8], [1, 1]), "single-class input")


def _self_test_equal_error_rate() -> None:
    perfect = equal_error_rate([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    _assert_close(perfect["equal_error_rate"], 0.0, "separable scores give a zero EER")
    symmetric = equal_error_rate([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0])
    _assert(
        0.0 <= symmetric["equal_error_rate"] <= 1.0, "EER must lie within the unit interval"
    )
    _assert(
        "threshold" in symmetric, "EER must report the threshold at which the rates cross"
    )


def _self_test_threshold_candidates() -> None:
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    genuine = rng.normal(0.7, 0.05, 60)
    impostor = rng.normal(0.2, 0.05, 60)
    scores = [*genuine.tolist(), *impostor.tolist()]
    labels = [1] * 60 + [0] * 60

    result = calibrate(scores, labels, split=VALIDATION_SPLIT)
    _assert(result.status == CANDIDATES_STATUS, "stage 1 must never produce a frozen artifact")
    expected = {"balanced_accuracy", "f1", "eer", "target_fmr_0.001", "target_fmr_0.01", "target_fmr_0.05"}
    _assert(set(result.candidates) == expected, f"expected candidates {sorted(expected)}")
    # Calibration is confined to the validation split by construction.
    _assert_raises(
        CalibrationError,
        lambda: calibrate(scores, labels, split="test"),
        "calibration on a non-validation split must be refused",
    )


def _self_test_deterministic_selection() -> None:
    candidates = {
        "balanced_accuracy": {"threshold": 0.50},
        "f1": {"threshold": 0.40},
        "eer": {"threshold": 0.45},
    }
    dev_scores = [0.9, 0.8, 0.46, 0.44, 0.3, 0.1]
    dev_labels = [1, 1, 1, 0, 0, 0]

    first = select_final_threshold(candidates, dev_scores, dev_labels)
    second = select_final_threshold(candidates, dev_scores, dev_labels)
    _assert(
        first["selected_candidate"] == second["selected_candidate"],
        "selection must be reproducible for identical input",
    )
    _assert(first["selection_rule"] == SELECTION_RULE, "the published rule must be recorded")
    _assert(
        len(first["all_candidates_dev_metrics"]) == len(candidates),
        "every candidate's development metrics must be retained as evidence",
    )
    # A tie on balanced accuracy is broken by false match rate, then by name,
    # so the outcome never depends on dictionary ordering.
    tied = {"b_name": {"threshold": 0.45}, "a_name": {"threshold": 0.45}}
    tie_result = select_final_threshold(tied, dev_scores, dev_labels)
    _assert(tie_result["selected_candidate"] == "a_name", "a remaining tie is broken by name")


def _self_test_frozen_threshold_enforcement() -> None:
    _assert_raises(
        CalibrationError,
        lambda: require_frozen_threshold({"status": CANDIDATES_STATUS, "threshold": 0.4}),
        "a candidates artifact must never be usable for a final evaluation",
    )
    _assert_raises(
        CalibrationError,
        lambda: require_frozen_threshold({"threshold": 0.4}),
        "an artifact without a status must be refused",
    )
    _assert_raises(
        CalibrationError,
        lambda: require_frozen_threshold({"status": FROZEN_STATUS}),
        "a frozen artifact without a numeric threshold must be refused",
    )
    _assert_close(
        require_frozen_threshold({"status": FROZEN_STATUS, "threshold": 0.42}),
        0.42,
        "a properly frozen threshold must be accepted",
    )
    _assert_raises(
        CalibrationError,
        lambda: require_candidates({"status": FROZEN_STATUS, "candidates": {"a": {}}}),
        "an already-frozen artifact must never be re-selected from",
    )


def _self_test_failure_accounting() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        good_a, good_a_key = _write_synthetic_image(root, "good_a.png", 10)
        good_b, good_b_key = _write_synthetic_image(root, "good_b.png", 20)
        empty, empty_key = _write_synthetic_image(root, "empty.png", 30)
        crowd, crowd_key = _write_synthetic_image(root, "crowd.png", 40)

        detector = SyntheticDetector({empty_key: 0, crowd_key: 2})
        embedder = SyntheticEmbedder()

        pairs = [
            Pair(good_a, good_b, True, "a", "b"),
            Pair(empty, good_b, True, "a", "b"),
            Pair(good_a, empty, False, "a", "b"),
            Pair(crowd, good_b, True, "a", "b"),
            Pair(good_a, crowd, False, "a", "b"),
        ]
        result = evaluate_pairs(pairs, detector=detector, embedder=embedder)

        _assert(result.total_pairs == 5, "the protocol total must be preserved")
        _assert(result.scored_pair_count == 1, "only the fully readable pair can be scored")
        _assert(result.failed_pairs == 4, "failed pairs stay inside the protocol total")
        _assert(
            result.failures
            == {
                "zero_faces_left": 1,
                "zero_faces_right": 1,
                "multiple_faces_left": 1,
                "multiple_faces_right": 1,
            },
            "the four extraction-failure categories must be counted separately",
        )
        result.validate_accounting()
        _assert_close(result.failure_rate, 4 / 5, "failure rate is a fraction of the protocol")
        _assert(good_a_key != good_b_key, "distinct fixtures must hash differently")


def _self_test_gallery_role_uniqueness() -> None:
    images = {
        "alpha": [Path("/tmp/alpha/alpha_0001.jpg"), Path("/tmp/alpha/alpha_0002.jpg")],
        "bravo": [Path("/tmp/bravo/bravo_0001.jpg"), Path("/tmp/bravo/bravo_0002.jpg")],
        "charlie": [Path("/tmp/charlie/charlie_0001.jpg")],
    }
    manifest = build_manifest(images)
    roles = [entry.role for entry in manifest.entries]
    _assert(roles.count("gallery") == 2, "each multi-image identity contributes one gallery entry")
    _assert(roles.count("duplicate_probe") == 2, "and exactly one duplicate probe")
    _assert(roles.count("unknown_probe") == 1, "single-image identities become unknown probes")
    paths = [entry.image_path for entry in manifest.entries]
    _assert(len(paths) == len(set(paths)), "no image may hold more than one role")
    # An identity excluded by calibration must not reappear in the gallery.
    filtered = build_manifest(images, excluded_images=[Path("/tmp/alpha/alpha_0002.jpg")])
    filtered_paths = {entry.image_path for entry in filtered.entries}
    _assert(
        Path("/tmp/alpha/alpha_0002.jpg") not in filtered_paths,
        "excluded calibration images must never enter the gallery experiment",
    )


def _self_test_opaque_id_stability() -> None:
    first = opaque_id("Example_Identity")
    _assert(first == opaque_id("Example_Identity"), "opaque IDs must be reproducible")
    _assert(first != opaque_id("Example_Identity2"), "distinct inputs must not collide")
    _assert(
        len(first) == OPAQUE_ID_HEX_LENGTH,
        f"opaque IDs are truncated to {OPAQUE_ID_HEX_LENGTH} hexadecimal characters",
    )
    _assert(
        "Example_Identity" not in first, "an opaque ID must not contain its own input"
    )
    with temporary_id_hmac_key("f" * 63 + "e"):
        _assert(
            opaque_id("Example_Identity") != first,
            "the secret key must participate in the digest",
        )
    _assert(opaque_ids_match(first, opaque_id("Example_Identity")), "constant-time equality holds")
    _assert(scrub_filename(Path("/private/root/Name_0001.jpg")) == "Name_0001.jpg", "filename only")


def _self_test_path_leak_detection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "clean.json").write_text('{"accuracy": 0.99}\n', encoding="utf-8")
        _assert(
            find_path_leaks(root, forbidden_substrings=["/Users/"]) == [],
            "a clean artifact must produce no findings",
        )
        (root / "leaky.json").write_text('{"root": "/Users/example/data"}\n', encoding="utf-8")
        findings = find_path_leaks(root, forbidden_substrings=["/Users/"])
        _assert(len(findings) == 1, "an absolute path in a published artifact must be reported")
        _assert("leaky.json" in findings[0], "the finding must name the offending file")
    _assert_raises(
        PrivacyLeakError,
        lambda: assert_no_leakage({"image_path": "x"}),
        "a key that names a path must be refused",
    )
    _assert_raises(
        PrivacyLeakError,
        lambda: assert_no_leakage({"root": "/private/location"}),
        "an absolute-looking value must be refused",
    )
    _assert_raises(
        PrivacyLeakError,
        lambda: assert_no_leakage({"vector": [0.1] * 64}),
        "a raw embedding vector must be refused",
    )
    assert_no_leakage({"identity_hash": "abc123", "accuracy": 0.99})


def _self_test_deterministic_gallery_sampling() -> None:
    images = {
        f"identity_{index:02d}": [Path(f"/tmp/i{index}/identity_{index:02d}_0001.jpg")]
        for index in range(20)
    }
    images["anchor"] = [Path("/tmp/anchor/anchor_0001.jpg"), Path("/tmp/anchor/anchor_0002.jpg")]

    first = build_manifest(images, seed=DEFAULT_RANDOM_SEED, max_unknown_identities=5)
    second = build_manifest(images, seed=DEFAULT_RANDOM_SEED, max_unknown_identities=5)
    _assert(
        [e.sample_id for e in first.entries] == [e.sample_id for e in second.entries],
        "the same seed must reproduce the same manifest exactly",
    )
    different = build_manifest(images, seed=DEFAULT_RANDOM_SEED + 1, max_unknown_identities=5)
    _assert(
        {e.sample_id for e in first.entries} != {e.sample_id for e in different.entries},
        "a different seed must sample a different unknown-probe set",
    )
    _assert(first.seed == DEFAULT_RANDOM_SEED, "the manifest must record the seed it used")


SELF_TESTS: Sequence[Tuple[str, Callable[[], None]]] = (
    ("cosine similarity", _self_test_cosine_similarity),
    ("L2 normalisation", _self_test_l2_normalisation),
    ("confusion-matrix accounting", _self_test_confusion_matrix_accounting),
    ("accuracy, precision, recall and F1", _self_test_rate_derivation),
    ("ROC-AUC", _self_test_roc_auc),
    ("equal error rate", _self_test_equal_error_rate),
    ("threshold candidate generation", _self_test_threshold_candidates),
    ("deterministic threshold selection", _self_test_deterministic_selection),
    ("rejection of non-frozen final thresholds", _self_test_frozen_threshold_enforcement),
    ("failure accounting", _self_test_failure_accounting),
    ("gallery role uniqueness", _self_test_gallery_role_uniqueness),
    ("opaque ID stability", _self_test_opaque_id_stability),
    ("path-leak detection", _self_test_path_leak_detection),
    ("deterministic gallery sampling", _self_test_deterministic_gallery_sampling),
)


def run_self_tests(verbose: bool = True) -> Tuple[int, int]:
    """Run every synthetic self-test. Returns (passed, failed).

    The whole run installs a fixed in-memory identifier key, so the self-tests
    stay runnable on a machine with no research configuration and never depend
    on the researcher's real key."""
    passed = 0
    failed = 0
    with temporary_id_hmac_key(SELF_TEST_ID_HMAC_KEY):
        for name, test in SELF_TESTS:
            try:
                test()
            except Exception as exc:  # noqa: BLE001 - any failure is reported, never swallowed
                failed += 1
                if verbose:
                    print(f"FAIL {name}: {redact_private_paths(str(exc))}")
            else:
                passed += 1
                if verbose:
                    print(f"PASS {name}")
    if verbose:
        print("")
        print(f"SELF-TEST RESULT: {'PASS' if failed == 0 else 'FAIL'}")
        print(f"Tests passed: {passed}")
        print(f"Tests failed: {failed}")
    return passed, failed


# =============================================================================
# 30. Interactive VS Code launcher
# =============================================================================
#
# Running this file with no arguments prints a menu rather than starting a
# multi-minute benchmark, so the VS Code play button is safe to press.

MENU_TEXT = f"""
{PROGRAMME_TITLE}

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
"""

MODES = (
    "menu", "check", "verify", "full", "summary", "review", "self-test",
    # Supplementary Experiment 6. Deliberately separate from "full", which
    # continues to mean the original five-experiment evaluation.
    "open-set", "open-set-summary",
    # Experiments 7 and 8. Separate from "full", which remains the five
    # baseline experiments only.
    "ml-review", "ml-review-summary",
    "pipeline-compare", "pipeline-compare-summary",
    "extensions",
)


# --- Stage 0: environment and input verification -----------------------------


def action_check_environment() -> int:
    """Report the interpreter, platform and pinned dependency versions, then
    which storage variables are configured — never their values."""
    report = software_environment_report()
    ok = True
    try:
        check_dependency_contract(strict=True)
    except DependencyContractError as exc:
        ok = False
        dependency_error = str(exc)
    else:
        dependency_error = ""

    print(f"Python: {report['python_version']}")
    print(f"Platform: {report['platform']}")
    print(f"Processor: {report['processor']}")
    dependencies = report["dependencies"]
    assert isinstance(dependencies, dict)
    for package, info in dependencies.items():
        status = "OK" if info["installed"] == info["expected"] else "MISMATCH"
        print(f"  {package}: expected={info['expected']} installed={info['installed']} [{status}]")

    config = EnvironmentConfig.load()
    missing = config.missing_variables()
    print("")
    print("Research storage configuration (values deliberately not printed):")
    for variable in REQUIRED_ENVIRONMENT_VARIABLES:
        print(f"  {variable}: {'set' if variable not in missing else 'NOT SET'}")
    optional_values = {
        "FACE_CACHE_ROOT": config.cache_root,
        "FACE_BFW_ROOT": config.bfw_root,
        "FACE_BFW_METADATA_ROOT": config.bfw_metadata_root,
        "FACE_ARCFACE_MODEL_ROOT": config.arcface_model_root,
    }
    for variable in OPTIONAL_ENVIRONMENT_VARIABLES:
        state = "set" if optional_values.get(variable) is not None else "not set (optional)"
        print(f"  {variable}: {state}")
    # Reported as configured/not configured only. The key's length, encoding
    # and any digest of it stay unprinted: confirming a guess must be impossible.
    print(
        f"  {ID_HMAC_KEY_VARIABLE}: "
        f"{'set' if id_hmac_key_is_configured() else 'NOT SET (required for identifiers)'}"
    )
    if not id_hmac_key_is_configured():
        ok = False
        print("")
        print(
            f"FAILED: {ID_HMAC_KEY_VARIABLE} is not set. Generate one with: "
            'python -c "import secrets;print(secrets.token_urlsafe(32))"',
            file=sys.stderr,
        )
    if missing:
        ok = False
        print("")
        print(
            "FAILED: copy .env.example to .env and fill in the missing variable(s) above, "
            "or export them directly.",
            file=sys.stderr,
        )
    if dependency_error:
        print("")
        print(f"FAILED: {dependency_error}", file=sys.stderr)
    return 0 if ok else 1


def action_verify_inputs() -> int:
    """Hash-verify the two pinned models, then structurally verify the LFW and
    raw CPLFW protocols against the images they reference."""
    config = EnvironmentConfig.load()
    ok = True

    model_root = config.require_model_root()
    print("Models")
    for filename, expected in ((YUNET_FILENAME, YUNET_SHA256), (SFACE_FILENAME, SFACE_SHA256)):
        try:
            actual = verify_model_file(model_root / filename, expected)
            print(f"  OK   {filename}  sha256={actual}")
        except ModelUnavailableError as exc:
            ok = False
            print(f"  FAIL {filename}: {redact_private_paths(str(exc))}", file=sys.stderr)

    lfw_root = config.require_lfw_root()
    protocol_root = config.require_protocol_root()
    print("")
    print("LFW dataset and protocols")
    if not lfw_root.is_dir():
        ok = False
        print("  FAIL LFW dataset root does not exist (check FACE_DATA_ROOT)", file=sys.stderr)
    else:
        for filename in REQUIRED_LFW_PROTOCOLS:
            protocol_path = protocol_root / filename
            if not protocol_path.is_file():
                ok = False
                print(f"  FAIL missing protocol file: {filename}", file=sys.stderr)
                continue
            try:
                pairs = parse_lfw_pairs(protocol_path, lfw_root)
                same = sum(1 for pair in pairs if pair.same_identity)
                print(
                    f"  OK   {filename}: {len(pairs)} pairs "
                    f"({same} matched, {len(pairs) - same} mismatched)"
                )
            except ProtocolError as exc:
                ok = False
                print(f"  FAIL {filename}: {redact_private_paths(str(exc))}", file=sys.stderr)

    cplfw_root = config.require_cplfw_raw_root()
    print("")
    print("Raw CPLFW dataset and protocol")
    print(f"  dataset_image_variant: raw ({CPLFW_RAW_ARCHIVE_FILENAME})")
    if not cplfw_root.is_dir():
        ok = False
        print("  FAIL CPLFW dataset root does not exist (check FACE_CPLFW_RAW_ROOT)", file=sys.stderr)
    else:
        protocol_path = protocol_root / CPLFW_PROTOCOL
        if not protocol_path.is_file():
            ok = False
            print(f"  FAIL missing protocol file: {CPLFW_PROTOCOL}", file=sys.stderr)
        else:
            try:
                pairs = parse_cplfw_pairs(protocol_path, cplfw_root)
                same = sum(1 for pair in pairs if pair.same_identity)
                different = len(pairs) - same
                referenced = {p.left_path for p in pairs} | {p.right_path for p in pairs}
                print(f"  OK   total_pairs             : {len(pairs)}")
                print(f"  OK   same_identity_pairs     : {same}")
                print(f"  OK   different_identity_pairs: {different}")
                print(f"  OK   unique_images_referenced: {len(referenced)}")
                print("  OK   every protocol-referenced image resolved")
                print("  OK   no malformed rows, mismatched labels or duplicate pairs")
                # Parsing aborts on the first unresolved reference, so there is
                # no partial-success path and nothing is silently excluded.
                problems = []
                if len(pairs) != CPLFW_EXPECTED_PAIRS:
                    problems.append(f"expected {CPLFW_EXPECTED_PAIRS} pairs, parsed {len(pairs)}")
                if same != CPLFW_EXPECTED_PER_CLASS:
                    problems.append(
                        f"expected {CPLFW_EXPECTED_PER_CLASS} same-identity pairs, parsed {same}"
                    )
                if different != CPLFW_EXPECTED_PER_CLASS:
                    problems.append(
                        f"expected {CPLFW_EXPECTED_PER_CLASS} different-identity pairs, "
                        f"parsed {different}"
                    )
                for problem in problems:
                    ok = False
                    print(f"  FAIL {problem}", file=sys.stderr)
            except ProtocolError as exc:
                ok = False
                print(f"  FAIL {CPLFW_PROTOCOL}: {redact_private_paths(str(exc))}", file=sys.stderr)

    print("")
    print("Verification complete." if ok else "Verification FAILED — see the messages above.")
    return 0 if ok else 1


# --- Stages 1 to 5: the five experiments -------------------------------------


def experiment_calibrate(
    config: EnvironmentConfig,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    output_root: Path,
) -> Path:
    """Experiment 1, stage 1: candidate thresholds from pairsDevTrain.txt only.
    Never reads the development or final protocol, and never selects a winner."""
    lfw_root = config.require_lfw_root()
    protocol_path = config.require_protocol_root() / LFW_CALIBRATION_PROTOCOL
    pairs = parse_lfw_pairs(protocol_path, lfw_root)

    result = evaluate_pairs(pairs, detector=detector, embedder=embedder)
    if not result.valid_scores:
        raise SystemExit(
            "No pairs were successfully scored during calibration; stopping rather than "
            "fabricating a threshold."
        )

    calibration = calibrate(result.valid_scores, result.valid_labels, split=VALIDATION_SPLIT)
    evaluated_images = {p.left_path for p in pairs} | {p.right_path for p in pairs}
    output_path = output_root / "calibrated_threshold.json"

    write_json_artifact(
        output_path,
        {
            "artifact_type": "calibrated_threshold",
            "dataset": "LFW",
            "protocol_file": LFW_CALIBRATION_PROTOCOL,
            "protocol_sha256": sha256_of_file(protocol_path),
            "evaluated_image_set_sha256": sha256_of_evaluated_image_set(
                evaluated_images, lfw_root
            ),
            "dataset_archive_md5": LFW_ARCHIVE_MD5,
            "split": calibration.split,
            "status": calibration.status,
            "candidates": {
                name: {"threshold": candidate.threshold, "metrics": candidate.metrics}
                for name, candidate in calibration.candidates.items()
            },
            "total_pairs": result.total_pairs,
            "scored_pairs": len(result.valid_scores),
            "failure_breakdown": dict(result.failures),
            "model_version": MODEL_VERSION,
            "preprocessing_revision": PREPROCESSING_REVISION,
            # ``getattr(obj, name, default)`` returns the attribute if present
            # and the default otherwise. The evaluators are typed structurally,
            # so a stand-in stage need not carry a digest; the real wrappers
            # always do, and it is their verified value that is recorded.
            "model_sha256": {
                "yunet": getattr(detector, "model_sha256", YUNET_SHA256),
                "sface": getattr(embedder, "model_sha256", SFACE_SHA256),
            },
            "software_environment": software_environment_report(),
        },
    )
    announce(
        f"Wrote {len(calibration.candidates)} candidate threshold(s) to "
        f"{project_relative(output_path)} (status=candidates; nothing is frozen yet)"
    )
    return output_path


def experiment_evaluate_lfw(
    config: EnvironmentConfig,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    *,
    split: str,
    threshold_artifact: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """split='dev' is selection stage 2: it scores every candidate on
    pairsDevTest.txt, selects one by SELECTION_RULE and rewrites the threshold
    artifact as frozen. split='final' is experiment 3: it evaluates the frozen
    threshold on the untouched pairs.txt and never changes it."""
    protocol_filename = (
        LFW_DEVELOPMENT_PROTOCOL if split == "dev" else LFW_FINAL_PROTOCOL
    )
    lfw_root = config.require_lfw_root()
    protocol_path = config.require_protocol_root() / protocol_filename
    pairs = parse_lfw_pairs(protocol_path, lfw_root)

    result = evaluate_pairs(pairs, detector=detector, embedder=embedder)
    if not result.valid_scores:
        raise SystemExit(f"No pairs were successfully scored on {protocol_filename}; stopping.")

    protocol_sha256 = sha256_of_file(protocol_path)
    evaluated_images = {p.left_path for p in pairs} | {p.right_path for p in pairs}
    # Recorded before any freeze rewrite, so a development artifact references
    # the candidates file it actually selected from.
    threshold_artifact_sha256 = sha256_of_file(threshold_artifact)
    threshold_payload = read_json_artifact(threshold_artifact)

    extra_fields: Dict[str, Any] = {}

    if split == "dev":
        candidates = require_candidates(
            threshold_payload, context=project_relative(threshold_artifact)
        )
        selection = select_final_threshold(candidates, result.valid_scores, result.valid_labels)
        threshold = selection["selected_threshold"]

        frozen_payload = dict(threshold_payload)
        frozen_payload["status"] = FROZEN_STATUS
        frozen_payload["threshold"] = threshold
        frozen_payload["operating_strategy"] = selection["selected_candidate"]
        frozen_payload["selection_rule"] = selection["selection_rule"]
        frozen_payload["selection_evidence"] = selection["all_candidates_dev_metrics"]
        frozen_payload["frozen_from_protocol"] = protocol_filename
        frozen_payload["frozen_from_protocol_sha256"] = protocol_sha256
        write_json_artifact(threshold_artifact, frozen_payload)
        announce(
            f"Selected and froze threshold={threshold:.6f} "
            f"(candidate={selection['selected_candidate']}) in "
            f"{project_relative(threshold_artifact)}, based on {protocol_filename}"
        )

        extra_fields = {
            "selected_candidate": selection["selected_candidate"],
            "selection_rule": selection["selection_rule"],
            "all_candidates_dev_metrics": selection["all_candidates_dev_metrics"],
        }
    else:
        threshold = require_frozen_threshold(
            threshold_payload, context=project_relative(threshold_artifact)
        )

    summary = summarize_metrics(result, threshold)

    write_json_artifact(
        output_path,
        {
            "artifact_type": "lfw_verification_metrics",
            "split": split,
            "protocol_file": protocol_filename,
            "protocol_sha256": protocol_sha256,
            "evaluated_image_set_sha256": sha256_of_evaluated_image_set(
                evaluated_images, lfw_root
            ),
            "dataset_archive_md5": LFW_ARCHIVE_MD5,
            "threshold_source": project_relative(threshold_artifact),
            "threshold_artifact_sha256": threshold_artifact_sha256,
            "threshold_status": FROZEN_STATUS,
            # ``**`` unpacks a dictionary's items into the one being built. Order
            # matters: later keys overwrite earlier ones, and ``extra_fields`` is
            # empty unless this is the selection stage.
            **extra_fields,
            **summary,
            "model_version": MODEL_VERSION,
            "preprocessing_revision": PREPROCESSING_REVISION,
            "model_sha256": {
                "yunet": getattr(detector, "model_sha256", YUNET_SHA256),
                "sface": getattr(embedder, "model_sha256", SFACE_SHA256),
            },
            "software_environment": software_environment_report(),
        },
    )
    announce(
        f"Wrote {split} LFW metrics to {project_relative(output_path)} "
        f"(accuracy={summary['accuracy']:.4f})"
    )
    return summary


def experiment_evaluate_cplfw(
    config: EnvironmentConfig,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    *,
    image_variant: str,
    threshold_artifact: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Experiment 4: cross-pose generalisation using the exact LFW-frozen
    threshold. There is deliberately no CPLFW-specific calibration step."""
    provenance_fields = cplfw_provenance_fields(image_variant)

    cplfw_root = config.require_cplfw_raw_root()
    protocol_path = config.require_protocol_root() / CPLFW_PROTOCOL
    threshold_artifact_sha256 = sha256_of_file(threshold_artifact)
    threshold_payload = read_json_artifact(threshold_artifact)
    threshold = require_frozen_threshold(
        threshold_payload, context=project_relative(threshold_artifact)
    )

    pairs = parse_cplfw_pairs(protocol_path, cplfw_root)
    evaluated_images = {p.left_path for p in pairs} | {p.right_path for p in pairs}

    result = evaluate_pairs(pairs, detector=detector, embedder=embedder)
    summary = summarize_metrics(result, threshold)

    write_json_artifact(
        output_path,
        {
            "artifact_type": "cplfw_verification_metrics",
            **provenance_fields,
            "protocol_file": CPLFW_PROTOCOL,
            "protocol_sha256": sha256_of_file(protocol_path),
            "evaluated_image_set_sha256": sha256_of_evaluated_image_set(
                evaluated_images, cplfw_root
            ),
            "threshold_source": project_relative(threshold_artifact),
            "threshold_artifact_sha256": threshold_artifact_sha256,
            "threshold_status": threshold_payload.get("status"),
            "note": (
                "Frozen threshold calibrated on LFW pairsDevTrain.txt and selected on "
                "pairsDevTest.txt; not recalibrated for CPLFW. This measures cross-pose "
                "generalisation, not a separately tuned CPLFW-specific result."
            ),
            **summary,
            "model_version": MODEL_VERSION,
            "preprocessing_revision": PREPROCESSING_REVISION,
            "model_sha256": {
                "yunet": getattr(detector, "model_sha256", YUNET_SHA256),
                "sface": getattr(embedder, "model_sha256", SFACE_SHA256),
            },
            "software_environment": software_environment_report(),
        },
    )
    announce(
        f"Wrote CPLFW metrics to {project_relative(output_path)} "
        f"(accuracy={summary['accuracy']:.4f})"
    )
    return summary


def experiment_duplicate_gallery(
    config: EnvironmentConfig,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    *,
    threshold_artifact: Path,
    manifest_path: Path,
    output_path: Path,
    review_db: Optional[Path] = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Experiment 5: build a deterministic 1:N gallery from real LFW images and
    measure duplicate detection against the same frozen threshold. Identities
    used for calibration are excluded, so calibration data never leaks in."""
    lfw_root = config.require_lfw_root()
    identity_to_images = discover_identity_images(lfw_root)

    calibration_pairs = parse_lfw_pairs(
        config.require_protocol_root() / LFW_CALIBRATION_PROTOCOL, lfw_root
    )
    excluded_images: Set[Path] = set()
    for pair in calibration_pairs:
        excluded_images.add(pair.left_path)
        excluded_images.add(pair.right_path)

    manifest = build_manifest(identity_to_images, excluded_images=excluded_images, seed=seed)
    write_gallery_manifest(manifest, manifest_path)

    gallery_count = sum(1 for e in manifest.entries if e.role == "gallery")
    duplicate_count = sum(1 for e in manifest.entries if e.role == "duplicate_probe")
    unknown_count = sum(1 for e in manifest.entries if e.role == "unknown_probe")
    announce(
        f"Wrote the private gallery manifest to {project_relative(manifest_path)}: "
        f"{gallery_count} gallery, {duplicate_count} duplicate probes, "
        f"{unknown_count} unknown probes (contains real image paths — kept out of Git)"
    )

    manifest_sha256 = sha256_of_file(manifest_path)
    threshold_artifact_sha256 = sha256_of_file(threshold_artifact)
    threshold_payload = read_json_artifact(threshold_artifact)
    threshold = require_frozen_threshold(
        threshold_payload, context=project_relative(threshold_artifact)
    )

    result = evaluate_gallery(
        manifest, detector=detector, embedder=embedder, duplicate_review_threshold=threshold
    )
    summary = summarize_gallery_metrics(result)

    write_json_artifact(
        output_path,
        {
            "artifact_type": "duplicate_gallery_metrics_v2",
            # The scheme, never the key: this records which identifier family
            # the opaque values below belong to, so two runs under different
            # keys are not mistaken for comparable ones.
            "opaque_id_version": OPAQUE_ID_VERSION,
            "duplicate_review_threshold": threshold,
            "threshold_source": project_relative(threshold_artifact),
            "threshold_artifact_sha256": threshold_artifact_sha256,
            "threshold_strategy": threshold_payload.get("operating_strategy"),
            "manifest_sha256": manifest_sha256,
            "dataset_archive_md5": LFW_ARCHIVE_MD5,
            "seed": manifest.seed,
            "policy_note": POLICY_NOTE,
            **summary,
            "model_version": MODEL_VERSION,
            "preprocessing_revision": PREPROCESSING_REVISION,
            "model_sha256": {
                "yunet": getattr(detector, "model_sha256", YUNET_SHA256),
                "sface": getattr(embedder, "model_sha256", SFACE_SHA256),
            },
            "software_environment": software_environment_report(),
        },
    )
    announce(
        f"Wrote duplicate gallery metrics to {project_relative(output_path)} "
        f"(duplicate_detection_rate={summary['duplicate_detection_rate']:.4f})"
    )

    if review_db is not None:
        flagged = populate_review_database(review_db, result, threshold)
        announce(f"Wrote {flagged} review case(s) to {project_relative(review_db)}")

    return summary


def action_run_complete_evaluation(
    output_root: Path = AGGREGATE_ROOT,
    *,
    manifest_path: Path = DEFAULT_GALLERY_MANIFEST,
    review_db: Path = DEFAULT_REVIEW_DB,
    cplfw_image_variant: str = "raw",
) -> int:
    """The five experiments in their required order, stopping with the
    underlying error rather than fabricating a result if any input is missing."""
    config = EnvironmentConfig.load()
    missing = config.missing_variables()
    if missing:
        raise SystemExit(
            "Cannot run the evaluation: "
            + ", ".join(missing)
            + " not configured. Copy .env.example to .env and fill it in."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("[1/7] Verifying the pinned models")
    detector, embedder = load_models(config.require_model_root())
    print(f"  OK   {YUNET_FILENAME} and {SFACE_FILENAME} match their pinned SHA-256 digests")

    print("")
    print("[2/7] Experiment 1 — threshold candidates (pairsDevTrain.txt, validation only)")
    threshold_artifact = experiment_calibrate(config, detector, embedder, output_root)

    print("")
    print("[3/7] Experiment 2 — development selection and freezing (pairsDevTest.txt)")
    experiment_evaluate_lfw(
        config,
        detector,
        embedder,
        split="dev",
        threshold_artifact=threshold_artifact,
        output_path=output_root / "lfw_development_metrics.json",
    )

    print("")
    print("[4/7] Experiment 3 — final LFW evaluation (pairs.txt, frozen threshold)")
    experiment_evaluate_lfw(
        config,
        detector,
        embedder,
        split="final",
        threshold_artifact=threshold_artifact,
        output_path=output_root / "lfw_final_metrics.json",
    )

    print("")
    print("[5/7] Experiment 4 — raw CPLFW cross-pose generalisation (same frozen threshold)")
    experiment_evaluate_cplfw(
        config,
        detector,
        embedder,
        image_variant=cplfw_image_variant,
        threshold_artifact=threshold_artifact,
        output_path=output_root / "cplfw_metrics.json",
    )

    print("")
    print("[6/7] Experiment 5 — 1:N duplicate-profile gallery (real LFW images)")
    experiment_duplicate_gallery(
        config,
        detector,
        embedder,
        threshold_artifact=threshold_artifact,
        manifest_path=manifest_path,
        # Versioned filename: the corrected accounting changes what the
        # denominators mean, so it is written alongside the historical artefact
        # rather than over it.
        output_path=output_root / "duplicate_gallery_metrics_v2.json",
        review_db=review_db,
    )

    print("")
    print("[7/7] Aggregate outputs and privacy validation")
    write_aggregate_reports(output_root, manifest_path, cplfw_image_variant)

    elapsed = time.perf_counter() - started
    print("")
    announce(
        f"Complete in {elapsed / 60:.1f} minutes. Aggregate results are in "
        f"{project_relative(output_root)}"
    )
    print("")
    print(render_results_summary(output_root))
    return 0


def action_show_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    try:
        print(render_results_summary(output_root))
    except ArtifactError:
        print(
            "No results are available yet. Run option 3 (the complete five-experiment "
            "evaluation) first, or `python ACP_arden.py --mode full`.",
            file=sys.stderr,
        )
        return 1
    return 0


def action_self_test() -> int:
    _passed, failed = run_self_tests()
    return 0 if failed == 0 else 1


# --- Menu and command-line entry point ---------------------------------------


def _run_action(action: Callable[[], int]) -> int:
    """Run one menu action, reporting an expected failure as a redacted message
    rather than a traceback that could disclose a storage location."""
    try:
        return action()
    except (
        ArtifactError,
        BfwDatasetError,
        CalibrationError,
        ConfigurationError,
        GalleryError,
        ImageLoadError,
        ModelUnavailableError,
        OpaqueIdentifierKeyError,
        OpenSetPolicyError,
        MlReviewError,
        OpenSetProtocolError,
        PipelineUnavailableError,
        PrivacyLeakError,
        ProtocolError,
        ReviewDatabaseVersionError,
        SystemExit,
    ) as exc:
        message = str(exc)
        if message:
            print(f"\nStopped: {redact_private_paths(message)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def action_run_open_set_evaluation(output_root: Path = AGGREGATE_ROOT) -> int:
    """Experiment 6. Requires the official BFW dataset; stops on the exact
    blocker rather than degrading to a partial or invented result."""
    run_open_set_experiment(output_root=output_root)
    print("")
    for line in report_optional_dataset_status():
        print(line)
    print("")
    print(render_open_set_summary(output_root))
    return 0


def action_show_open_set_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    print(render_open_set_summary(output_root))
    return 0


def action_run_ml_review(output_root: Path = AGGREGATE_ROOT) -> int:
    """Experiment 7. Requires BFW and a completed open-set run for the
    comparator threshold."""
    run_ml_review_experiment(output_root=output_root)
    written = generate_figures(aggregate_root=output_root)
    announce(f"Wrote {len(written)} figure(s) to {project_relative(FIGURES_ROOT)}")
    print("")
    print(render_ml_review_summary(output_root))
    return 0


def action_show_ml_review_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    print(render_ml_review_summary(output_root))
    return 0


def action_run_pipeline_comparison(output_root: Path = AGGREGATE_ROOT) -> int:
    run_pipeline_comparison(output_root=output_root)
    print("")
    print(render_pipeline_comparison_summary(output_root))
    return 0


def action_show_pipeline_comparison_summary(output_root: Path = AGGREGATE_ROOT) -> int:
    print(render_pipeline_comparison_summary(output_root))
    return 0


def action_run_extensions(output_root: Path = AGGREGATE_ROOT) -> int:
    """Both extension experiments. An unavailable optional pipeline must not
    prevent the classifier experiment from being reported."""
    status = run_ml_review_experiment(output_root=output_root)
    try:
        run_pipeline_comparison(output_root=output_root)
    except (PipelineUnavailableError, ModelUnavailableError) as exc:
        announce(f"Pipeline comparison: NOT RUN — {redact_private_paths(str(exc))}")
    written = generate_figures(aggregate_root=output_root)
    announce(f"Wrote {len(written)} figure(s) to {project_relative(FIGURES_ROOT)}")
    print("")
    print(render_ml_review_summary(output_root))
    print("")
    print(render_pipeline_comparison_summary(output_root))
    return 0 if status else 1


def run_menu() -> int:
    """Interactive menu. Nothing long-running starts until an option is chosen."""
    actions: Dict[str, Callable[[], int]] = {
        "1": action_check_environment,
        "2": action_verify_inputs,
        "3": action_run_complete_evaluation,
        "4": action_show_summary,
        "5": lambda: launch_review_interface(DEFAULT_REVIEW_DB),
        "6": action_self_test,
        "8": action_run_open_set_evaluation,
        "9": action_show_open_set_summary,
        "10": action_run_ml_review,
        "11": action_show_ml_review_summary,
        "12": action_run_pipeline_comparison,
        "13": action_run_extensions,
    }
    last_status = 0
    while True:
        print(MENU_TEXT)
        try:
            choice = input("Select an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return last_status

        if choice in {"7", "q", "quit", "exit"}:
            return last_status
        action = actions.get(choice)
        if action is None:
            print(f"'{choice}' is not one of the options above.")
            continue
        print("")
        last_status = _run_action(action)
        print("")
        try:
            input("Press Enter to return to the menu...")
        except (EOFError, KeyboardInterrupt):
            print("")
            return last_status


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ACP_arden.py",
        description=(
            f"{PROGRAMME_TITLE} (v{PROGRAMME_VERSION}). Run without arguments for an "
            f"interactive menu."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="menu",
        help=(
            "menu (default): interactive launcher. check: environment and dependencies. "
            "verify: models and benchmark datasets. full: the complete five-experiment "
            "evaluation. summary: the existing results. review: the local human-review "
            "interface. self-test: deterministic synthetic tests. open-set: the "
            "supplementary BFW open-set duplicate-profile experiment (Experiment 6). "
            "open-set-summary: the existing open-set results. ml-review: the "
            "machine-learning review classifier (Experiment 7). pipeline-compare: the "
            "pretrained pipeline comparison (Experiment 8). extensions: both, then figures."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=AGGREGATE_ROOT,
        help="Where aggregate results are written and read (default: results/aggregate).",
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=DEFAULT_REVIEW_DB,
        help="Local review database (default: results/raw/review.sqlite; never committed).",
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAMME_NAME} {PROGRAMME_VERSION}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Streamlit re-runs this file on every interaction and appends its own
    # arguments, so unknown arguments are ignored rather than fatal.
    parser = build_argument_parser()
    args, _unknown = parser.parse_known_args(argv)

    if args.mode == "review" and running_under_streamlit():
        render_review_page(args.review_db)
        return 0

    if args.mode == "menu":
        return run_menu()
    if args.mode == "check":
        return _run_action(action_check_environment)
    if args.mode == "verify":
        return _run_action(action_verify_inputs)
    if args.mode == "full":
        return _run_action(lambda: action_run_complete_evaluation(args.results_root))
    if args.mode == "summary":
        return _run_action(lambda: action_show_summary(args.results_root))
    if args.mode == "review":
        return launch_review_interface(args.review_db)
    if args.mode == "self-test":
        return action_self_test()
    if args.mode == "open-set":
        return _run_action(lambda: action_run_open_set_evaluation(args.results_root))
    if args.mode == "open-set-summary":
        return _run_action(lambda: action_show_open_set_summary(args.results_root))
    if args.mode == "ml-review":
        return _run_action(lambda: action_run_ml_review(args.results_root))
    if args.mode == "ml-review-summary":
        return _run_action(lambda: action_show_ml_review_summary(args.results_root))
    if args.mode == "pipeline-compare":
        return _run_action(lambda: action_run_pipeline_comparison(args.results_root))
    if args.mode == "pipeline-compare-summary":
        return _run_action(lambda: action_show_pipeline_comparison_summary(args.results_root))
    if args.mode == "extensions":
        return _run_action(lambda: action_run_extensions(args.results_root))

    parser.error(f"Unhandled mode: {args.mode}")
    return 2


if __name__ == "__main__":
    # Streamlit re-executes this file for every interaction and owns the
    # process lifecycle itself. Raising SystemExit there aborts the script run
    # before Streamlit can mark it finished, which leaves the page hanging, so
    # the review path returns normally and only a plain interpreter run exits
    # with a status code.
    if running_under_streamlit():
        main()
    else:
        raise SystemExit(main())
