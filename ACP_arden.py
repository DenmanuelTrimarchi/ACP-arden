#!/usr/bin/env python3
"""ACP-arden — a single-file face-verification research artefact.

A standalone, local proof of concept that measures how well a fixed,
pretrained face-verification pipeline (OpenCV YuNet detection followed by
OpenCV SFace embedding) can decide whether two unconstrained facial images
belong to the same person, and whether the same similarity signal can
surface duplicate profiles in a 1:N gallery under a human-review policy.

This is a research artefact, not a dating application and not a fraud
detector. No model is trained or fine-tuned here, no website is scraped, and
no account is ever banned, rejected, accused or classified as a scam. A
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

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
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
# public APIs, which is use rather than adaptation of their source. The two
# pretrained ONNX files are external artefacts, not code; they are published
# in the OpenCV Zoo repository (github.com/opencv/opencv_zoo) under the MIT
# licence (YuNet) and the Apache-2.0 licence (SFace), and are pinned by the
# SHA-256 digests in section 2. LFW and CPLFW are external datasets used under
# their authors' published terms; neither is redistributed here.


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
}

DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
HARD_MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 12_000_000
HARD_MAX_IMAGE_PIXELS = 40_000_000

DEFAULT_RANDOM_SEED = 20260727

# Archive checksums recorded for this project's own dataset acquisition. They
# describe the copies evaluated here; the CPLFW authors do not publish an
# official archive checksum of their own.
LFW_ARCHIVE_FILENAME = "lfwfunneled.tgz"
LFW_ARCHIVE_MD5 = "1b42dfed7d15c9b2dd63d5e5840c86ad"
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
OPTIONAL_ENVIRONMENT_VARIABLES = ("FACE_CACHE_ROOT",)


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

    @classmethod
    def load(cls, env: Optional[Mapping[str, str]] = None) -> "EnvironmentConfig":
        """Process environment first, then the local ``.env`` as a fallback, so
        an explicit export always wins over the file."""
        source: Dict[str, str] = dict(load_env_file())
        source.update({k: v for k, v in (os.environ if env is None else env).items() if v})

        def optional(name: str) -> Optional[Path]:
            value = source.get(name)
            return Path(value).expanduser() if value else None

        return cls(
            data_root=optional("FACE_DATA_ROOT"),
            protocol_root=optional("FACE_PROTOCOL_ROOT"),
            model_root=optional("FACE_MODEL_ROOT"),
            cplfw_raw_root=optional("FACE_CPLFW_RAW_ROOT"),
            cache_root=optional("FACE_CACHE_ROOT"),
        )

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


OPAQUE_ID_SALT = "face-verification-opaque-id-v1"


def sha256_of_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
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


def opaque_id(value: str, *, salt: str = OPAQUE_ID_SALT) -> str:
    """Deterministic, one-way identifier standing in for a real identity or
    sample name. Deterministic rather than random so a re-run reproduces the
    same identifiers without ever storing the reversible mapping."""
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return digest[:16]


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


@dataclass(frozen=True)
class LoadedImage:
    bgr: np.ndarray  # HxWx3 uint8, OpenCV's BGR channel order
    width: int
    height: int
    source_path: Path


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
        raise ImageLoadError(f"Could not decode image {path}: {exc}") from exc

    bgr = np.ascontiguousarray(array[:, :, ::-1])
    return LoadedImage(bgr=bgr, width=width, height=height, source_path=path)


# =============================================================================
# 6. YuNet face detection
# =============================================================================
#
# Exactly one detectable face is required, matching the research question
# ("does this photo show one identifiable face"). Zero and multiple detections
# are counted as explicit outcomes in section 12, never silently dropped.


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
        count = 0 if faces is None else len(faces)
        if count != 1:
            raise FaceCountError(count)
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


def roc_auc(scores: ScoreInput, labels: LabelInput) -> float:
    """Rank-based ROC-AUC (the Mann-Whitney U identity), ties resolved with
    average ranks. Equivalent to the trapezoidal-rule area, without pulling in
    a machine-learning framework for one statistic."""
    scores_arr, labels_arr = _validate_inputs(scores, labels)
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


@dataclass(frozen=True)
class GalleryEvaluationResult:
    gallery_size: int
    probe_results: List[ProbeResult]
    search_times_seconds: List[float] = field(default_factory=list)


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
    for entry in gallery_entries:
        embedding, _failure = _embed_entry(entry, detector, embedder)
        if embedding is not None:
            gallery_embeddings.append((entry, embedding))
    if not gallery_embeddings:
        raise GalleryError("No gallery entry could be embedded; cannot run the experiment.")

    results: List[ProbeResult] = []
    search_times: List[float] = []
    for probe in probe_entries:
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
            key=lambda item: (-item[1], item[0].sample_id),
        )
        search_times.append(time.perf_counter() - search_start)
        top_entry, top_similarity = similarities[0]
        rank1_correct = (
            top_entry.identity_hash == probe.identity_hash
            if probe.role == "duplicate_probe"
            else None
        )

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
            )
        )

    return GalleryEvaluationResult(
        gallery_size=len(gallery_embeddings),
        probe_results=results,
        search_times_seconds=search_times,
    )


def summarize_gallery_metrics(result: GalleryEvaluationResult) -> Dict[str, Any]:
    duplicate_probes = [r for r in result.probe_results if r.role == "duplicate_probe"]
    unknown_probes = [r for r in result.probe_results if r.role == "unknown_probe"]
    scored_duplicates = [r for r in duplicate_probes if r.failure_code is None]
    scored_unknowns = [r for r in unknown_probes if r.failure_code is None]

    duplicate_detection_rate = (
        sum(1 for r in scored_duplicates if r.exceeds_duplicate_threshold) / len(scored_duplicates)
        if scored_duplicates
        else float("nan")
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
    # Self-comparison is the NaN test: with no scored duplicate probes the miss
    # rate stays undefined instead of being reported as a perfect 1.0.
    true_duplicate_miss_rate = (
        1.0 - duplicate_detection_rate
        if duplicate_detection_rate == duplicate_detection_rate
        else float("nan")
    )

    search_times_ms = [t * 1000.0 for t in result.search_times_seconds]

    return {
        "gallery_size": result.gallery_size,
        "duplicate_probe_count": len(duplicate_probes),
        "unknown_probe_count": len(unknown_probes),
        "duplicate_probe_failures": len(duplicate_probes) - len(scored_duplicates),
        "unknown_probe_failures": len(unknown_probes) - len(scored_unknowns),
        "duplicate_detection_rate": duplicate_detection_rate,
        "false_duplicate_review_rate": false_duplicate_review_rate,
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
        f"- Duplicate detection rate: "
        f"**{format_percentage(gallery_payload.get('duplicate_detection_rate'))}**",
        f"- Rank-1 identification rate: "
        f"{format_percentage(gallery_payload.get('rank1_identification_rate'))}",
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
    gallery_payload = read_json_artifact(output_root / "duplicate_gallery_metrics.json")
    threshold_payload = read_json_artifact(threshold_path)

    output_files = {
        "calibrated_threshold.json": threshold_path,
        "lfw_development_metrics.json": output_root / "lfw_development_metrics.json",
        "lfw_final_metrics.json": output_root / "lfw_final_metrics.json",
        "cplfw_metrics.json": output_root / "cplfw_metrics.json",
        "duplicate_gallery_metrics.json": output_root / "duplicate_gallery_metrics.json",
    }

    write_json_artifact(
        output_root / "run_manifest.json",
        {
            "artifact_type": "run_manifest",
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

    announce(
        f"Wrote run_manifest.json, metrics_summary.csv ({len(summary_rows)} rows), "
        f"confusion_matrices.csv ({len(confusion_rows)} rows), roc_points.csv "
        f"({len(roc_rows)} rows) and FINAL_EVALUATION_REPORT.md to "
        f"{project_relative(output_root)}"
    )


def render_results_summary(output_root: Path = AGGREGATE_ROOT) -> str:
    """Headline figures for the terminal, read from the aggregate artifacts
    rather than hard-coded. Each conditional figure is printed together with
    the limitation that makes it interpretable: the CPLFW accuracy never
    appears without its extraction-failure rate, and the gallery detection rate
    never appears without its false-review rate."""
    final = read_json_artifact(output_root / "lfw_final_metrics.json")
    cplfw = read_json_artifact(output_root / "cplfw_metrics.json")
    gallery = read_json_artifact(output_root / "duplicate_gallery_metrics.json")
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
        f"  Duplicate detection rate: "
        f"{format_percentage(gallery.get('duplicate_detection_rate'))}",
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

        if (
            isinstance(value, (list, tuple))
            and len(value) >= 32
            and all(isinstance(item, (int, float)) for item in value)
        ):
            raise PrivacyLeakError(f"{label}: value looks like a raw embedding vector")

        if isinstance(value, Mapping):
            assert_no_leakage(value, context=label)


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
    decided_at TEXT
);
"""


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


@contextmanager
def review_database(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(REVIEW_SCHEMA)
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
             status, created_at)
        VALUES (?, ?, ?, ?, ?, 'open', ?)
        ON CONFLICT(case_id) DO UPDATE SET
            probe_sample_id=excluded.probe_sample_id,
            candidate_identity_hash=excluded.candidate_identity_hash,
            similarity=excluded.similarity,
            threshold=excluded.threshold
        """,
        (case_id, probe_sample_id, candidate_identity_hash, similarity, threshold, utc_now_iso()),
    )


def list_review_cases(
    connection: sqlite3.Connection, *, status: Optional[str] = None
) -> List[ReviewCase]:
    if status is not None and status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"Unknown status filter: {status}")
    if status:
        rows = connection.execute(
            "SELECT * FROM review_cases WHERE status = ? ORDER BY similarity DESC", (status,)
        )
    else:
        rows = connection.execute("SELECT * FROM review_cases ORDER BY similarity DESC")
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
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # noqa: BLE001 - Streamlit is optional at import time
        return False
    try:
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

    status_filter = st.selectbox("Filter by status", ["all", *REVIEW_STATUSES])

    with review_database(db_path) as connection:
        cases = list_review_cases(
            connection, status=None if status_filter == "all" else status_filter
        )

        if not cases:
            st.info(
                "No cases match this filter. Run the complete evaluation "
                "(`python ACP_arden.py --mode full`) to populate the local review database."
            )
            return

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
# 17. Synthetic self-test mode
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
    _assert(len(first) == 16, "opaque IDs are truncated to 16 hexadecimal characters")
    _assert(
        "Example_Identity" not in first, "an opaque ID must not contain its own input"
    )
    _assert(
        opaque_id("Example_Identity", salt="other") != first,
        "the salt must participate in the digest",
    )
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
    """Run every synthetic self-test. Returns (passed, failed)."""
    passed = 0
    failed = 0
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
# 18. Interactive VS Code launcher
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
"""

MODES = ("menu", "check", "verify", "full", "summary", "review", "self-test")


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
    for variable in OPTIONAL_ENVIRONMENT_VARIABLES:
        state = "set" if config.cache_root is not None else "not set (optional)"
        print(f"  {variable}: {state}")
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
            "artifact_type": "duplicate_gallery_metrics",
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
        output_path=output_root / "duplicate_gallery_metrics.json",
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
        CalibrationError,
        ConfigurationError,
        GalleryError,
        ImageLoadError,
        ModelUnavailableError,
        PrivacyLeakError,
        ProtocolError,
        SystemExit,
    ) as exc:
        message = str(exc)
        if message:
            print(f"\nStopped: {redact_private_paths(message)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def run_menu() -> int:
    """Interactive menu. Nothing long-running starts until an option is chosen."""
    actions: Dict[str, Callable[[], int]] = {
        "1": action_check_environment,
        "2": action_verify_inputs,
        "3": action_run_complete_evaluation,
        "4": action_show_summary,
        "5": lambda: launch_review_interface(DEFAULT_REVIEW_DB),
        "6": action_self_test,
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
            "interface. self-test: deterministic synthetic tests."
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

    parser.error(f"Unhandled mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
