"""
Configuration loader for Resume Matcher.

Loads configuration from YAML files and environment variables.
Environment variables override YAML values.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = Path("config/default.yaml")
_ENV_PREFIX = "RESUME_MATCHER_"


# ---------------------------------------------------------------------------
# Typed configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessingConfig:
    """File processing limits and settings."""

    max_file_size_mb: int = 25
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx")
    batch_size: int = 32
    max_lines_per_document: int = 10_000


@dataclass(frozen=True)
class EmbeddingConfig:
    """BGE embedding model settings."""

    model_name: str = "BAAI/bge-base-en-v1.5"
    dimension: int = 768
    normalize: bool = True
    batch_size: int = 32
    show_progress: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    """FAISS retrieval settings."""

    top_k: int = 20
    similarity_threshold: float = 0.45
    section_match_boost: float = 0.05


@dataclass(frozen=True)
class ScoringWeights:
    """Scoring component weights — must sum to 1.0."""

    skill: float = 0.25
    work_experience: float = 0.25
    education: float = 0.25
    content_score: float = 0.25
    semantic_match: float | None = None

    def __post_init__(self) -> None:
        # Backward compatibility: if semantic_match provided, use it for content_score
        if self.semantic_match is not None and self.content_score == 0.25 and self.semantic_match != 0.25:
            object.__setattr__(self, "content_score", self.semantic_match)
        elif self.semantic_match is None:
            object.__setattr__(self, "semantic_match", self.content_score)

        total = self.skill + self.work_experience + self.education + self.content_score
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.6f} "
                f"(skill={self.skill}, work_experience={self.work_experience}, "
                f"education={self.education}, content_score={self.content_score})"
            )


@dataclass(frozen=True)
class ScoringThresholds:
    """Thresholds for categorizing match strength."""

    strong_match: float = 0.70
    moderate_match: float = 0.50
    weak_match: float = 0.30
    cosine_floor: float = 0.30  # Hard floor: below this → 0.0 score


@dataclass(frozen=True)
class ScoringConfig:
    """Complete scoring configuration."""

    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: ScoringThresholds = field(default_factory=ScoringThresholds)


@dataclass(frozen=True)
class SegmentationConfig:
    """Block building and sentence segmentation settings."""

    max_words_per_chunk: int = 40
    min_block_length_chars: int = 20
    merge_short_blocks: bool = True
    use_spacy: bool = True
    max_sentences_per_block: int | None = None


@dataclass(frozen=True)
class SectionDetectionConfig:
    """Section detection and inference settings."""

    min_confidence: float = 0.5
    flag_low_confidence: bool = True


@dataclass(frozen=True)
class ReportingConfig:
    """Report generation settings."""

    schema_version: str = "1.0.0"
    top_matches_count: int = 5
    bottom_matches_count: int = 5
    include_explanations: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    sanitize_pii: bool = True


@dataclass(frozen=True)
class StorageConfig:
    """Storage backend settings."""

    backend: str = "memory"
    output_dir: str = "output"


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration."""

    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    section_detection: SectionDetectionConfig = field(default_factory=SectionDetectionConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    if not path.exists():
        logger.warning("Config file not found: %s — using defaults", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Override specific config values from environment variables."""
    env_log_level = os.environ.get(f"{_ENV_PREFIX}LOG_LEVEL")
    if env_log_level:
        raw.setdefault("logging", {})["level"] = env_log_level

    env_data_dir = os.environ.get(f"{_ENV_PREFIX}DATA_DIR")
    if env_data_dir:
        raw.setdefault("storage", {})["output_dir"] = env_data_dir

    return raw


# ---------------------------------------------------------------------------
# Typed construction from raw dict
# ---------------------------------------------------------------------------
def _build_config(raw: dict[str, Any]) -> AppConfig:
    """Construct a typed AppConfig from a raw dictionary."""
    proc_raw = raw.get("processing", {})
    emb_raw = raw.get("embedding", {})
    ret_raw = raw.get("retrieval", {})
    scoring_raw = raw.get("scoring", {})
    seg_raw = raw.get("segmentation", {})
    sec_raw = raw.get("section_detection", {})
    rep_raw = raw.get("reporting", {})
    log_raw = raw.get("logging", {})
    stor_raw = raw.get("storage", {})

    # Processing — convert list to tuple for frozen dataclass
    if "allowed_extensions" in proc_raw and isinstance(proc_raw["allowed_extensions"], list):
        proc_raw["allowed_extensions"] = tuple(proc_raw["allowed_extensions"])

    # Scoring — nested structure
    weights_raw = scoring_raw.get("weights", {})
    thresholds_raw = scoring_raw.get("thresholds", {})

    return AppConfig(
        processing=ProcessingConfig(**proc_raw),
        embedding=EmbeddingConfig(**emb_raw),
        retrieval=RetrievalConfig(**ret_raw),
        scoring=ScoringConfig(
            weights=ScoringWeights(**weights_raw),
            thresholds=ScoringThresholds(**thresholds_raw),
        ),
        segmentation=SegmentationConfig(**seg_raw),
        section_detection=SectionDetectionConfig(**sec_raw),
        reporting=ReportingConfig(**rep_raw),
        logging=LoggingConfig(**log_raw),
        storage=StorageConfig(**stor_raw),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config(
    config_path: str | Path | None = None,
    override_path: str | Path | None = None,
) -> AppConfig:
    """
    Load application configuration.

    Priority (highest wins):
        1. Environment variables
        2. Override YAML (e.g., test.yaml)
        3. Base YAML (e.g., default.yaml)
        4. Dataclass defaults

    Args:
        config_path: Path to base YAML config. Defaults to config/default.yaml
                     or RESUME_MATCHER_CONFIG_PATH env var.
        override_path: Optional path to override YAML (merged on top of base).

    Returns:
        Fully typed AppConfig instance.
    """
    if config_path is None:
        config_path = os.environ.get(f"{_ENV_PREFIX}CONFIG_PATH", str(_DEFAULT_CONFIG_PATH))
    config_path = Path(config_path)

    # Load base config
    raw = _load_yaml(config_path)

    # Merge override config if provided
    if override_path is not None:
        override_raw = _load_yaml(Path(override_path))
        raw = _deep_merge(raw, override_raw)

    # Apply environment variable overrides
    raw = _apply_env_overrides(raw)

    config = _build_config(raw)
    logger.info("Configuration loaded from: %s", config_path)
    return config


def setup_logging(config: AppConfig) -> None:
    """Configure the root logger based on application config."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format=config.logging.format,
        force=True,
    )
    logger.debug("Logging configured: level=%s", config.logging.level)
