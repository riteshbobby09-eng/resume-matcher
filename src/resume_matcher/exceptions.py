"""
Custom exception hierarchy for Resume Matcher.

All application-specific exceptions inherit from ResumeMatcherError,
enabling callers to catch the base class for broad error handling or
specific subclasses for targeted handling.
"""

from __future__ import annotations


class ResumeMatcherError(Exception):
    """Base exception for all Resume Matcher errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


# ---------------------------------------------------------------------------
# Ingestion Errors
# ---------------------------------------------------------------------------
class FileValidationError(ResumeMatcherError):
    """Raised when a file fails validation checks (size, type, safety)."""


class ExtractionError(ResumeMatcherError):
    """Raised when text extraction from a document fails."""


class CorruptedDocumentError(ExtractionError):
    """Raised when a document is corrupted or unreadable."""


# ---------------------------------------------------------------------------
# Preprocessing Errors
# ---------------------------------------------------------------------------
class CleaningError(ResumeMatcherError):
    """Raised when text cleaning encounters an unrecoverable issue."""


class NormalizationError(ResumeMatcherError):
    """Raised when text normalization fails."""


# ---------------------------------------------------------------------------
# Segmentation Errors
# ---------------------------------------------------------------------------
class SegmentationError(ResumeMatcherError):
    """Raised when sentence segmentation or block building fails."""


class BlockBuildingError(SegmentationError):
    """Raised when block creation encounters invalid input."""


# ---------------------------------------------------------------------------
# Metadata Errors
# ---------------------------------------------------------------------------
class SectionDetectionError(ResumeMatcherError):
    """Raised when section detection encounters an issue."""


class SectionInferenceError(ResumeMatcherError):
    """Raised when section inference cannot determine a section."""


# ---------------------------------------------------------------------------
# Embedding & Retrieval Errors
# ---------------------------------------------------------------------------
class EmbeddingError(ResumeMatcherError):
    """Raised when embedding generation fails."""


class ModelLoadError(EmbeddingError):
    """Raised when the embedding model cannot be loaded."""


class IndexError(ResumeMatcherError):
    """Raised when FAISS index operations fail."""


class RetrievalError(ResumeMatcherError):
    """Raised when vector retrieval fails."""


# ---------------------------------------------------------------------------
# Matching & Scoring Errors
# ---------------------------------------------------------------------------
class MatchingError(ResumeMatcherError):
    """Raised when evidence matching fails."""


class ScoringError(ResumeMatcherError):
    """Raised when score calculation encounters invalid state."""


class WeightValidationError(ScoringError):
    """Raised when scoring weights are invalid (don't sum to 1.0)."""


# ---------------------------------------------------------------------------
# Reporting Errors
# ---------------------------------------------------------------------------
class ReportingError(ResumeMatcherError):
    """Raised when report generation fails."""


class ExportError(ReportingError):
    """Raised when report export fails."""


# ---------------------------------------------------------------------------
# Configuration Errors
# ---------------------------------------------------------------------------
class ConfigurationError(ResumeMatcherError):
    """Raised when configuration is invalid or missing."""


# ---------------------------------------------------------------------------
# Repository / Storage Errors
# ---------------------------------------------------------------------------
class StorageError(ResumeMatcherError):
    """Raised when storage operations fail."""


class DocumentNotFoundError(StorageError):
    """Raised when a requested document is not found in storage."""
