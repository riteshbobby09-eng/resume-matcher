"""
Domain models for Resume Matcher.

Pure data containers (dataclasses) representing the core entities that flow
through the processing pipeline.  No I/O, no business logic.

Traceability chain:
    Document → IndexedLine → Sentence → Block → MatchEvidence → ScoreComponent → CandidateScore
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from resume_matcher.domain.enums import (
    DocumentType,
    FileFormat,
    MatchStrength,
    ProcessingStage,
    SectionType,
    ValidationStatus,
)


# ---------------------------------------------------------------------------
# Document-level metadata
# ---------------------------------------------------------------------------
@dataclass
class DocumentMeta:
    """Metadata about a source document."""

    document_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_filename: str = ""
    sanitized_filename: str = ""
    document_type: DocumentType = DocumentType.RESUME
    file_format: FileFormat = FileFormat.PDF
    file_size_bytes: int = 0
    page_count: int = 0
    processing_version: str = "0.1.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_stage: ProcessingStage = ProcessingStage.INGESTED
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Line-level (traceability anchor)
# ---------------------------------------------------------------------------
@dataclass
class IndexedLine:
    """
    A single line of text with full source traceability.

    This is the fundamental unit of traceability — every sentence, block,
    and match can be traced back to specific IndexedLines.
    """

    global_line_number: int
    text: str
    page_number: int = 1
    local_line_number: int = 0
    source_filename: str = ""
    document_id: str = ""
    is_empty: bool = False


# ---------------------------------------------------------------------------
# Sentence-level
# ---------------------------------------------------------------------------
@dataclass
class Sentence:
    """
    A sentence extracted from one or more IndexedLines.

    Preserves references to source lines for traceability.
    """

    sentence_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    text: str = ""
    source_line_numbers: list[int] = field(default_factory=list)
    start_char_offset: int = 0
    end_char_offset: int = 0
    document_id: str = ""


# ---------------------------------------------------------------------------
# Block-level (meaningful unit)
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """
    A meaningful unit of information composed of one or more sentences.

    Blocks are the primary unit for embedding and matching.
    """

    block_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    sentences: list[Sentence] = field(default_factory=list)
    text: str = ""  # concatenated sentence texts
    section: SectionType = SectionType.UNKNOWN
    document_id: str = ""
    document_type: DocumentType = DocumentType.RESUME
    source_line_numbers: list[int] = field(default_factory=list)
    position_in_document: float = 0.0  # 0.0 = start, 1.0 = end
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-compute text and line numbers from sentences if not set."""
        if not self.text and self.sentences:
            self.text = " ".join(s.text for s in self.sentences)
        if not self.source_line_numbers and self.sentences:
            all_lines: list[int] = []
            for s in self.sentences:
                all_lines.extend(s.source_line_numbers)
            self.source_line_numbers = sorted(set(all_lines))


# ---------------------------------------------------------------------------
# Section inference
# ---------------------------------------------------------------------------
@dataclass
class SectionInference:
    """
    Result of inferring a section for a block that has no explicit heading.

    Provides explainability: section name, confidence, reasoning, and
    whether a human reviewer should check it.
    """

    section: SectionType = SectionType.UNKNOWN
    confidence: float = 0.0  # 0.0 – 1.0
    reasoning: str = ""
    review_flag: bool = False
    evidence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Match evidence
# ---------------------------------------------------------------------------
@dataclass
class MatchEvidence:
    """
    Evidence linking a JD block to a resume block.

    Contains everything needed for audit: matched text, scores, source
    locations, strength classification, and explanations.
    """

    evidence_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    jd_block_id: str = ""
    resume_block_id: str = ""
    jd_text: str = ""
    resume_text: str = ""
    similarity_score: float = 0.0
    jd_section: SectionType = SectionType.UNKNOWN
    resume_section: SectionType = SectionType.UNKNOWN
    jd_line_numbers: list[int] = field(default_factory=list)
    resume_line_numbers: list[int] = field(default_factory=list)
    match_strength: MatchStrength = MatchStrength.NONE
    explanation: str = ""
    uncertainty: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class ScoreComponent:
    """A single scoring component with its weight and contribution."""

    name: str = ""
    raw_score: float = 0.0  # 0 – 100
    weight: float = 0.0  # 0.0 – 1.0
    weighted_contribution: float = 0.0  # raw_score × weight
    evidence_count: int = 0
    explanation: str = ""
    missing_evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Auto-compute weighted contribution."""
        self.weighted_contribution = self.raw_score * self.weight


@dataclass
class MandatoryRequirement:
    """Tracks whether a mandatory JD requirement is satisfied."""

    requirement_text: str = ""
    jd_block_id: str = ""
    is_met: bool = False
    best_match_score: float = 0.0
    best_match_text: str = ""
    explanation: str = ""


@dataclass
class CandidateScore:
    """Complete scoring result for a single candidate–JD pair."""

    candidate_id: str = ""
    jd_id: str = ""
    final_score: float = 0.0  # 0 – 100
    components: list[ScoreComponent] = field(default_factory=list)
    mandatory_requirements: list[MandatoryRequirement] = field(default_factory=list)
    all_mandatory_met: bool = True
    review_flags: list[str] = field(default_factory=list)
    processing_version: str = "0.1.0"
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationMessage:
    """A single validation finding."""

    status: ValidationStatus = ValidationStatus.VALID
    code: str = ""
    message: str = ""
    location: str = ""  # e.g., "block:abc123" or "line:42"


@dataclass
class QualityReport:
    """Quality check results for a processed document."""

    document_id: str = ""
    overall_status: ValidationStatus = ValidationStatus.VALID
    messages: list[ValidationMessage] = field(default_factory=list)
    total_lines: int = 0
    total_sentences: int = 0
    total_blocks: int = 0
    sections_found: list[SectionType] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Processed Document (aggregate)
# ---------------------------------------------------------------------------
@dataclass
class ProcessedDocument:
    """
    A fully processed document ready for embedding and matching.

    This is the aggregate output of the ingestion pipeline, containing
    the complete traceability chain from raw lines to validated blocks.
    """

    meta: DocumentMeta = field(default_factory=DocumentMeta)
    lines: list[IndexedLine] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    quality_report: QualityReport | None = None
    raw_text: str = ""
