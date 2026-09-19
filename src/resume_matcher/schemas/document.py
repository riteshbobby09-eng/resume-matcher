"""
Pydantic schemas for documents.

Used for input validation (files coming in), output serialization
(processed documents going out), and API contracts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from resume_matcher.domain.enums import DocumentType, FileFormat, ProcessingStage


class DocumentInputSchema(BaseModel):
    """Validates an incoming document before processing."""

    source_filename: str = Field(..., min_length=1, max_length=500)
    document_type: DocumentType
    file_format: FileFormat | None = None

    @field_validator("source_filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Reject dangerous path characters."""
        forbidden = ["..", "//", "\\", "\x00"]
        for char in forbidden:
            if char in v:
                raise ValueError(f"Filename contains forbidden characters: {char!r}")
        return v.strip()


class IndexedLineSchema(BaseModel):
    """Schema for a single indexed line."""

    global_line_number: int = Field(..., ge=1)
    text: str
    page_number: int = Field(1, ge=1)
    local_line_number: int = Field(0, ge=0)
    source_filename: str = ""
    document_id: str = ""
    is_empty: bool = False


class DocumentMetaSchema(BaseModel):
    """Schema for document metadata."""

    document_id: str = Field(..., min_length=1)
    source_filename: str = ""
    sanitized_filename: str = ""
    document_type: DocumentType
    file_format: FileFormat
    file_size_bytes: int = Field(0, ge=0)
    page_count: int = Field(0, ge=0)
    processing_version: str = "0.1.0"
    created_at: datetime
    processing_stage: ProcessingStage
    extra: dict[str, Any] = Field(default_factory=dict)


class ProcessedDocumentSchema(BaseModel):
    """Schema for a fully processed document (output)."""

    meta: DocumentMetaSchema
    total_lines: int = Field(0, ge=0)
    total_sentences: int = Field(0, ge=0)
    total_blocks: int = Field(0, ge=0)
    raw_text_length: int = Field(0, ge=0)
    has_quality_report: bool = False

    model_config = {"from_attributes": True}
