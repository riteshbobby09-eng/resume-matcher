"""
Pydantic schemas for blocks and sections.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from resume_matcher.domain.enums import DocumentType, SectionType


class SentenceSchema(BaseModel):
    """Schema for a sentence within a block."""

    sentence_id: str
    text: str
    source_line_numbers: list[int] = Field(default_factory=list)
    document_id: str = ""


class SectionInferenceSchema(BaseModel):
    """Schema for section inference results — explainability output."""

    section: SectionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    review_flag: bool = False
    evidence: list[str] = Field(default_factory=list)


class BlockSchema(BaseModel):
    """Schema for a content block."""

    block_id: str
    text: str
    section: SectionType = SectionType.UNKNOWN
    document_id: str = ""
    document_type: DocumentType = DocumentType.RESUME
    source_line_numbers: list[int] = Field(default_factory=list)
    position_in_document: float = Field(0.0, ge=0.0, le=1.0)
    sentence_count: int = Field(0, ge=0)
    inference: SectionInferenceSchema | None = None

    model_config = {"from_attributes": True}
