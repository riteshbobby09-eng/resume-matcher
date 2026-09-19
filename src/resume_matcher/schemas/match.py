"""
Pydantic schemas for match evidence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from resume_matcher.domain.enums import MatchStrength, SectionType


class MatchEvidenceSchema(BaseModel):
    """Schema for a single piece of match evidence between JD and resume."""

    evidence_id: str
    jd_block_id: str
    resume_block_id: str
    jd_text: str
    resume_text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    jd_section: SectionType = SectionType.UNKNOWN
    resume_section: SectionType = SectionType.UNKNOWN
    jd_line_numbers: list[int] = Field(default_factory=list)
    resume_line_numbers: list[int] = Field(default_factory=list)
    match_strength: MatchStrength = MatchStrength.NONE
    explanation: str = ""
    uncertainty: bool = False

    model_config = {"from_attributes": True}


class MandatoryRequirementSchema(BaseModel):
    """Schema for tracking mandatory requirement satisfaction."""

    requirement_text: str
    jd_block_id: str
    is_met: bool = False
    best_match_score: float = Field(0.0, ge=0.0, le=1.0)
    best_match_text: str = ""
    explanation: str = ""

    model_config = {"from_attributes": True}
