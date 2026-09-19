"""
Pydantic schemas for recruiter reports and exports.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from resume_matcher.schemas.match import MandatoryRequirementSchema, MatchEvidenceSchema
from resume_matcher.schemas.score import CandidateScoreSchema, ScoreComponentSchema


class ProcessingDetailsSchema(BaseModel):
    """Processing metadata included in every report."""

    processing_version: str
    schema_version: str
    processed_at: datetime
    config_hash: str = ""
    jd_filename: str = ""
    resume_filename: str = ""
    jd_block_count: int = Field(0, ge=0)
    resume_block_count: int = Field(0, ge=0)


class EvidenceSummarySchema(BaseModel):
    """Grouped evidence for a specific scoring component."""

    component_name: str
    score: ScoreComponentSchema
    evidence: list[MatchEvidenceSchema] = Field(default_factory=list)


class RecruiterReportSchema(BaseModel):
    """
    Complete, auditable recruiter report.

    Contains everything a recruiter needs:
    scores, evidence, flags, and a human-readable summary.
    """

    report_id: str
    processing_details: ProcessingDetailsSchema
    candidate_score: CandidateScoreSchema

    # Evidence grouped by component
    skill_evidence: EvidenceSummarySchema
    experience_evidence: EvidenceSummarySchema
    education_evidence: EvidenceSummarySchema
    semantic_evidence: EvidenceSummarySchema

    # Highlights
    strongest_matches: list[MatchEvidenceSchema] = Field(default_factory=list)
    weakest_matches: list[MatchEvidenceSchema] = Field(default_factory=list)

    # Mandatory requirements
    mandatory_requirements: list[MandatoryRequirementSchema] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)

    # Flags & warnings
    review_flags: list[str] = Field(default_factory=list)
    data_quality_warnings: list[str] = Field(default_factory=list)

    # Human-readable summary
    recruiter_summary: str = ""

    model_config = {"from_attributes": True}
