"""
Pydantic schemas for scoring.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class ScoreComponentSchema(BaseModel):
    """Schema for a single scoring component."""

    name: str
    raw_score: float = Field(..., ge=0.0, le=100.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    weighted_contribution: float = Field(..., ge=0.0, le=100.0)
    evidence_count: int = Field(0, ge=0)
    explanation: str = ""
    missing_evidence: list[str] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class CandidateScoreSchema(BaseModel):
    """Schema for a complete candidate scoring result."""

    candidate_id: str
    jd_id: str
    final_score: float = Field(..., ge=0.0, le=100.0)
    components: list[ScoreComponentSchema]
    all_mandatory_met: bool = True
    review_flags: list[str] = Field(default_factory=list)
    processing_version: str = "0.1.0"
    scored_at: datetime

    @model_validator(mode="after")
    def validate_components(self) -> "CandidateScoreSchema":
        """Verify component weights sum to 1.0 and contributions match final score."""
        if self.components:
            total_weight = sum(c.weight for c in self.components)
            if abs(total_weight - 1.0) > 1e-6:
                raise ValueError(
                    f"Component weights must sum to 1.0, got {total_weight:.6f}"
                )
            expected_final = sum(c.weighted_contribution for c in self.components)
            if abs(expected_final - self.final_score) > 0.1:
                raise ValueError(
                    f"Final score ({self.final_score:.2f}) does not match "
                    f"sum of contributions ({expected_final:.2f})"
                )
        return self

    model_config = {"from_attributes": True}
