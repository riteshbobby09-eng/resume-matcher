"""
Final scorer for Resume Matcher.

Applies the exact scoring formula:
    Final = Skill × 0.30 + WorkExp × 0.15 + Education × 0.15 + Semantic × 0.40

Validates weights, handles missing evidence, and preserves
per-component contributions for auditability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from resume_matcher.config import ScoringWeights
from resume_matcher.domain.models import (
    CandidateScore,
    ComparisonParameter,
    MandatoryRequirement,
    ScoreComponent,
)
from resume_matcher.exceptions import ScoringError

logger = logging.getLogger(__name__)


class FinalScorer:
    """
    Computes the final weighted score from component scores.

    Formula: Final = Skill×0.30 + WorkExp×0.15 + Education×0.15 + Semantic×0.40
    """

    def __init__(self, weights: ScoringWeights) -> None:
        self._weights = weights

    def calculate(
        self,
        components: list[ScoreComponent],
        mandatory_requirements: list[MandatoryRequirement] | None = None,
        candidate_id: str = "",
        jd_id: str = "",
        comparison_parameters: list[ComparisonParameter] | None = None,
    ) -> CandidateScore:
        """
        Calculate the final weighted score.

        Args:
            components: List of ScoreComponent (must have all 4 components).
            mandatory_requirements: Optional mandatory requirement results.
            candidate_id: Identifier for the candidate.
            jd_id: Identifier for the job description.
            comparison_parameters: Optional comparison parameters evaluated.

        Returns:
            CandidateScore with final score and audit trail.

        Raises:
            ScoringError: If components are invalid.
        """
        # Validate components
        self._validate_components(components)

        # Compute final score
        final_score = sum(c.weighted_contribution for c in components)
        final_score = min(max(round(final_score, 2), 0.0), 100.0)

        # Determine mandatory status
        mandatory = mandatory_requirements or []
        all_met = all(r.is_met for r in mandatory) if mandatory else True

        # Collect review flags
        review_flags: list[str] = []
        for comp in components:
            if comp.raw_score == 0.0:
                review_flags.append(f"Zero score for {comp.name}: {comp.explanation}")
            if comp.missing_evidence:
                review_flags.append(
                    f"Missing evidence for {comp.name}: {len(comp.missing_evidence)} items"
                )

        if not all_met:
            unmet = [r.requirement_text[:80] for r in mandatory if not r.is_met]
            review_flags.append(
                f"Unmet mandatory requirements: {len(unmet)}"
            )

        result = CandidateScore(
            candidate_id=candidate_id,
            jd_id=jd_id,
            final_score=final_score,
            components=components,
            mandatory_requirements=mandatory,
            all_mandatory_met=all_met,
            review_flags=review_flags,
            comparison_parameters=comparison_parameters or [],
        )

        logger.info(
            "Final score: %.2f (skill=%.1f, exp=%.1f, edu=%.1f, semantic=%.1f) "
            "mandatory=%s",
            final_score,
            self._get_component_score(components, "skill"),
            self._get_component_score(components, "work_experience"),
            self._get_component_score(components, "education"),
            self._get_component_score(components, "semantic_match"),
            "all met" if all_met else f"{sum(1 for r in mandatory if r.is_met)}/{len(mandatory)} met",
        )

        return result

    def _validate_components(self, components: list[ScoreComponent]) -> None:
        """Validate that all required components are present with correct weights."""
        expected_names = {"skill", "work_experience", "education", "semantic_match"}
        actual_names = {c.name for c in components}

        missing = expected_names - actual_names
        if missing:
            raise ScoringError(
                f"Missing score components: {missing}",
                details={"expected": sorted(expected_names), "actual": sorted(actual_names)},
            )

        # Validate individual component scores
        for comp in components:
            if not (0.0 <= comp.raw_score <= 100.0):
                raise ScoringError(
                    f"Component '{comp.name}' score {comp.raw_score} is outside 0–100 range"
                )

        # Validate weight sum
        total_weight = sum(c.weight for c in components)
        if abs(total_weight - 1.0) > 1e-6:
            raise ScoringError(
                f"Component weights sum to {total_weight:.6f}, expected 1.0"
            )

    def _get_component_score(
        self, components: list[ScoreComponent], name: str
    ) -> float:
        """Get a component's raw score by name."""
        for c in components:
            if c.name == name:
                return c.raw_score
        return 0.0
