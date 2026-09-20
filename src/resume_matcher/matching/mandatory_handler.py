"""
Mandatory requirement handler for Resume Matcher.

Detects mandatory requirements from JD text and tracks
which ones have matching evidence in the resume.
"""

from __future__ import annotations

import logging
import re

from resume_matcher.domain.enums import MatchStrength
from resume_matcher.domain.models import Block, MandatoryRequirement, MatchEvidence

logger = logging.getLogger(__name__)

# Patterns that indicate a requirement is mandatory
_MANDATORY_PATTERNS = [
    re.compile(r"\b(?:must\s+have|must\s+possess)\b", re.IGNORECASE),
    re.compile(r"\b(?:required|requirement|mandatory|essential)\b", re.IGNORECASE),
    re.compile(r"\b(?:minimum|at\s+least)\b", re.IGNORECASE),
    re.compile(r"\b(?:necessary|compulsory|obligatory)\b", re.IGNORECASE),
    re.compile(r"\b(?:non[- ]?negotiable)\b", re.IGNORECASE),
]

# Threshold for considering a mandatory requirement "met" (calibrated scale)
_MANDATORY_MET_THRESHOLD = 0.40


class MandatoryHandler:
    """
    Detects and tracks mandatory JD requirements.

    Identifies which JD blocks contain mandatory requirements,
    then checks whether matching evidence meets the threshold.
    """

    def detect_mandatory_blocks(self, jd_blocks: list[Block]) -> list[Block]:
        """
        Identify JD blocks that contain mandatory requirements.

        Args:
            jd_blocks: All blocks from the job description.

        Returns:
            Subset of blocks that are mandatory.
        """
        mandatory: list[Block] = []
        for block in jd_blocks:
            if self._is_mandatory(block.text):
                block.metadata["is_mandatory"] = True
                mandatory.append(block)

        logger.debug(
            "Detected %d mandatory blocks out of %d JD blocks",
            len(mandatory),
            len(jd_blocks),
        )
        return mandatory

    def evaluate_mandatory(
        self,
        mandatory_blocks: list[Block],
        evidence: list[MatchEvidence],
    ) -> list[MandatoryRequirement]:
        """
        Evaluate whether mandatory requirements are met.

        Args:
            mandatory_blocks: Blocks identified as mandatory.
            evidence: All match evidence from the evidence matcher.

        Returns:
            List of MandatoryRequirement with satisfaction status.
        """
        requirements: list[MandatoryRequirement] = []

        for block in mandatory_blocks:
            # Find best matching evidence for this JD block
            block_evidence = [
                e for e in evidence
                if e.jd_block_id == block.block_id
                and e.match_strength != MatchStrength.NONE
            ]

            if block_evidence:
                best = max(block_evidence, key=lambda e: e.similarity_score)
                is_met = best.similarity_score >= _MANDATORY_MET_THRESHOLD

                requirements.append(
                    MandatoryRequirement(
                        requirement_text=block.text[:200],
                        jd_block_id=block.block_id,
                        is_met=is_met,
                        best_match_score=best.similarity_score,
                        best_match_text=best.resume_text[:200],
                        explanation=(
                            f"{'Met' if is_met else 'NOT met'}: "
                            f"best match score {best.similarity_score:.3f} "
                            f"(threshold: {_MANDATORY_MET_THRESHOLD})"
                        ),
                    )
                )
            else:
                requirements.append(
                    MandatoryRequirement(
                        requirement_text=block.text[:200],
                        jd_block_id=block.block_id,
                        is_met=False,
                        best_match_score=0.0,
                        explanation="No matching evidence found in resume",
                    )
                )

        met_count = sum(1 for r in requirements if r.is_met)
        logger.info(
            "Mandatory requirements: %d met / %d total",
            met_count,
            len(requirements),
        )
        return requirements

    def _is_mandatory(self, text: str) -> bool:
        """Check if text contains mandatory requirement indicators."""
        return any(pattern.search(text) for pattern in _MANDATORY_PATTERNS)
