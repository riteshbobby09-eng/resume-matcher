"""
Component scorers for Resume Matcher.

Four independent scorers, each producing a 0–100 score:
1. SkillScorer — SKILLS-section match quality and coverage
2. WorkExperienceScorer — EXPERIENCE-section match quality
3. EducationScorer — EDUCATION-section match quality
4. SemanticMatchScorer — Overall semantic similarity across all sections

Projects, certifications, and achievements contribute to
existing component scores (not separate components).
"""

from __future__ import annotations

import logging
from typing import Callable

from resume_matcher.domain.enums import MatchStrength, SectionType
from resume_matcher.domain.models import Block, MatchEvidence, ScoreComponent

logger = logging.getLogger(__name__)

# Sections that contribute supporting evidence to components
_SUPPORTING_SECTIONS: dict[str, list[SectionType]] = {
    "skill": [SectionType.SKILLS, SectionType.PROJECTS, SectionType.CERTIFICATIONS],
    "work_experience": [SectionType.EXPERIENCE, SectionType.PROJECTS, SectionType.ACHIEVEMENTS],
    "education": [SectionType.EDUCATION, SectionType.CERTIFICATIONS],
}


def _safe_mean(values: list[float]) -> float:
    """Compute mean of values, return 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0


class SkillScorer:
    """
    Scores skill match quality.

    Considers:
    - Average similarity of JD skill blocks matched to resume
    - Coverage ratio (how many JD skill blocks have matches)
    - Supporting evidence from PROJECTS and CERTIFICATIONS
    """

    def score(
        self,
        evidence: list[MatchEvidence],
        jd_blocks: list[Block],
        weight: float = 0.30,
    ) -> ScoreComponent:
        """Calculate skill score (0–100)."""
        # Filter evidence for skill-relevant sections
        relevant_jd = [
            b for b in jd_blocks
            if b.section in _SUPPORTING_SECTIONS["skill"]
            or b.section == SectionType.REQUIREMENTS
        ]

        if not relevant_jd:
            return ScoreComponent(
                name="skill",
                raw_score=0.0,
                weight=weight,
                explanation="No skill-related requirements found in JD",
                missing_evidence=["No skill requirements in JD"],
            )

        relevant_jd_ids = {b.block_id for b in relevant_jd}
        skill_evidence = [
            e for e in evidence
            if e.jd_block_id in relevant_jd_ids
            and e.match_strength != MatchStrength.NONE
        ]

        # Best match per JD block
        best_per_jd: dict[str, float] = {}
        for e in skill_evidence:
            current = best_per_jd.get(e.jd_block_id, 0.0)
            best_per_jd[e.jd_block_id] = max(current, e.similarity_score)

        # Quality: average of best matches
        match_quality = _safe_mean(list(best_per_jd.values())) if best_per_jd else 0.0

        # Coverage: ratio of JD blocks that have any match
        coverage = len(best_per_jd) / len(relevant_jd) if relevant_jd else 0.0

        # Combined score: 60% quality + 40% coverage, scaled to 100
        raw_score = (match_quality * 0.6 + coverage * 0.4) * 100
        raw_score = min(max(raw_score, 0.0), 100.0)

        # Identify missing evidence
        matched_ids = set(best_per_jd.keys())
        missing = [
            b.text[:100] for b in relevant_jd
            if b.block_id not in matched_ids
        ]

        return ScoreComponent(
            name="skill",
            raw_score=round(raw_score, 2),
            weight=weight,
            evidence_count=len(skill_evidence),
            explanation=(
                f"Quality: {match_quality:.3f}, Coverage: {coverage:.1%}, "
                f"Matched {len(best_per_jd)}/{len(relevant_jd)} skill requirements"
            ),
            missing_evidence=missing[:5],
        )


class WorkExperienceScorer:
    """
    Scores work experience match quality.

    Considers EXPERIENCE section matches plus supporting
    evidence from PROJECTS and ACHIEVEMENTS.
    """

    def score(
        self,
        evidence: list[MatchEvidence],
        jd_blocks: list[Block],
        weight: float = 0.15,
    ) -> ScoreComponent:
        """Calculate work experience score (0–100)."""
        relevant_jd = [
            b for b in jd_blocks
            if b.section in _SUPPORTING_SECTIONS["work_experience"]
            or b.section == SectionType.RESPONSIBILITIES
        ]

        if not relevant_jd:
            return ScoreComponent(
                name="work_experience",
                raw_score=0.0,
                weight=weight,
                explanation="No experience-related requirements found in JD",
                missing_evidence=["No experience requirements in JD"],
            )

        relevant_jd_ids = {b.block_id for b in relevant_jd}
        exp_evidence = [
            e for e in evidence
            if e.jd_block_id in relevant_jd_ids
            and e.match_strength != MatchStrength.NONE
        ]

        best_per_jd: dict[str, float] = {}
        for e in exp_evidence:
            current = best_per_jd.get(e.jd_block_id, 0.0)
            best_per_jd[e.jd_block_id] = max(current, e.similarity_score)

        match_quality = _safe_mean(list(best_per_jd.values())) if best_per_jd else 0.0
        coverage = len(best_per_jd) / len(relevant_jd) if relevant_jd else 0.0
        raw_score = (match_quality * 0.6 + coverage * 0.4) * 100
        raw_score = min(max(raw_score, 0.0), 100.0)

        missing = [
            b.text[:100] for b in relevant_jd
            if b.block_id not in set(best_per_jd.keys())
        ]

        return ScoreComponent(
            name="work_experience",
            raw_score=round(raw_score, 2),
            weight=weight,
            evidence_count=len(exp_evidence),
            explanation=(
                f"Quality: {match_quality:.3f}, Coverage: {coverage:.1%}, "
                f"Matched {len(best_per_jd)}/{len(relevant_jd)} experience requirements"
            ),
            missing_evidence=missing[:5],
        )


class EducationScorer:
    """
    Scores education match quality.

    Considers EDUCATION section matches plus supporting
    evidence from CERTIFICATIONS.
    """

    def score(
        self,
        evidence: list[MatchEvidence],
        jd_blocks: list[Block],
        weight: float = 0.15,
    ) -> ScoreComponent:
        """Calculate education score (0–100)."""
        relevant_jd = [
            b for b in jd_blocks
            if b.section in _SUPPORTING_SECTIONS["education"]
        ]

        if not relevant_jd:
            return ScoreComponent(
                name="education",
                raw_score=0.0,
                weight=weight,
                explanation="No education-related requirements found in JD",
                missing_evidence=["No education requirements in JD"],
            )

        relevant_jd_ids = {b.block_id for b in relevant_jd}
        edu_evidence = [
            e for e in evidence
            if e.jd_block_id in relevant_jd_ids
            and e.match_strength != MatchStrength.NONE
        ]

        best_per_jd: dict[str, float] = {}
        for e in edu_evidence:
            current = best_per_jd.get(e.jd_block_id, 0.0)
            best_per_jd[e.jd_block_id] = max(current, e.similarity_score)

        match_quality = _safe_mean(list(best_per_jd.values())) if best_per_jd else 0.0
        coverage = len(best_per_jd) / len(relevant_jd) if relevant_jd else 0.0
        raw_score = (match_quality * 0.6 + coverage * 0.4) * 100
        raw_score = min(max(raw_score, 0.0), 100.0)

        missing = [
            b.text[:100] for b in relevant_jd
            if b.block_id not in set(best_per_jd.keys())
        ]

        return ScoreComponent(
            name="education",
            raw_score=round(raw_score, 2),
            weight=weight,
            evidence_count=len(edu_evidence),
            explanation=(
                f"Quality: {match_quality:.3f}, Coverage: {coverage:.1%}, "
                f"Matched {len(best_per_jd)}/{len(relevant_jd)} education requirements"
            ),
            missing_evidence=missing[:5],
        )


class SemanticMatchScorer:
    """
    Scores overall semantic similarity across requirement sections.

    Filters out non-requirement sections (ABOUT, CONTACT, OTHER) so that
    company self-descriptions and contact details don't skew the score.
    """

    def score(
        self,
        evidence: list[MatchEvidence],
        jd_blocks: list[Block],
        weight: float = 0.40,
    ) -> ScoreComponent:
        """Calculate overall semantic match score (0–100)."""
        if not jd_blocks:
            return ScoreComponent(
                name="semantic_match",
                raw_score=0.0,
                weight=weight,
                explanation="No JD blocks to match against",
            )

        # Filter out non-requirement JD blocks (ABOUT, CONTACT, OTHER)
        # unless they are marked mandatory
        requirement_jd = [
            b for b in jd_blocks
            if b.section not in (SectionType.ABOUT, SectionType.CONTACT, SectionType.OTHER)
            or b.metadata.get("is_mandatory")
        ]
        target_blocks = requirement_jd if requirement_jd else jd_blocks
        target_ids = {b.block_id for b in target_blocks}

        # Consider evidence matching requirement blocks and excluding contact resume blocks
        matched_evidence = [
            e for e in evidence
            if e.jd_block_id in target_ids
            and e.match_strength != MatchStrength.NONE
            and e.resume_section != SectionType.CONTACT
        ]

        # Best match per JD block (across requirement sections)
        best_per_jd: dict[str, float] = {}
        for e in matched_evidence:
            current = best_per_jd.get(e.jd_block_id, 0.0)
            best_per_jd[e.jd_block_id] = max(current, e.similarity_score)

        match_quality = _safe_mean(list(best_per_jd.values())) if best_per_jd else 0.0
        coverage = len(best_per_jd) / len(target_blocks) if target_blocks else 0.0
        raw_score = (match_quality * 0.5 + coverage * 0.5) * 100
        raw_score = min(max(raw_score, 0.0), 100.0)

        return ScoreComponent(
            name="semantic_match",
            raw_score=round(raw_score, 2),
            weight=weight,
            evidence_count=len(matched_evidence),
            explanation=(
                f"Overall quality: {match_quality:.3f}, "
                f"Overall coverage: {coverage:.1%}, "
                f"Matched {len(best_per_jd)}/{len(target_blocks)} requirement blocks"
            ),
        )
