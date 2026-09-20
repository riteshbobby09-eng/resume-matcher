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
import re
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


def _weighted_coverage(best_scores: list[float], total_count: int) -> float:
    """Calculate weighted coverage giving graduated credit for match strength."""
    if not total_count:
        return 0.0
    return sum(
        1.0 if s >= 0.70 else (0.70 if s >= 0.50 else (0.35 if s >= 0.30 else 0.0))
        for s in best_scores
    ) / total_count


def _estimate_cand_years(blocks: list[Block]) -> int:
    """Estimate total years of candidate experience from block text."""
    full_text = " ".join(b.text for b in blocks)
    m = re.search(r"(\d+)\+?\s*years?\s*(?:of\s*)?experience", full_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    years = [int(y) for y in re.findall(r"\b(20\d\d|19\d\d)\b", full_text)]
    if years:
        min_yr = min(years)
        max_yr = 2026 if "present" in full_text.lower() or "current" in full_text.lower() else max(years)
        return max(1, max_yr - min_yr)
    return 1


def _extract_req_years(jd_blocks: list[Block]) -> int:
    """Extract required years of experience from JD text."""
    full_text = " ".join(b.text for b in jd_blocks)
    m = re.search(r"(\d+)\+?\s*years?", full_text, re.IGNORECASE)
    return int(m.group(1)) if m else 0


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
        weight: float = 0.15,
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

        # Weighted coverage: strong=1.0, moderate=0.7, weak=0.35
        coverage = _weighted_coverage(list(best_per_jd.values()), len(relevant_jd))

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
        weight: float = 0.35,
        resume_blocks: list[Block] | None = None,
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
        coverage = _weighted_coverage(list(best_per_jd.values()), len(relevant_jd))
        raw_score = (match_quality * 0.6 + coverage * 0.4) * 100

        # Seniority duration scaling factor if candidate experience is below required
        if resume_blocks:
            req_years = _extract_req_years(jd_blocks)
            cand_years = _estimate_cand_years(resume_blocks)
            if req_years > 0 and cand_years < req_years:
                raw_score *= min(1.0, max(0.40, cand_years / req_years))

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
        coverage = _weighted_coverage(list(best_per_jd.values()), len(relevant_jd))
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


class ContentScorer:
    """
    Scores content match quality across sections not covered by Skill, Experience, and Education.

    Covers: Summary, Projects, Certifications, Achievements, Objective,
    Responsibilities, Requirements, About, Other, Unknown, and Content.
    """

    def score(
        self,
        evidence: list[MatchEvidence],
        jd_blocks: list[Block],
        weight: float = 0.35,
    ) -> ScoreComponent:
        """Calculate content match score (0–100)."""
        if not jd_blocks:
            return ScoreComponent(
                name="content_score",
                raw_score=0.0,
                weight=weight,
                explanation="No JD blocks to match against",
            )

        # Target blocks outside of primary 3 (or all non-contact blocks)
        target_blocks = [
            b for b in jd_blocks
            if b.section not in (SectionType.SKILLS, SectionType.EXPERIENCE, SectionType.EDUCATION, SectionType.CONTACT)
        ]
        # If no non-core blocks exist in JD, evaluate all requirement blocks
        if not target_blocks:
            target_blocks = [
                b for b in jd_blocks
                if b.section != SectionType.CONTACT
            ]
        target_ids = {b.block_id for b in target_blocks}

        # Consider evidence matching target blocks and excluding contact resume blocks
        matched_evidence = [
            e for e in evidence
            if e.jd_block_id in target_ids
            and e.match_strength != MatchStrength.NONE
            and e.resume_section != SectionType.CONTACT
        ]

        # Best match per JD block
        best_per_jd: dict[str, float] = {}
        for e in matched_evidence:
            current = best_per_jd.get(e.jd_block_id, 0.0)
            best_per_jd[e.jd_block_id] = max(current, e.similarity_score)

        match_quality = _safe_mean(list(best_per_jd.values())) if best_per_jd else 0.0
        coverage = _weighted_coverage(list(best_per_jd.values()), len(target_blocks))
        raw_score = (match_quality * 0.5 + coverage * 0.5) * 100
        raw_score = min(max(raw_score, 0.0), 100.0)

        return ScoreComponent(
            name="content_score",
            raw_score=round(raw_score, 2),
            weight=weight,
            evidence_count=len(matched_evidence),
            explanation=(
                f"Content quality: {match_quality:.3f}, "
                f"Content coverage: {coverage:.1%}, "
                f"Matched {len(best_per_jd)}/{len(target_blocks)} content blocks"
            ),
        )


# Backward compatibility alias
SemanticMatchScorer = ContentScorer
