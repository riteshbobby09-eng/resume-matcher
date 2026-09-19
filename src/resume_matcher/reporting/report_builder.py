"""
Report builder for Resume Matcher.

Assembles a complete, auditable recruiter report from:
- Candidate & JD identifiers
- Processing details
- Final and component scores with contributions
- Evidence grouped by component
- Strongest/weakest matches
- Mandatory requirement status
- Review flags and quality warnings
- Human-readable recruiter summary
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone

from resume_matcher.config import AppConfig
from resume_matcher.domain.enums import MatchStrength, SectionType
from resume_matcher.domain.models import (
    Block,
    CandidateScore,
    MatchEvidence,
    ProcessedDocument,
    ScoreComponent,
)

logger = logging.getLogger(__name__)


class RecruiterReport:
    """A complete recruiter report (plain data container for serialization)."""

    def __init__(self) -> None:
        self.report_id: str = uuid.uuid4().hex[:12]
        self.processing_details: dict = {}
        self.candidate_score: dict = {}
        self.skill_evidence: dict = {}
        self.experience_evidence: dict = {}
        self.education_evidence: dict = {}
        self.semantic_evidence: dict = {}
        self.strongest_matches: list[dict] = []
        self.weakest_matches: list[dict] = []
        self.mandatory_requirements: list[dict] = []
        self.missing_requirements: list[str] = []
        self.review_flags: list[str] = []
        self.data_quality_warnings: list[str] = []
        self.recruiter_summary: str = ""


class ReportBuilder:
    """
    Assembles complete recruiter reports.

    Takes processing results and constructs a structured report
    suitable for JSON export.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._top_n = config.reporting.top_matches_count
        self._bottom_n = config.reporting.bottom_matches_count

    def build_report(
        self,
        jd_doc: ProcessedDocument,
        resume_doc: ProcessedDocument,
        candidate_score: CandidateScore,
        evidence: list[MatchEvidence],
    ) -> RecruiterReport:
        """
        Build a complete recruiter report.

        Args:
            jd_doc: Processed job description.
            resume_doc: Processed resume.
            candidate_score: Final scoring results.
            evidence: All match evidence.

        Returns:
            RecruiterReport ready for export.
        """
        report = RecruiterReport()

        # Processing details
        config_str = f"{self._config.scoring.weights}"
        report.processing_details = {
            "processing_version": candidate_score.processing_version,
            "schema_version": self._config.reporting.schema_version,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": hashlib.md5(config_str.encode()).hexdigest()[:8],
            "jd_filename": jd_doc.meta.sanitized_filename,
            "resume_filename": resume_doc.meta.sanitized_filename,
            "jd_block_count": len(jd_doc.blocks),
            "resume_block_count": len(resume_doc.blocks),
        }

        # Scores
        report.candidate_score = {
            "candidate_id": candidate_score.candidate_id,
            "jd_id": candidate_score.jd_id,
            "final_score": candidate_score.final_score,
            "components": [
                {
                    "name": c.name,
                    "raw_score": c.raw_score,
                    "weight": c.weight,
                    "weighted_contribution": c.weighted_contribution,
                    "evidence_count": c.evidence_count,
                    "explanation": c.explanation,
                    "missing_evidence": c.missing_evidence,
                }
                for c in candidate_score.components
            ],
            "all_mandatory_met": candidate_score.all_mandatory_met,
        }

        # Group evidence by component
        report.skill_evidence = self._build_evidence_group(
            "skill", candidate_score, evidence, jd_doc.blocks
        )
        report.experience_evidence = self._build_evidence_group(
            "work_experience", candidate_score, evidence, jd_doc.blocks
        )
        report.education_evidence = self._build_evidence_group(
            "education", candidate_score, evidence, jd_doc.blocks
        )
        report.semantic_evidence = self._build_evidence_group(
            "semantic_match", candidate_score, evidence, jd_doc.blocks
        )

        # Strongest and weakest matches
        matched = [e for e in evidence if e.match_strength != MatchStrength.NONE]
        matched_sorted = sorted(matched, key=lambda e: e.similarity_score, reverse=True)
        report.strongest_matches = [
            self._evidence_to_dict(e) for e in matched_sorted[: self._top_n]
        ]
        report.weakest_matches = [
            self._evidence_to_dict(e) for e in matched_sorted[-self._bottom_n:]
        ] if len(matched_sorted) > self._bottom_n else [
            self._evidence_to_dict(e) for e in matched_sorted
        ]

        # Mandatory requirements
        report.mandatory_requirements = [
            {
                "requirement_text": r.requirement_text,
                "is_met": r.is_met,
                "best_match_score": r.best_match_score,
                "best_match_text": r.best_match_text,
                "explanation": r.explanation,
            }
            for r in candidate_score.mandatory_requirements
        ]
        report.missing_requirements = [
            r.requirement_text
            for r in candidate_score.mandatory_requirements
            if not r.is_met
        ]

        # Flags and warnings
        report.review_flags = list(candidate_score.review_flags)
        report.data_quality_warnings = self._collect_quality_warnings(jd_doc, resume_doc)

        # Recruiter summary
        report.recruiter_summary = self._generate_summary(candidate_score, evidence)

        logger.info("Report built: %s", report.report_id)
        return report

    def _build_evidence_group(
        self,
        component_name: str,
        candidate_score: CandidateScore,
        evidence: list[MatchEvidence],
        jd_blocks: list[Block],
    ) -> dict:
        """Build evidence group for a scoring component."""
        # Find the component
        comp = next(
            (c for c in candidate_score.components if c.name == component_name),
            None,
        )
        if not comp:
            return {"component_name": component_name, "score": {}, "evidence": []}

        # Filter evidence relevant to this component's sections
        section_map = {
            "skill": {SectionType.SKILLS, SectionType.PROJECTS, SectionType.CERTIFICATIONS, SectionType.REQUIREMENTS},
            "work_experience": {SectionType.EXPERIENCE, SectionType.PROJECTS, SectionType.ACHIEVEMENTS, SectionType.RESPONSIBILITIES},
            "education": {SectionType.EDUCATION, SectionType.CERTIFICATIONS},
            "semantic_match": set(SectionType),
        }
        relevant_sections = section_map.get(component_name, set(SectionType))
        relevant_jd_ids = {
            b.block_id for b in jd_blocks if b.section in relevant_sections
        }

        if component_name == "semantic_match":
            relevant_jd_ids = {b.block_id for b in jd_blocks}

        component_evidence = [
            e for e in evidence
            if e.jd_block_id in relevant_jd_ids
            and e.match_strength != MatchStrength.NONE
        ]

        return {
            "component_name": component_name,
            "score": {
                "raw_score": comp.raw_score,
                "weight": comp.weight,
                "weighted_contribution": comp.weighted_contribution,
            },
            "evidence": [self._evidence_to_dict(e) for e in component_evidence[:10]],
        }

    def _evidence_to_dict(self, e: MatchEvidence) -> dict:
        """Convert MatchEvidence to a serializable dict."""
        return {
            "evidence_id": e.evidence_id,
            "jd_text": e.jd_text[:200],
            "resume_text": e.resume_text[:200],
            "similarity_score": round(e.similarity_score, 4),
            "jd_section": e.jd_section.value,
            "resume_section": e.resume_section.value,
            "jd_line_numbers": e.jd_line_numbers,
            "resume_line_numbers": e.resume_line_numbers,
            "match_strength": e.match_strength.value,
            "explanation": e.explanation,
            "uncertainty": e.uncertainty,
        }

    def _collect_quality_warnings(
        self, jd_doc: ProcessedDocument, resume_doc: ProcessedDocument
    ) -> list[str]:
        """Collect quality warnings from both documents."""
        warnings: list[str] = []
        for doc, label in [(jd_doc, "JD"), (resume_doc, "Resume")]:
            if doc.quality_report:
                for msg in doc.quality_report.messages:
                    if msg.status.value in ("warning", "error"):
                        warnings.append(f"[{label}] {msg.message}")
        return warnings

    def _generate_summary(
        self, score: CandidateScore, evidence: list[MatchEvidence]
    ) -> str:
        """Generate a recruiter-friendly summary paragraph."""
        final = score.final_score
        strong = sum(1 for e in evidence if e.match_strength == MatchStrength.STRONG)
        moderate = sum(1 for e in evidence if e.match_strength == MatchStrength.MODERATE)
        weak = sum(1 for e in evidence if e.match_strength == MatchStrength.WEAK)

        # Rating
        if final >= 75:
            rating = "Strong Candidate"
        elif final >= 55:
            rating = "Moderate Candidate"
        elif final >= 35:
            rating = "Below Average Candidate"
        else:
            rating = "Weak Candidate"

        parts = [
            f"Overall Assessment: {rating} (Score: {final:.1f}/100).",
            f"The candidate shows {strong} strong, {moderate} moderate, "
            f"and {weak} weak matches against the job requirements.",
        ]

        # Component highlights
        for comp in score.components:
            parts.append(
                f"{comp.name.replace('_', ' ').title()}: {comp.raw_score:.1f}/100 "
                f"(weight: {comp.weight:.0%})."
            )

        # Mandatory status
        if score.mandatory_requirements:
            met = sum(1 for r in score.mandatory_requirements if r.is_met)
            total = len(score.mandatory_requirements)
            parts.append(f"Mandatory Requirements: {met}/{total} met.")

        if not score.all_mandatory_met:
            parts.append(
                "⚠ WARNING: Not all mandatory requirements are satisfied. "
                "Review required before proceeding."
            )

        return " ".join(parts)
