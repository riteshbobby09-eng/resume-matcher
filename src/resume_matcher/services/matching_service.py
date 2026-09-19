"""
Matching service for Resume Matcher.

Orchestrates: embed → index → retrieve → match → score → report.
"""

from __future__ import annotations

import logging

from resume_matcher.config import AppConfig
from resume_matcher.domain.models import CandidateScore, MatchEvidence, ProcessedDocument
from resume_matcher.embeddings.bge_embedder import BgeEmbedder
from resume_matcher.matching.evidence_matcher import EvidenceMatcher
from resume_matcher.matching.mandatory_handler import MandatoryHandler
from resume_matcher.scoring.component_scorers import (
    EducationScorer,
    SemanticMatchScorer,
    SkillScorer,
    WorkExperienceScorer,
)
from resume_matcher.scoring.final_scorer import FinalScorer

logger = logging.getLogger(__name__)


class MatchingService:
    """
    Orchestrates JD–resume matching and scoring.

    Takes processed documents (from IngestionService) and produces
    a CandidateScore with full evidence trail.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._embedder = BgeEmbedder(config.embedding)
        self._evidence_matcher = EvidenceMatcher(
            self._embedder, config.retrieval, config.scoring.thresholds
        )
        self._mandatory_handler = MandatoryHandler()
        self._skill_scorer = SkillScorer()
        self._exp_scorer = WorkExperienceScorer()
        self._edu_scorer = EducationScorer()
        self._semantic_scorer = SemanticMatchScorer()
        self._final_scorer = FinalScorer(config.scoring.weights)

    def match_and_score(
        self,
        jd_doc: ProcessedDocument,
        resume_doc: ProcessedDocument,
    ) -> tuple[CandidateScore, list[MatchEvidence]]:
        """
        Match a resume against a JD and produce scores.

        Args:
            jd_doc: Processed job description.
            resume_doc: Processed resume.

        Returns:
            Tuple of (CandidateScore, list of MatchEvidence).
        """
        logger.info(
            "Matching: %s vs %s",
            jd_doc.meta.sanitized_filename,
            resume_doc.meta.sanitized_filename,
        )

        jd_blocks = jd_doc.blocks
        resume_blocks = resume_doc.blocks

        # 1. Detect mandatory requirements
        mandatory_blocks = self._mandatory_handler.detect_mandatory_blocks(jd_blocks)

        # 2. Match evidence
        evidence = self._evidence_matcher.match(jd_blocks, resume_blocks)

        # 3. Evaluate mandatory requirements
        mandatory_reqs = self._mandatory_handler.evaluate_mandatory(
            mandatory_blocks, evidence
        )

        # 4. Extract comparison parameters based on JD requirements
        comparison_params = self._evidence_matcher.extract_comparison_parameters(
            jd_blocks, evidence, mandatory_reqs, resume_blocks=resume_blocks
        )

        # 5. Component scores
        weights = self._config.scoring.weights
        skill_score = self._skill_scorer.score(evidence, jd_blocks, weights.skill)
        exp_score = self._exp_scorer.score(evidence, jd_blocks, weights.work_experience)
        edu_score = self._edu_scorer.score(evidence, jd_blocks, weights.education)
        semantic_score = self._semantic_scorer.score(
            evidence, jd_blocks, weights.semantic_match
        )

        components = [skill_score, exp_score, edu_score, semantic_score]

        # 6. Final score
        candidate_score = self._final_scorer.calculate(
            components=components,
            mandatory_requirements=mandatory_reqs,
            candidate_id=resume_doc.meta.document_id,
            jd_id=jd_doc.meta.document_id,
            comparison_parameters=comparison_params,
        )

        return candidate_score, evidence
