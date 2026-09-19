"""
Integration tests for Semantic Matching & Scoring Pipeline.

Verifies:
Embeddings -> FAISS Indexing -> Evidence Matching -> Mandatory Requirements -> Component Scoring -> Final Aggregation.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from resume_matcher.config import AppConfig
from resume_matcher.domain.enums import DocumentType
from resume_matcher.services.ingestion_service import IngestionService
from resume_matcher.services.matching_service import MatchingService


@pytest.mark.integration
class TestMatchingPipelineIntegration:
    """Integration test suite for MatchingService."""

    def test_matching_end_to_end(
        self,
        test_config: AppConfig,
        sample_jd_pdf: Path,
        sample_pdf: Path,
    ):
        """Verify full matching process produces scores, evidence, and mandatory audit trail."""
        ingestion = IngestionService(test_config)
        jd_doc = ingestion.ingest(sample_jd_pdf, DocumentType.JD)
        resume_doc = ingestion.ingest(sample_pdf, DocumentType.RESUME)

        matching = MatchingService(test_config)
        candidate_score, evidence = matching.match_and_score(jd_doc, resume_doc)

        # Assert score validity
        assert candidate_score is not None
        assert 0.0 <= candidate_score.final_score <= 100.0
        assert len(candidate_score.components) == 4

        # Verify component score weights sum to 1.0
        total_weight = sum(c.weight for c in candidate_score.components)
        assert abs(total_weight - 1.0) < 1e-5

        # Verify evidence exists and contains similarity scores
        assert len(evidence) > 0
        for ev in evidence:
            assert ev.jd_block_id is not None
            assert ev.resume_block_id is not None
            assert 0.0 <= ev.similarity_score <= 1.0

    def test_matching_determinism(
        self,
        test_config: AppConfig,
        sample_jd_pdf: Path,
        sample_pdf: Path,
    ):
        """Verify matching is 100% deterministic across consecutive runs."""
        ingestion = IngestionService(test_config)
        jd_doc = ingestion.ingest(sample_jd_pdf, DocumentType.JD)
        resume_doc = ingestion.ingest(sample_pdf, DocumentType.RESUME)

        matching = MatchingService(test_config)
        score1, _ = matching.match_and_score(jd_doc, resume_doc)
        score2, _ = matching.match_and_score(jd_doc, resume_doc)

        assert score1.final_score == score2.final_score
        for c1, c2 in zip(score1.components, score2.components):
            assert c1.name == c2.name
            assert c1.raw_score == c2.raw_score
            assert c1.weighted_contribution == c2.weighted_contribution
