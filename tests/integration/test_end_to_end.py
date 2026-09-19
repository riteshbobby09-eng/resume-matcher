"""
End-to-end pipeline integration tests.

Tests:
PipelineService.process_single
PipelineService.process_batch
JSON report persistence, structure, and schema compliance.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from resume_matcher.config import AppConfig
from resume_matcher.services.pipeline_service import PipelineService


@pytest.mark.integration
class TestEndToEndPipeline:
    """Complete end-to-end integration tests."""

    def test_single_pipeline_execution(
        self,
        test_config: AppConfig,
        sample_jd_pdf: Path,
        sample_pdf: Path,
        tmp_path: Path,
    ):
        """Test complete single resume-JD matching and report generation."""
        pipeline = PipelineService(test_config)
        report_path = pipeline.process_single(
            jd_path=sample_jd_pdf,
            resume_path=sample_pdf,
            output_dir=tmp_path,
        )

        assert report_path.exists()
        assert report_path.suffix == ".json"

        # Verify JSON report structure
        with open(report_path) as f:
            data = json.load(f)

        assert "_schema_version" in data
        assert "_report_id" in data
        assert "processing_details" in data
        assert "candidate_score" in data
        assert "recruiter_summary" in data
        assert "strongest_matches" in data

        cand_score = data["candidate_score"]
        assert 0.0 <= cand_score["final_score"] <= 100.0
        assert len(cand_score["components"]) == 4

    def test_batch_pipeline_execution(
        self,
        test_config: AppConfig,
        sample_jd_pdf: Path,
        sample_pdf: Path,
        sample_docx: Path,
        tmp_path: Path,
    ):
        """Test batch processing with multiple resumes against one JD."""
        pipeline = PipelineService(test_config)
        resumes = [sample_pdf, sample_docx]
        reports = pipeline.process_batch(
            jd_path=sample_jd_pdf,
            resume_paths=resumes,
            output_dir=tmp_path,
        )

        assert len(reports) == 2
        for r_path in reports:
            assert r_path.exists()
            with open(r_path) as f:
                data = json.load(f)
            assert "candidate_score" in data
            assert 0.0 <= data["candidate_score"]["final_score"] <= 100.0
