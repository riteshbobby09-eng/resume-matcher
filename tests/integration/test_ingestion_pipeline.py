"""
Integration tests for Document Ingestion Pipeline.

Verifies end-to-end ingestion flow:
File -> Validation -> Extraction -> Cleaning -> Segmentation -> Section Detection -> Quality Check.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from resume_matcher.config import AppConfig
from resume_matcher.domain.enums import DocumentType, SectionType
from resume_matcher.services.ingestion_service import IngestionService


@pytest.mark.integration
class TestIngestionPipelineIntegration:
    """Integration test suite for IngestionService with PDF and DOCX files."""

    def test_ingest_pdf_sample(self, test_config: AppConfig, sample_pdf: Path):
        """Test full PDF ingestion pipeline from file to ProcessedDocument."""
        service = IngestionService(test_config)
        doc = service.ingest(sample_pdf, DocumentType.RESUME)

        assert doc is not None
        assert doc.meta.document_id is not None
        assert doc.meta.sanitized_filename == "test_resume.pdf"
        assert len(doc.blocks) > 0

        # Verify blocks contain valid text and source metadata
        for block in doc.blocks:
            assert block.text.strip() != ""
            assert block.document_id == doc.meta.document_id
            assert len(block.source_line_numbers) > 0
            assert isinstance(block.section, SectionType)

    def test_ingest_docx_sample(self, test_config: AppConfig, sample_docx: Path):
        """Test full DOCX ingestion pipeline from file to ProcessedDocument."""
        service = IngestionService(test_config)
        doc = service.ingest(sample_docx, DocumentType.RESUME)

        assert doc is not None
        assert doc.meta.document_id is not None
        assert len(doc.blocks) > 0
        assert any(b.section == SectionType.SKILLS for b in doc.blocks)

    def test_ingest_jd_pdf(self, test_config: AppConfig, sample_jd_pdf: Path):
        """Test JD ingestion and section detection for job requirements."""
        service = IngestionService(test_config)
        doc = service.ingest(sample_jd_pdf, DocumentType.JD)

        assert doc is not None
        assert doc.meta.document_type == DocumentType.JD
        assert len(doc.blocks) > 0

        # Verify section detection captured skills or requirements
        sections = {b.section for b in doc.blocks}
        assert SectionType.SKILLS in sections or SectionType.REQUIREMENTS in sections
