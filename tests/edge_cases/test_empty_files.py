"""
Edge case tests: empty and whitespace-only files.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from resume_matcher.config import AppConfig
from resume_matcher.domain.enums import DocumentType, ValidationStatus
from resume_matcher.services.ingestion_service import IngestionService


@pytest.mark.edge_case
class TestEmptyFiles:
    """Test behavior when encountering empty or whitespace-only files."""

    def test_zero_byte_pdf(self, test_config: AppConfig, tmp_path: Path):
        """Zero-byte file must fail validation with ValueError."""
        empty_file = tmp_path / "zero_byte.pdf"
        empty_file.write_bytes(b"")

        service = IngestionService(test_config)
        with pytest.raises(ValueError, match="File validation failed"):
            service.ingest(empty_file, DocumentType.RESUME)

    def test_zero_byte_docx(self, test_config: AppConfig, tmp_path: Path):
        """Zero-byte DOCX must fail validation with ValueError."""
        empty_file = tmp_path / "zero_byte.docx"
        empty_file.write_bytes(b"")

        service = IngestionService(test_config)
        with pytest.raises(ValueError, match="File validation failed"):
            service.ingest(empty_file, DocumentType.RESUME)

    def test_empty_pdf_no_text(self, test_config: AppConfig, empty_pdf: Path):
        """PDF with valid structure but zero text ingests with 0 blocks and flags quality warnings."""
        service = IngestionService(test_config)
        doc = service.ingest(empty_pdf, DocumentType.RESUME)
        assert len(doc.blocks) == 0
        assert len(doc.sentences) == 0
        assert doc.quality_report is not None
        assert doc.quality_report.overall_status != ValidationStatus.VALID or len(doc.quality_report.messages) > 0

    def test_empty_docx_no_text(self, test_config: AppConfig, empty_docx: Path):
        """DOCX with valid structure but zero text ingests with 0 blocks and flags quality warnings."""
        service = IngestionService(test_config)
        doc = service.ingest(empty_docx, DocumentType.RESUME)
        assert len(doc.blocks) == 0
        assert len(doc.sentences) == 0
        assert doc.quality_report is not None
        assert doc.quality_report.overall_status != ValidationStatus.VALID or len(doc.quality_report.messages) > 0
