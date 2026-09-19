"""
Edge case tests: corrupted and invalid documents.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from resume_matcher.config import AppConfig
from resume_matcher.domain.enums import DocumentType
from resume_matcher.exceptions import CorruptedDocumentError, FileValidationError
from resume_matcher.services.ingestion_service import IngestionService


@pytest.mark.edge_case
class TestCorruptedDocs:
    """Test behavior when encountering corrupted or invalid files."""

    def test_corrupted_pdf(self, test_config: AppConfig, tmp_path: Path):
        """Corrupted PDF content must raise CorruptedDocumentError."""
        corrupted = tmp_path / "corrupt.pdf"
        # Magic bytes '%PDF' passes file validation, but invalid internals fail extraction
        corrupted.write_bytes(b"%PDF-1.4\nInvalid binary garbage\x00\xff\xfe\x00" * 20)

        service = IngestionService(test_config)
        with pytest.raises(CorruptedDocumentError):
            service.ingest(corrupted, DocumentType.RESUME)

    def test_corrupted_docx(self, test_config: AppConfig, tmp_path: Path):
        """Invalid DOCX zip archive must raise CorruptedDocumentError."""
        corrupted = tmp_path / "corrupt.docx"
        # Magic bytes 'PK\x03\x04' passes file validation, but invalid zip internals fail extraction
        corrupted.write_bytes(b"PK\x03\x04not a real zip archive file" * 10)

        service = IngestionService(test_config)
        with pytest.raises(CorruptedDocumentError):
            service.ingest(corrupted, DocumentType.RESUME)

    def test_nonexistent_file(self, test_config: AppConfig, tmp_path: Path):
        """Non-existent file path must raise FileValidationError."""
        missing = tmp_path / "does_not_exist.pdf"

        service = IngestionService(test_config)
        with pytest.raises(FileValidationError, match="File not found"):
            service.ingest(missing, DocumentType.RESUME)

    def test_unsupported_format(self, test_config: AppConfig, tmp_path: Path):
        """Unsupported file extension must raise ValueError from validation."""
        txt_file = tmp_path / "resume.txt"
        txt_file.write_text("Hello world")

        service = IngestionService(test_config)
        with pytest.raises(ValueError, match="File validation failed"):
            service.ingest(txt_file, DocumentType.RESUME)
