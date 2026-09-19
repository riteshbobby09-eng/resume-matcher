"""Unit tests for PDF extractor."""

import pytest

from resume_matcher.domain.enums import FileFormat
from resume_matcher.domain.models import DocumentMeta
from resume_matcher.exceptions import CorruptedDocumentError
from resume_matcher.ingestion.pdf_extractor import PdfExtractor


@pytest.fixture
def extractor():
    return PdfExtractor()


class TestPdfExtractor:
    def test_extract_basic(self, extractor, sample_pdf):
        meta = DocumentMeta(source_filename="test.pdf", file_format=FileFormat.PDF)
        lines = extractor.extract(sample_pdf, meta)
        assert len(lines) > 0
        assert all(line.global_line_number > 0 for line in lines)
        assert meta.page_count >= 1

    def test_extract_preserves_line_numbers(self, extractor, sample_pdf):
        meta = DocumentMeta(source_filename="test.pdf", file_format=FileFormat.PDF)
        lines = extractor.extract(sample_pdf, meta)
        numbers = [l.global_line_number for l in lines]
        assert numbers == sorted(numbers)
        assert numbers[0] == 1

    def test_extract_empty_pdf(self, extractor, empty_pdf):
        meta = DocumentMeta(source_filename="empty.pdf", file_format=FileFormat.PDF)
        lines = extractor.extract(empty_pdf, meta)
        # Empty PDF should return 0 or very few lines
        assert isinstance(lines, list)

    def test_extract_corrupted_file(self, extractor, tmp_dir):
        bad_file = tmp_dir / "corrupted.pdf"
        bad_file.write_bytes(b"THIS IS NOT A PDF")
        meta = DocumentMeta(source_filename="corrupted.pdf", file_format=FileFormat.PDF)
        with pytest.raises(CorruptedDocumentError):
            extractor.extract(bad_file, meta)

    def test_lines_have_source_info(self, extractor, sample_pdf):
        meta = DocumentMeta(
            source_filename="test_resume.pdf",
            file_format=FileFormat.PDF,
            document_id="doc_001",
        )
        lines = extractor.extract(sample_pdf, meta)
        for line in lines:
            assert line.source_filename == "test_resume.pdf"
            assert line.document_id == "doc_001"
            assert line.page_number >= 1
