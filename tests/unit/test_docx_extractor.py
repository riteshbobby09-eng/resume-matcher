"""Unit tests for DOCX extractor."""

import pytest

from resume_matcher.domain.enums import FileFormat
from resume_matcher.domain.models import DocumentMeta
from resume_matcher.exceptions import CorruptedDocumentError
from resume_matcher.ingestion.docx_extractor import DocxExtractor


@pytest.fixture
def extractor():
    return DocxExtractor()


class TestDocxExtractor:
    def test_extract_basic(self, extractor, sample_docx):
        meta = DocumentMeta(source_filename="test.docx", file_format=FileFormat.DOCX)
        lines = extractor.extract(sample_docx, meta)
        assert len(lines) > 0
        assert all(line.global_line_number > 0 for line in lines)

    def test_extract_with_table(self, extractor, docx_with_table):
        meta = DocumentMeta(source_filename="table.docx", file_format=FileFormat.DOCX)
        lines = extractor.extract(docx_with_table, meta)
        assert len(lines) > 0
        # Table rows should be extracted with | separator
        table_lines = [l for l in lines if "|" in l.text]
        assert len(table_lines) > 0

    def test_extract_empty_docx(self, extractor, empty_docx):
        meta = DocumentMeta(source_filename="empty.docx", file_format=FileFormat.DOCX)
        lines = extractor.extract(empty_docx, meta)
        assert isinstance(lines, list)

    def test_extract_corrupted(self, extractor, tmp_dir):
        bad_file = tmp_dir / "bad.docx"
        bad_file.write_bytes(b"NOT A DOCX")
        meta = DocumentMeta(source_filename="bad.docx", file_format=FileFormat.DOCX)
        with pytest.raises((CorruptedDocumentError, Exception)):
            extractor.extract(bad_file, meta)

    def test_line_numbers_sequential(self, extractor, sample_docx):
        meta = DocumentMeta(source_filename="test.docx", file_format=FileFormat.DOCX)
        lines = extractor.extract(sample_docx, meta)
        for i, line in enumerate(lines):
            assert line.global_line_number == i + 1
