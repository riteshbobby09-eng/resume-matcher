"""Unit tests for file validator."""

import os
import tempfile
from pathlib import Path

import pytest

from resume_matcher.config import ProcessingConfig
from resume_matcher.domain.enums import FileFormat, ValidationStatus
from resume_matcher.exceptions import FileValidationError
from resume_matcher.ingestion.validator import FileValidator


@pytest.fixture
def validator():
    config = ProcessingConfig(max_file_size_mb=5)
    return FileValidator(config)


class TestFileValidator:
    """Tests for FileValidator."""

    def test_valid_pdf(self, validator, sample_pdf):
        result = validator.validate(sample_pdf)
        assert result.is_valid
        assert result.file_format == FileFormat.PDF
        assert result.file_size_bytes > 0

    def test_valid_docx(self, validator, sample_docx):
        result = validator.validate(sample_docx)
        assert result.is_valid
        assert result.file_format == FileFormat.DOCX

    def test_nonexistent_file(self, validator):
        with pytest.raises(FileValidationError, match="File not found"):
            validator.validate("/nonexistent/file.pdf")

    def test_unsupported_extension(self, validator, tmp_dir):
        txt_file = tmp_dir / "test.txt"
        txt_file.write_text("hello")
        result = validator.validate(txt_file)
        assert result.status == ValidationStatus.ERROR
        assert "Unsupported extension" in result.messages[0]

    def test_empty_file(self, validator, tmp_dir):
        empty = tmp_dir / "empty.pdf"
        empty.touch()
        result = validator.validate(empty)
        assert result.status == ValidationStatus.ERROR
        assert "empty" in result.messages[0].lower()

    def test_file_too_large(self, tmp_dir):
        config = ProcessingConfig(max_file_size_mb=0)  # 0 MB limit
        validator = FileValidator(config)
        small_file = tmp_dir / "test.pdf"
        small_file.write_bytes(b"%PDF" + b"x" * 100)
        result = validator.validate(small_file)
        assert result.status == ValidationStatus.ERROR
        assert "too large" in result.messages[0].lower()

    def test_filename_sanitization(self, validator):
        dangerous_names = [
            "../../../etc/passwd.pdf",
            "file\x00name.pdf",
            "path/to/file.pdf",
            "file\\name.pdf",
        ]
        for name in dangerous_names:
            sanitized = validator._sanitize_filename(name)
            assert ".." not in sanitized
            assert "/" not in sanitized
            assert "\\" not in sanitized
            assert "\x00" not in sanitized

    def test_magic_bytes_mismatch(self, validator, tmp_dir):
        fake_pdf = tmp_dir / "fake.pdf"
        fake_pdf.write_bytes(b"NOT_A_PDF_FILE_CONTENTS")
        result = validator.validate(fake_pdf)
        assert result.status == ValidationStatus.WARNING
        assert "magic bytes" in result.messages[0].lower()
