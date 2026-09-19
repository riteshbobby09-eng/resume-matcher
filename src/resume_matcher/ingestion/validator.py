"""
File validator for Resume Matcher.

Performs safety and format checks before any processing:
- File existence and readability
- Extension allowlist
- File size limits
- Magic-byte verification (PDF/DOCX signatures)
- Filename sanitization (path traversal prevention)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from resume_matcher.config import ProcessingConfig
from resume_matcher.domain.enums import FileFormat, ValidationStatus
from resume_matcher.exceptions import FileValidationError

logger = logging.getLogger(__name__)

# Magic bytes for supported formats
_MAGIC_BYTES: dict[FileFormat, list[bytes]] = {
    FileFormat.PDF: [b"%PDF"],
    FileFormat.DOCX: [b"PK\x03\x04", b"PK\x05\x06"],  # ZIP (OOXML)
}


@dataclass
class ValidationResult:
    """Outcome of file validation."""

    status: ValidationStatus = ValidationStatus.VALID
    file_path: Path = field(default_factory=lambda: Path("."))
    file_format: FileFormat = FileFormat.PDF
    file_size_bytes: int = 0
    sanitized_filename: str = ""
    messages: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status != ValidationStatus.ERROR


class FileValidator:
    """
    Validates files before ingestion.

    Checks file existence, extension, size, magic bytes, and filename safety.
    Does NOT read or process file contents beyond the first few bytes.
    """

    def __init__(self, config: ProcessingConfig) -> None:
        self._config = config
        self._max_size_bytes = config.max_file_size_mb * 1024 * 1024
        self._allowed_extensions = set(config.allowed_extensions)

    def validate(self, file_path: str | Path) -> ValidationResult:
        """
        Validate a file for processing.

        Args:
            file_path: Path to the file to validate.

        Returns:
            ValidationResult with status, messages, and sanitized filename.

        Raises:
            FileValidationError: If the file is fundamentally invalid
                                 (doesn't exist, can't be read).
        """
        path = Path(file_path).resolve()
        result = ValidationResult(file_path=path)
        messages: list[str] = []

        # 1. Existence and readability
        if not path.exists():
            raise FileValidationError(
                f"File not found: {path.name}",
                details={"path": str(path)},
            )
        if not path.is_file():
            raise FileValidationError(
                f"Not a regular file: {path.name}",
                details={"path": str(path)},
            )
        if not os.access(path, os.R_OK):
            raise FileValidationError(
                f"File is not readable: {path.name}",
                details={"path": str(path)},
            )

        # 2. Filename sanitization
        result.sanitized_filename = self._sanitize_filename(path.name)

        # 3. Extension check
        ext = path.suffix.lower()
        if ext not in self._allowed_extensions:
            result.status = ValidationStatus.ERROR
            messages.append(
                f"Unsupported extension '{ext}'. "
                f"Allowed: {sorted(self._allowed_extensions)}"
            )
            result.messages = messages
            return result

        # Determine format from extension
        result.file_format = FileFormat.PDF if ext == ".pdf" else FileFormat.DOCX

        # 4. File size check
        result.file_size_bytes = path.stat().st_size
        if result.file_size_bytes == 0:
            result.status = ValidationStatus.ERROR
            messages.append("File is empty (0 bytes)")
            result.messages = messages
            return result

        if result.file_size_bytes > self._max_size_bytes:
            result.status = ValidationStatus.ERROR
            size_mb = result.file_size_bytes / (1024 * 1024)
            messages.append(
                f"File too large: {size_mb:.1f}MB exceeds "
                f"limit of {self._config.max_file_size_mb}MB"
            )
            result.messages = messages
            return result

        # 5. Magic-byte verification
        if not self._verify_magic_bytes(path, result.file_format):
            result.status = ValidationStatus.WARNING
            messages.append(
                f"File extension is '{ext}' but magic bytes do not match "
                f"expected {result.file_format.value.upper()} signature. "
                f"File may be corrupted or mislabeled."
            )

        if messages:
            result.messages = messages
            if result.status == ValidationStatus.VALID:
                result.status = ValidationStatus.WARNING

        logger.info(
            "File validated: %s [%s, %d bytes, %s]",
            result.sanitized_filename,
            result.file_format.value,
            result.file_size_bytes,
            result.status.value,
        )
        return result

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename for safe internal use.

        Removes path traversal sequences, null bytes, and special characters.
        Preserves the extension.
        """
        # Remove null bytes
        sanitized = filename.replace("\x00", "")
        # Remove path separators and traversal
        sanitized = sanitized.replace("..", "")
        sanitized = sanitized.replace("/", "_")
        sanitized = sanitized.replace("\\", "_")
        # Keep only safe characters: alphanumeric, underscore, hyphen, dot, space
        sanitized = re.sub(r"[^\w\s.\-]", "_", sanitized)
        # Collapse multiple underscores/spaces
        sanitized = re.sub(r"[_\s]+", "_", sanitized)
        # Strip leading/trailing underscores and dots (except the extension dot)
        name_part = Path(sanitized).stem.strip("_.")
        ext_part = Path(sanitized).suffix
        sanitized = f"{name_part}{ext_part}" if name_part else f"unnamed{ext_part}"
        return sanitized

    def _verify_magic_bytes(self, path: Path, expected_format: FileFormat) -> bool:
        """
        Check if file starts with expected magic bytes.

        Returns True if magic bytes match, False otherwise.
        """
        expected_sigs = _MAGIC_BYTES.get(expected_format, [])
        if not expected_sigs:
            return True  # No signature to check

        max_sig_len = max(len(sig) for sig in expected_sigs)
        try:
            with open(path, "rb") as fh:
                header = fh.read(max_sig_len)
        except OSError:
            return False

        return any(header.startswith(sig) for sig in expected_sigs)
