"""
Text cleaner for Resume Matcher.

Handles low-level text hygiene:
- Unicode normalization (NFKC)
- Control character removal (preserving newlines)
- Encoding fix attempts (mojibake patterns)
- Whitespace normalization
- Header/footer artifact removal

Does NOT change the semantic content of text.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from resume_matcher.domain.models import IndexedLine

logger = logging.getLogger(__name__)

# Patterns for header/footer artifacts
_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),  # - 3 -
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),  # 3/10
    re.compile(r"^\s*\[\s*\d+\s*\]\s*$"),  # [3]
]

# Common mojibake patterns and their fixes
_MOJIBAKE_MAP = {
    "\u00e2\u0080\u0099": "'",
    "\u00e2\u0080\u0098": "'",
    "\u00e2\u0080\u009c": '"',
    "\u00e2\u0080\u009d": '"',
    "\u00e2\u0080\u0094": "\u2014",  # em-dash
    "\u00e2\u0080\u0093": "\u2013",  # en-dash
    "\u00e2\u0080\u00a6": "\u2026",  # ellipsis
    "\u00c3\u00a9": "\u00e9",  # é
    "\u00c3\u00a8": "\u00e8",  # è
    "\u00c3\u00bc": "\u00fc",  # ü
    "\u00c3\u00b6": "\u00f6",  # ö
    "\u00c3\u00a4": "\u00e4",  # ä
    "\u00c3\u00b1": "\u00f1",  # ñ
}

# Control characters to remove (except newline, tab, carriage return)
_CONTROL_CHAR_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


class TextCleaner:
    """
    Cleans raw extracted text without altering semantic content.

    Processing order:
    1. Fix mojibake encoding artifacts
    2. Unicode NFKC normalization
    3. Remove control characters
    4. Normalize whitespace
    5. Remove header/footer artifacts
    """

    def clean_lines(self, lines: list[IndexedLine]) -> list[IndexedLine]:
        """
        Clean a list of IndexedLines in place.

        Args:
            lines: Lines to clean. The .text attribute is modified.

        Returns:
            The same list with cleaned text (preserving line numbers).
        """
        cleaned_count = 0
        for line in lines:
            original = line.text
            line.text = self.clean_text(line.text)
            if line.text != original:
                cleaned_count += 1
            line.is_empty = (line.text.strip() == "")

        if cleaned_count:
            logger.debug("Cleaned %d of %d lines", cleaned_count, len(lines))

        return lines

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        # 1. Fix mojibake
        text = self._fix_mojibake(text)

        # 2. Unicode normalization (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # 3. Remove control characters (keep \n, \t, \r)
        text = _CONTROL_CHAR_PATTERN.sub("", text)

        # 4. Normalize whitespace
        text = self._normalize_whitespace(text)

        return text

    def remove_artifacts(self, lines: list[IndexedLine]) -> list[IndexedLine]:
        """
        Mark header/footer artifact lines as empty.

        Does NOT remove them — preserving line numbering for traceability.
        Instead, marks them as empty so downstream processing skips them.
        """
        for line in lines:
            if self._is_page_artifact(line.text):
                line.text = ""
                line.is_empty = True
        return lines

    def _fix_mojibake(self, text: str) -> str:
        """Replace common mojibake sequences with correct characters."""
        for bad, good in _MOJIBAKE_MAP.items():
            text = text.replace(bad, good)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/tabs to single spaces, strip edges."""
        # Replace tabs with spaces
        text = text.replace("\t", " ")
        # Collapse multiple spaces to one
        text = re.sub(r" {2,}", " ", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _is_page_artifact(self, text: str) -> bool:
        """Check if a line looks like a page number or header/footer artifact."""
        stripped = text.strip()
        if not stripped:
            return False
        # Very short lines that are just numbers
        if stripped.isdigit() and len(stripped) <= 4:
            return True
        return any(pat.match(stripped) for pat in _PAGE_NUMBER_PATTERNS)
