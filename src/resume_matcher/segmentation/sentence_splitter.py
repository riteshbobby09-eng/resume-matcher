"""
Rule-based sentence splitter for Resume Matcher.

Splits text into sentences using regex rules — NO spaCy or ML models.

Key design:
- Split on sentence-ending punctuation (. ! ?) followed by whitespace + uppercase
- Protect abbreviations, technical terms, decimal numbers, URLs, and emails
- Handle bullet points and list items as separate sentences
- Preserve source line number references

Protected patterns (will NOT trigger a split):
- Abbreviations: Mr., Mrs., Dr., Sr., Jr., Inc., Ltd., Corp., etc.
- Technical terms: 3.5 years, v2.1, GPA 3.8
- Decimal numbers: 3.5, 2.0, 0.75
- URLs and email addresses
- Ellipsis: ...
- e.g., i.e., vs., etc.
"""

from __future__ import annotations

import logging
import re

from resume_matcher.domain.models import IndexedLine, Sentence

logger = logging.getLogger(__name__)

# Abbreviations that end with a period but don't end sentences
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "sr", "jr", "prof", "dept",
    "inc", "ltd", "corp", "assoc", "mgr", "engr", "pvt",
    "st", "ave", "blvd", "rd",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    "vol", "rev", "gen", "gov", "sgt", "cpl", "capt", "lt", "col",
    "approx", "est", "min", "max", "avg",
    "no", "nos", "ref", "tel", "fax", "ext",
    "fig", "eq", "ch", "sec", "pt",
}

# Pattern that matches "e.g.", "i.e.", "vs.", "etc."
_SPECIAL_ABBREV_PATTERN = re.compile(
    r"\b(?:e\.g|i\.e|vs|etc|approx|est|al)\.\s*",
    re.IGNORECASE,
)

# Pattern for decimal numbers: 3.5, 2.0, 0.75
_DECIMAL_PATTERN = re.compile(r"\d+\.\d+")

# Pattern for URLs
_URL_PATTERN = re.compile(r"https?://[^\s]+")

# Pattern for email addresses
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Pattern for tech names with dots
_TECH_DOT_PATTERN = re.compile(
    r"\b(?:node|vue|react|next|express|d3|three|angular|jquery)\.js\b",
    re.IGNORECASE,
)

# Pattern for .NET
_DOTNET_PATTERN = re.compile(r"\.NET\b")

# Ellipsis
_ELLIPSIS_PATTERN = re.compile(r"\.{2,}")


class SentenceSplitter:
    """
    Rule-based sentence splitter.

    Splits cleaned, normalized text lines into individual sentences
    while preserving technical expressions and abbreviations.
    """

    def split_lines(
        self,
        lines: list[IndexedLine],
        document_id: str = "",
    ) -> list[Sentence]:
        """
        Split a list of IndexedLines into Sentences.

        Each line may produce one or more sentences.
        Bullet-point lines are treated as single sentences.

        Args:
            lines: Cleaned, non-empty IndexedLines.
            document_id: Document ID for traceability.

        Returns:
            List of Sentence objects with source line references.
        """
        sentences: list[Sentence] = []

        # Accumulate multi-line text (consecutive non-empty lines
        # that form a paragraph get joined before splitting)
        buffer_text = ""
        buffer_lines: list[int] = []

        for line in lines:
            if line.is_empty or not line.text.strip():
                # Flush buffer on blank lines (paragraph break)
                if buffer_text:
                    new_sents = self._split_text(
                        buffer_text, buffer_lines, document_id
                    )
                    sentences.extend(new_sents)
                    buffer_text = ""
                    buffer_lines = []
                continue

            text = line.text.strip()

            # Bullet points / list items → standalone sentence
            if self._is_list_item(text):
                # Flush any buffered paragraph first
                if buffer_text:
                    new_sents = self._split_text(
                        buffer_text, buffer_lines, document_id
                    )
                    sentences.extend(new_sents)
                    buffer_text = ""
                    buffer_lines = []

                sentences.append(
                    Sentence(
                        text=text,
                        source_line_numbers=[line.global_line_number],
                        document_id=document_id,
                    )
                )
            elif self._is_section_heading(text):
                # Section headings flush the buffer and become standalone
                if buffer_text:
                    new_sents = self._split_text(
                        buffer_text, buffer_lines, document_id
                    )
                    sentences.extend(new_sents)
                    buffer_text = ""
                    buffer_lines = []

                sentences.append(
                    Sentence(
                        text=text,
                        source_line_numbers=[line.global_line_number],
                        document_id=document_id,
                    )
                )
            else:
                # Add to paragraph buffer
                if buffer_text:
                    buffer_text += " " + text
                else:
                    buffer_text = text
                buffer_lines.append(line.global_line_number)

        # Flush remaining buffer
        if buffer_text:
            new_sents = self._split_text(buffer_text, buffer_lines, document_id)
            sentences.extend(new_sents)

        logger.debug("Split %d lines into %d sentences", len(lines), len(sentences))
        return sentences

    def _split_text(
        self,
        text: str,
        source_lines: list[int],
        document_id: str,
    ) -> list[Sentence]:
        """
        Split a text block into sentences using rule-based patterns.

        Uses placeholder protection for abbreviations and technical terms.
        """
        if not text.strip():
            return []

        # Protect patterns from incorrect splitting
        protected_text, placeholders = self._protect_patterns(text)

        # Split on sentence boundaries:
        # Period/exclamation/question followed by one+ whitespace and an uppercase letter
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected_text)

        # Restore placeholders and create sentences
        sentences: list[Sentence] = []
        for part in parts:
            restored = self._restore_patterns(part, placeholders)
            restored = restored.strip()
            if restored:
                sentences.append(
                    Sentence(
                        text=restored,
                        source_line_numbers=list(source_lines),
                        document_id=document_id,
                    )
                )

        return sentences if sentences else [
            Sentence(
                text=text.strip(),
                source_line_numbers=list(source_lines),
                document_id=document_id,
            )
        ]

    def _protect_patterns(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Replace patterns that shouldn't trigger sentence splits with placeholders.

        Returns (modified_text, {placeholder: original}) mapping.
        """
        placeholders: dict[str, str] = {}
        counter = 0

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            key = f"__SENT_PROT_{counter:04d}__"
            placeholders[key] = match.group(0)
            counter += 1
            return key

        # Order matters: more specific patterns first
        text = _URL_PATTERN.sub(_replace, text)
        text = _EMAIL_PATTERN.sub(_replace, text)
        text = _TECH_DOT_PATTERN.sub(_replace, text)
        text = _DOTNET_PATTERN.sub(_replace, text)
        text = _ELLIPSIS_PATTERN.sub(_replace, text)
        text = _SPECIAL_ABBREV_PATTERN.sub(_replace, text)
        text = _DECIMAL_PATTERN.sub(_replace, text)

        # Protect known abbreviations: "Dr. " → placeholder
        for abbrev in _ABBREVIATIONS:
            pattern = re.compile(
                rf"\b{re.escape(abbrev)}\.\s",
                re.IGNORECASE,
            )
            text = pattern.sub(_replace, text)

        return text, placeholders

    def _restore_patterns(self, text: str, placeholders: dict[str, str]) -> str:
        """Restore all placeholders with original text."""
        for key, original in placeholders.items():
            text = text.replace(key, original)
        return text

    def _is_list_item(self, text: str) -> bool:
        """Check if a line is a bullet point or numbered list item."""
        return bool(re.match(
            r"^\s*(?:[•●○◦▪▸►➤➢★✦✧→⮞⬥\-\*]|\d+[.)]\s|[a-zA-Z][.)]\s|[ivxIVX]+[.)]\s)\s*\S",
            text,
        ))

    def _is_section_heading(self, text: str) -> bool:
        """
        Check if a line is likely a section heading.

        Headings break the paragraph buffer to ensure proper block formation.
        Detects:
        - ALL CAPS short lines (e.g. "WORK EXPERIENCE", "SKILLS")
        - Colon-terminated short lines (e.g. "Skills:", "Education:")
        - Known section title patterns
        """
        stripped = text.strip()
        words = stripped.split()
        word_count = len(words)

        # ALL CAPS with 1-6 words
        if stripped.isupper() and 1 <= word_count <= 6 and len(stripped) <= 60:
            return True

        # Colon-terminated with 1-5 words
        if stripped.endswith(":") and 1 <= word_count <= 5 and len(stripped) <= 50:
            return True

        # Title-case short lines that look like headings
        if (
            word_count <= 4
            and len(stripped) <= 40
            and stripped[0].isupper()
            and not stripped.endswith(".")
            and not any(c.isdigit() for c in stripped)
            and stripped.istitle()
        ):
            return True

        return False
