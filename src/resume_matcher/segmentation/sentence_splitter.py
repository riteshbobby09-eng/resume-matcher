"""
Sentence splitter for Resume Matcher.

Splits text into sentences using SpaCy for linguistically-aware segmentation.
Falls back to rule-based splitting if SpaCy is unavailable.

Key features:
- SpaCy `en_core_web_sm` for sentence boundary detection
- Bullet points and list items → standalone sentences
- Section headings → standalone sentences
- Protected patterns: abbreviations, technical terms, decimal numbers, URLs, emails
"""

from __future__ import annotations

import logging
import re

from resume_matcher.domain.models import IndexedLine, Sentence

logger = logging.getLogger(__name__)

# ── SpaCy singleton ──
_nlp_cache = {}


def _get_spacy_nlp():
    """Load or retrieve cached SpaCy model (singleton)."""
    if "nlp" in _nlp_cache:
        return _nlp_cache["nlp"]
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        # Disable components we don't need for sentence splitting
        nlp.select_pipes(enable=["tok2vec", "parser", "senter"])
        _nlp_cache["nlp"] = nlp
        logger.info("SpaCy model loaded: en_core_web_sm")
        return nlp
    except Exception as e:
        logger.warning("SpaCy not available, using rule-based fallback: %s", e)
        _nlp_cache["nlp"] = None
        return None


# ── Rule-based fallback patterns ──
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

_SPECIAL_ABBREV_PATTERN = re.compile(
    r"\b(?:e\.g|i\.e|vs|etc|approx|est|al)\.\s*",
    re.IGNORECASE,
)
_DECIMAL_PATTERN = re.compile(r"\d+\.\d+")
_URL_PATTERN = re.compile(r"https?://[^\s]+")
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_TECH_DOT_PATTERN = re.compile(
    r"\b(?:node|vue|react|next|express|d3|three|angular|jquery)\.js\b",
    re.IGNORECASE,
)
_DOTNET_PATTERN = re.compile(r"\.NET\b")
_ELLIPSIS_PATTERN = re.compile(r"\.{2,}")


class SentenceSplitter:
    """
    Sentence splitter with SpaCy and rule-based fallback.

    Uses SpaCy's en_core_web_sm for linguistically-aware sentence
    boundary detection, with fallback to regex rules if SpaCy is unavailable.
    """

    def __init__(self, use_spacy: bool = True) -> None:
        self._use_spacy = use_spacy
        if use_spacy:
            self._nlp = _get_spacy_nlp()
        else:
            self._nlp = None

    def split_text(
        self,
        text: str,
        document_id: str = "",
    ) -> list[Sentence]:
        """
        Split a raw text string into Sentences.

        Convenience method that wraps split_lines.
        """
        raw_lines = text.splitlines() or [text]
        indexed = [
            IndexedLine(global_line_number=i + 1, text=l, is_empty=not l.strip())
            for i, l in enumerate(raw_lines)
        ]
        return self.split_lines(indexed, document_id)

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
        Split a text block into sentences using SpaCy or rule-based fallback.
        """
        if not text.strip():
            return []

        # Use SpaCy if available
        if self._nlp is not None:
            return self._split_with_spacy(text, source_lines, document_id)

        # Fallback: rule-based splitting
        return self._split_with_rules(text, source_lines, document_id)

    def _split_with_spacy(
        self,
        text: str,
        source_lines: list[int],
        document_id: str,
    ) -> list[Sentence]:
        """Split using SpaCy's sentence boundary detection."""
        doc = self._nlp(text)
        sentences: list[Sentence] = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                sentences.append(
                    Sentence(
                        text=sent_text,
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

    def _split_with_rules(
        self,
        text: str,
        source_lines: list[int],
        document_id: str,
    ) -> list[Sentence]:
        """
        Fallback rule-based sentence splitting.

        Uses placeholder protection for abbreviations and technical terms.
        """
        # Protect patterns from incorrect splitting
        protected_text, placeholders = self._protect_patterns(text)

        # Split on sentence boundaries
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
        """Replace patterns that shouldn't trigger sentence splits with placeholders."""
        placeholders: dict[str, str] = {}
        counter = 0

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            key = f"__SENT_PROT_{counter:04d}__"
            placeholders[key] = match.group(0)
            counter += 1
            return key

        text = _URL_PATTERN.sub(_replace, text)
        text = _EMAIL_PATTERN.sub(_replace, text)
        text = _TECH_DOT_PATTERN.sub(_replace, text)
        text = _DOTNET_PATTERN.sub(_replace, text)
        text = _ELLIPSIS_PATTERN.sub(_replace, text)
        text = _SPECIAL_ABBREV_PATTERN.sub(_replace, text)
        text = _DECIMAL_PATTERN.sub(_replace, text)

        for abbrev in _ABBREVIATIONS:
            pattern = re.compile(rf"\b{re.escape(abbrev)}\.\s", re.IGNORECASE)
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
