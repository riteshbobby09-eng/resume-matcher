"""
Text normalizer for Resume Matcher.

Performs controlled normalization that preserves important technical expressions.

Key preserved patterns:
- "3.5 years", "5+ years", "10+ years"
- "C++", "C#", ".NET", "node.js", "Vue.js"
- Version numbers: "v2.1", "Python 3.10"
- GPA values: "GPA 3.8/4.0"
- Email addresses and phone numbers
- Decimal numbers: "3.5", "2.0"
- Abbreviations with periods: "Sr.", "Jr.", "Inc."

Uses a placeholder-based approach: protect patterns → normalize → restore.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from resume_matcher.domain.models import IndexedLine

logger = logging.getLogger(__name__)


@dataclass
class _ProtectedToken:
    """A text pattern that should be preserved during normalization."""

    placeholder: str
    original: str


class TextNormalizer:
    """
    Normalizes text while preserving technical expressions.

    Uses placeholder tokens to protect important patterns from
    being mangled by normalization steps.
    """

    # Patterns to protect — ORDER MATTERS (more specific first)
    # Each tuple: (compiled regex, group index to protect)
    _PROTECTION_PATTERNS: list[tuple[re.Pattern[str], int]] = [
        # Email addresses
        (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), 0),
        # URLs
        (re.compile(r"https?://[^\s]+"), 0),
        # Technology names with special chars
        (re.compile(r"\bC\+\+\b"), 0),
        (re.compile(r"\bC#\b"), 0),
        (re.compile(r"\.NET\b"), 0),
        (re.compile(r"\b[Nn]ode\.js\b"), 0),
        (re.compile(r"\b[Vv]ue\.js\b"), 0),
        (re.compile(r"\b[Rr]eact\.js\b"), 0),
        (re.compile(r"\b[Nn]ext\.js\b"), 0),
        (re.compile(r"\b[Ee]xpress\.js\b"), 0),
        (re.compile(r"\b[Dd]3\.js\b"), 0),
        (re.compile(r"\b[Tt]hree\.js\b"), 0),
        (re.compile(r"\bAngular\.js\b"), 0),
        # GPA patterns: "GPA 3.8", "GPA 3.8/4.0", "CGPA: 8.5"
        (re.compile(r"(?:C?GPA|gpa)[\s:]*\d+\.?\d*(?:\s*/\s*\d+\.?\d*)?"), 0),
        # Version numbers: "v2.1", "Python 3.10", "Java 17"
        (re.compile(r"\bv\d+(?:\.\d+)+\b"), 0),
        (re.compile(r"\b(?:Python|Java|PHP|Ruby|Go|Rust|Swift|Kotlin)\s+\d+(?:\.\d+)*\b"), 0),
        # Experience patterns: "3.5 years", "5+ years", "10+ yrs"
        (re.compile(r"\d+\.?\d*\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE), 0),
        # Decimal numbers in context: "3.5", "2.0", "0.75"
        (re.compile(r"\b\d+\.\d+\b"), 0),
        # Phone numbers (various formats)
        (re.compile(r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}"), 0),
        # Abbreviations with periods
        (re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Sr|Jr|Inc|Ltd|Corp|Prof|Dept|Assoc|Mgr|Engr)\.", re.IGNORECASE), 0),
        # Common abbreviations
        (re.compile(r"\b(?:e\.g|i\.e|vs|etc|approx|est)\.", re.IGNORECASE), 0),
    ]

    # Bullet point / list marker normalization
    _BULLET_PATTERN = re.compile(
        r"^\s*(?:[•●○◦▪▸►➤➢★✦✧→⮞⬥\-\*]|\d+[.)]\s|[a-zA-Z][.)]\s|[ivxIVX]+[.)]\s)\s*"
    )

    def normalize_lines(self, lines: list[IndexedLine]) -> list[IndexedLine]:
        """
        Normalize text in a list of IndexedLines.

        Args:
            lines: Lines to normalize.

        Returns:
            Same list with normalized text.
        """
        for line in lines:
            if not line.is_empty:
                line.text = self.normalize_text(line.text)
        return lines

    def normalize_text(self, text: str) -> str:
        """
        Normalize a single text string while preserving technical expressions.

        Steps:
        1. Protect technical patterns with placeholders
        2. Normalize bullets/list markers to "• "
        3. Normalize dashes and special punctuation
        4. Restore protected patterns
        """
        if not text or not text.strip():
            return text

        # 1. Protect important patterns
        tokens: list[_ProtectedToken] = []
        protected_text = self._protect_patterns(text, tokens)

        # 2. Normalize bullet points
        protected_text = self._normalize_bullets(protected_text)

        # 3. Normalize dashes and punctuation
        protected_text = self._normalize_punctuation(protected_text)

        # 4. Restore protected patterns
        result = self._restore_patterns(protected_text, tokens)

        return result

    def _protect_patterns(self, text: str, tokens: list[_ProtectedToken]) -> str:
        """Replace protected patterns with unique placeholders."""
        for pattern, group_idx in self._PROTECTION_PATTERNS:
            for match in pattern.finditer(text):
                original = match.group(group_idx)
                placeholder = f"__PROT_{len(tokens):04d}__"
                tokens.append(_ProtectedToken(placeholder=placeholder, original=original))
                text = text.replace(original, placeholder, 1)
        return text

    def _restore_patterns(self, text: str, tokens: list[_ProtectedToken]) -> str:
        """Restore protected patterns from placeholders (reverse order)."""
        for token in reversed(tokens):
            text = text.replace(token.placeholder, token.original)
        return text

    def _normalize_bullets(self, text: str) -> str:
        """Normalize various bullet point characters to a standard '• '."""
        return self._BULLET_PATTERN.sub("• ", text)

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize special dashes and quotation marks."""
        # Smart quotes → straight quotes
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        # Em-dash / en-dash → standard dash with spaces
        text = text.replace("\u2014", " - ").replace("\u2013", " - ")
        # Collapse resulting multiple spaces
        text = re.sub(r" {2,}", " ", text)
        return text
