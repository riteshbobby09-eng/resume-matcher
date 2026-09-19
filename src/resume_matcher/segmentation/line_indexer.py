"""
Line indexer for Resume Matcher.

Assigns globally unique line numbers across the entire document and
maintains a mapping from global_line_num → (page, local_line, source_file).

This is the anchor of the traceability chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from resume_matcher.domain.models import IndexedLine

logger = logging.getLogger(__name__)


@dataclass
class LineMap:
    """Mapping from global line number to source location."""

    global_line_number: int
    page_number: int
    local_line_number: int
    source_filename: str
    is_empty: bool


class LineIndexer:
    """
    Manages global line numbering and provides lookup by line number.

    The extractors already assign global_line_numbers; this class
    validates them, builds the lookup index, and filters empty lines
    while preserving the numbering for traceability.
    """

    def __init__(self) -> None:
        self._index: dict[int, LineMap] = {}

    def index_lines(self, lines: list[IndexedLine]) -> list[IndexedLine]:
        """
        Build a line index from extracted lines.

        Validates numbering continuity and builds the internal lookup.

        Args:
            lines: Lines from extractors (already numbered).

        Returns:
            The same lines list (unchanged), after building the index.
        """
        self._index.clear()

        for line in lines:
            if line.global_line_number in self._index:
                logger.warning(
                    "Duplicate line number %d found — overwriting",
                    line.global_line_number,
                )
            self._index[line.global_line_number] = LineMap(
                global_line_number=line.global_line_number,
                page_number=line.page_number,
                local_line_number=line.local_line_number,
                source_filename=line.source_filename,
                is_empty=line.is_empty,
            )

        # Validate continuity
        if lines:
            expected = set(range(1, len(lines) + 1))
            actual = set(self._index.keys())
            missing = expected - actual
            if missing:
                logger.warning(
                    "Line numbering has %d gaps: %s",
                    len(missing),
                    sorted(missing)[:10],
                )

        logger.debug("Indexed %d lines", len(self._index))
        return lines

    def get_content_lines(self, lines: list[IndexedLine]) -> list[IndexedLine]:
        """Return only non-empty lines (preserving their original line numbers)."""
        return [line for line in lines if not line.is_empty and line.text.strip()]

    def lookup(self, global_line_number: int) -> LineMap | None:
        """Look up source location by global line number."""
        return self._index.get(global_line_number)

    @property
    def total_lines(self) -> int:
        """Total number of indexed lines (including empty)."""
        return len(self._index)

    @property
    def content_line_count(self) -> int:
        """Number of non-empty lines."""
        return sum(1 for m in self._index.values() if not m.is_empty)
