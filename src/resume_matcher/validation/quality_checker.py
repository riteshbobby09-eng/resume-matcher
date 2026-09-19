"""
Quality checker for Resume Matcher.

Validates processed documents for completeness and consistency:
- Document has minimum content
- All blocks have assigned sections
- No duplicate blocks
- Line number continuity
- Section distribution sanity
"""

from __future__ import annotations

import hashlib
import logging

from resume_matcher.domain.enums import SectionType, ValidationStatus
from resume_matcher.domain.models import (
    Block,
    ProcessedDocument,
    QualityReport,
    ValidationMessage,
)

logger = logging.getLogger(__name__)

# Minimum thresholds for a valid document
_MIN_LINES = 3
_MIN_BLOCKS = 1
_DUPLICATE_SIMILARITY_THRESHOLD = 0.95


class QualityChecker:
    """
    Performs quality checks on processed documents.

    Produces a QualityReport with messages, warnings, and review flags.
    Does NOT reject documents — only flags issues for review.
    """

    def check(self, document: ProcessedDocument) -> QualityReport:
        """
        Run all quality checks on a processed document.

        Args:
            document: Fully processed document.

        Returns:
            QualityReport with findings.
        """
        report = QualityReport(
            document_id=document.meta.document_id,
            total_lines=len(document.lines),
            total_sentences=len(document.sentences),
            total_blocks=len(document.blocks),
        )

        messages: list[ValidationMessage] = []

        # Check 1: Minimum content
        messages.extend(self._check_minimum_content(document))

        # Check 2: Section assignments
        messages.extend(self._check_section_assignments(document.blocks, report))

        # Check 3: Duplicate blocks
        messages.extend(self._check_duplicates(document.blocks))

        # Check 4: Line number continuity
        messages.extend(self._check_line_continuity(document))

        # Check 5: Section distribution
        messages.extend(self._check_section_distribution(document.blocks))

        report.messages = messages

        # Determine overall status
        has_errors = any(m.status == ValidationStatus.ERROR for m in messages)
        has_warnings = any(m.status == ValidationStatus.WARNING for m in messages)

        if has_errors:
            report.overall_status = ValidationStatus.ERROR
        elif has_warnings:
            report.overall_status = ValidationStatus.WARNING
        else:
            report.overall_status = ValidationStatus.VALID

        # Collect review flags
        report.review_flags = [
            m.message for m in messages if m.status != ValidationStatus.VALID
        ]

        logger.info(
            "Quality check: %s — %d messages (%s)",
            document.meta.sanitized_filename,
            len(messages),
            report.overall_status.value,
        )
        return report

    def _check_minimum_content(
        self, doc: ProcessedDocument
    ) -> list[ValidationMessage]:
        """Check document has minimum meaningful content."""
        messages: list[ValidationMessage] = []

        if len(doc.lines) < _MIN_LINES:
            messages.append(
                ValidationMessage(
                    status=ValidationStatus.WARNING,
                    code="MIN_LINES",
                    message=f"Document has only {len(doc.lines)} lines (minimum: {_MIN_LINES})",
                    location=f"document:{doc.meta.document_id}",
                )
            )

        if len(doc.blocks) < _MIN_BLOCKS:
            messages.append(
                ValidationMessage(
                    status=ValidationStatus.ERROR,
                    code="MIN_BLOCKS",
                    message=f"Document has {len(doc.blocks)} blocks (minimum: {_MIN_BLOCKS})",
                    location=f"document:{doc.meta.document_id}",
                )
            )

        content_text = " ".join(b.text for b in doc.blocks)
        if len(content_text.strip()) < 50:
            messages.append(
                ValidationMessage(
                    status=ValidationStatus.WARNING,
                    code="LOW_CONTENT",
                    message=f"Document has very little text content ({len(content_text)} chars)",
                    location=f"document:{doc.meta.document_id}",
                )
            )

        return messages

    def _check_section_assignments(
        self, blocks: list[Block], report: QualityReport
    ) -> list[ValidationMessage]:
        """Check that blocks have section assignments."""
        messages: list[ValidationMessage] = []
        sections_found: set[SectionType] = set()
        unknown_count = 0

        for block in blocks:
            sections_found.add(block.section)
            if block.section == SectionType.UNKNOWN:
                unknown_count += 1

        report.sections_found = sorted(sections_found, key=lambda s: s.value)

        if unknown_count > 0:
            ratio = unknown_count / max(len(blocks), 1)
            status = ValidationStatus.WARNING if ratio > 0.3 else ValidationStatus.VALID
            messages.append(
                ValidationMessage(
                    status=status,
                    code="UNKNOWN_SECTIONS",
                    message=(
                        f"{unknown_count} of {len(blocks)} blocks have UNKNOWN sections "
                        f"({ratio:.0%})"
                    ),
                    location=f"document:blocks",
                )
            )

        return messages

    def _check_duplicates(self, blocks: list[Block]) -> list[ValidationMessage]:
        """Check for duplicate or near-duplicate blocks."""
        messages: list[ValidationMessage] = []
        seen_hashes: dict[str, str] = {}  # hash → block_id

        for block in blocks:
            text_hash = hashlib.md5(
                block.text.strip().lower().encode()
            ).hexdigest()

            if text_hash in seen_hashes:
                messages.append(
                    ValidationMessage(
                        status=ValidationStatus.WARNING,
                        code="DUPLICATE_BLOCK",
                        message=(
                            f"Block {block.block_id} is a duplicate of "
                            f"{seen_hashes[text_hash]}"
                        ),
                        location=f"block:{block.block_id}",
                    )
                )
            else:
                seen_hashes[text_hash] = block.block_id

        return messages

    def _check_line_continuity(
        self, doc: ProcessedDocument
    ) -> list[ValidationMessage]:
        """Check that line numbers are continuous."""
        messages: list[ValidationMessage] = []
        if not doc.lines:
            return messages

        line_nums = [ln.global_line_number for ln in doc.lines]
        expected = set(range(1, max(line_nums) + 1))
        actual = set(line_nums)
        missing = expected - actual

        if missing:
            messages.append(
                ValidationMessage(
                    status=ValidationStatus.WARNING,
                    code="LINE_GAPS",
                    message=f"Line numbering has {len(missing)} gaps",
                    location=f"document:{doc.meta.document_id}",
                )
            )

        return messages

    def _check_section_distribution(
        self, blocks: list[Block]
    ) -> list[ValidationMessage]:
        """Sanity check section distribution."""
        messages: list[ValidationMessage] = []
        if not blocks:
            return messages

        section_counts: dict[SectionType, int] = {}
        for block in blocks:
            section_counts[block.section] = section_counts.get(block.section, 0) + 1

        # Flag if no EXPERIENCE or SKILLS sections found (common sections)
        if (
            SectionType.EXPERIENCE not in section_counts
            and SectionType.SKILLS not in section_counts
        ):
            messages.append(
                ValidationMessage(
                    status=ValidationStatus.WARNING,
                    code="MISSING_KEY_SECTIONS",
                    message="No EXPERIENCE or SKILLS sections found in document",
                    location="document:sections",
                )
            )

        return messages
