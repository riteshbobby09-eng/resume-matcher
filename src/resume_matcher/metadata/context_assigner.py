"""
Context assigner for Resume Matcher.

Enriches blocks with metadata:
- Assigned section (from detector or inferencer)
- Document type (JD vs Resume)
- Position metadata (document quartile)
- Processing version stamp
"""

from __future__ import annotations

import logging

from resume_matcher.domain.enums import DocumentType, SectionType
from resume_matcher.domain.models import Block, DocumentMeta
from resume_matcher.metadata.section_detector import DetectedSection

logger = logging.getLogger(__name__)


class ContextAssigner:
    """
    Assigns context and metadata to blocks using detected sections.

    Bridges section detection results with block-level metadata.
    """

    def assign_sections_from_detection(
        self,
        blocks: list[Block],
        detected_sections: list[DetectedSection],
        lines_count: int,
    ) -> list[Block]:
        """
        Assign sections to blocks based on detected section headings.

        Each block gets the section of the nearest preceding heading.
        Blocks before the first heading remain UNKNOWN (for inference later).

        Args:
            blocks: Blocks to assign sections to.
            detected_sections: Section headings detected in the document.
            lines_count: Total lines in document (for position calculation).

        Returns:
            Same blocks list with sections assigned.
        """
        if not detected_sections:
            return blocks

        # Sort sections by line number
        sorted_sections = sorted(detected_sections, key=lambda s: s.line_number)

        for block in blocks:
            if not block.source_line_numbers:
                continue

            block_start = min(block.source_line_numbers)

            # Find the section heading that precedes this block
            current_section = SectionType.UNKNOWN
            for sec in sorted_sections:
                if sec.line_number <= block_start:
                    current_section = sec.section_type
                else:
                    break

            if current_section != SectionType.UNKNOWN:
                block.section = current_section

        assigned_count = sum(1 for b in blocks if b.section != SectionType.UNKNOWN)
        logger.debug(
            "Assigned sections to %d of %d blocks from %d headings",
            assigned_count,
            len(blocks),
            len(detected_sections),
        )
        return blocks

    def assign_document_metadata(
        self,
        blocks: list[Block],
        meta: DocumentMeta,
    ) -> list[Block]:
        """
        Assign document-level metadata to all blocks.

        Args:
            blocks: Blocks to enrich.
            meta: Document metadata.

        Returns:
            Same blocks list with document metadata assigned.
        """
        for block in blocks:
            block.document_id = meta.document_id
            block.document_type = meta.document_type
            block.metadata["source_filename"] = meta.sanitized_filename
            block.metadata["processing_version"] = meta.processing_version

        return blocks
