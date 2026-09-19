"""
Block builder for Resume Matcher.

Groups sentences into meaningful blocks — the primary unit for
embedding and matching.

Blocks are created based on:
1. Section boundaries (new section heading = new block)
2. Blank-line gaps (paragraph breaks)
3. Maximum block size (configurable, default 5 sentences)
4. Semantic coherence: bullet lists stay together, paragraph content groups

Each Block preserves its constituent sentences and their line numbers.
"""

from __future__ import annotations

import logging
from typing import Any

from resume_matcher.config import SegmentationConfig
from resume_matcher.domain.enums import DocumentType, SectionType
from resume_matcher.domain.models import Block, Sentence

logger = logging.getLogger(__name__)


class BlockBuilder:
    """
    Builds meaningful blocks from sentences.

    Does NOT assign sections (that's the metadata module's job).
    Focuses purely on grouping sentences into coherent blocks.
    """

    def __init__(self, config: SegmentationConfig) -> None:
        self._max_sentences = config.max_sentences_per_block
        self._min_chars = config.min_block_length_chars
        self._merge_short = config.merge_short_blocks

    def build_blocks(
        self,
        sentences: list[Sentence],
        document_id: str = "",
        document_type: DocumentType = DocumentType.RESUME,
        total_lines: int = 0,
    ) -> list[Block]:
        """
        Group sentences into blocks.

        Args:
            sentences: Ordered list of sentences from the splitter.
            document_id: Document ID for traceability.
            document_type: Whether this is a JD or resume.
            total_lines: Total lines in document (for position calculation).

        Returns:
            List of Block objects.
        """
        if not sentences:
            return []

        raw_blocks = self._create_raw_blocks(sentences, document_id, document_type)

        # Merge short blocks if configured
        if self._merge_short:
            raw_blocks = self._merge_short_blocks(raw_blocks)

        # Compute position in document
        if total_lines > 0:
            for block in raw_blocks:
                if block.source_line_numbers:
                    avg_line = sum(block.source_line_numbers) / len(
                        block.source_line_numbers
                    )
                    block.position_in_document = min(avg_line / total_lines, 1.0)

        logger.debug(
            "Built %d blocks from %d sentences", len(raw_blocks), len(sentences)
        )
        return raw_blocks

    def _create_raw_blocks(
        self,
        sentences: list[Sentence],
        document_id: str,
        document_type: DocumentType,
    ) -> list[Block]:
        """
        Create initial blocks by grouping sentences.

        Groups consecutive sentences up to max_sentences limit.
        Starts a new block on natural breaks (gaps in line numbers).
        """
        blocks: list[Block] = []
        current_sentences: list[Sentence] = []

        for i, sentence in enumerate(sentences):
            # Check for natural break: gap in line numbers
            is_break = False
            if current_sentences and sentence.source_line_numbers and current_sentences[-1].source_line_numbers:
                prev_max = max(current_sentences[-1].source_line_numbers)
                curr_min = min(sentence.source_line_numbers)
                # Gap of 2+ lines suggests a paragraph break
                if curr_min - prev_max > 2:
                    is_break = True

            # Check for max size
            at_max = len(current_sentences) >= self._max_sentences

            if (is_break or at_max) and current_sentences:
                blocks.append(
                    self._make_block(current_sentences, document_id, document_type)
                )
                current_sentences = []

            current_sentences.append(sentence)

        # Flush remaining
        if current_sentences:
            blocks.append(
                self._make_block(current_sentences, document_id, document_type)
            )

        return blocks

    def _merge_short_blocks(self, blocks: list[Block]) -> list[Block]:
        """
        Merge blocks that are too short (below min_chars) with neighbors.

        Short blocks are merged with the NEXT block if possible,
        otherwise with the PREVIOUS block.
        """
        if len(blocks) <= 1:
            return blocks

        merged: list[Block] = []
        i = 0

        while i < len(blocks):
            block = blocks[i]

            if len(block.text) < self._min_chars and i + 1 < len(blocks):
                # Merge with next block
                next_block = blocks[i + 1]
                combined_sentences = block.sentences + next_block.sentences
                merged_block = Block(
                    sentences=combined_sentences,
                    document_id=block.document_id,
                    document_type=block.document_type,
                )
                merged.append(merged_block)
                i += 2  # Skip next block
            else:
                merged.append(block)
                i += 1

        return merged

    def _make_block(
        self,
        sentences: list[Sentence],
        document_id: str,
        document_type: DocumentType,
    ) -> Block:
        """Create a Block from a list of sentences."""
        return Block(
            sentences=list(sentences),
            document_id=document_id,
            document_type=document_type,
        )
