"""
Block builder for Resume Matcher.

Groups sentences into meaningful chunks — the primary unit for
embedding and matching.

Hybrid chunking strategy:
1. Word-count limit: No chunk exceeds 40 words (configurable).
2. Section-aware boundaries: New section headings start new chunks.
3. Semantic grouping: Consecutive sentences in the same section group together.
4. Metadata preservation: Each chunk stores index_position, position_in_document,
   topic_boundary, section, and source_line_numbers.

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
    Builds meaningful blocks from sentences using hybrid chunking.

    Enforces a maximum word count per chunk (default 40 words).
    Respects section boundaries and preserves metadata for traceability.
    """

    def __init__(self, config: SegmentationConfig) -> None:
        self._max_words = config.max_words_per_chunk
        self._min_chars = config.min_block_length_chars
        self._merge_short = config.merge_short_blocks

    def build_blocks(
        self,
        sentences: list[Sentence],
        document_id: str = "",
        document_type: DocumentType = DocumentType.RESUME,
        total_lines: int = 0,
        section_break_lines: set[int] | None = None,
    ) -> list[Block]:
        """
        Group sentences into blocks respecting 40-word max and section boundaries.

        Args:
            sentences: Ordered list of sentences from the splitter.
            document_id: Document ID for traceability.
            document_type: Whether this is a JD or resume.
            total_lines: Total lines in document (for position calculation).
            section_break_lines: Optional line numbers of detected section headings.

        Returns:
            List of Block objects with metadata.
        """
        if not sentences:
            return []

        raw_blocks = self._create_raw_blocks(
            sentences,
            document_id,
            document_type,
            section_break_lines=section_break_lines,
        )

        # Merge short blocks if configured
        if self._merge_short:
            raw_blocks = self._merge_short_blocks(
                raw_blocks, section_breaks=section_break_lines
            )

        # Compute position in document and assign index metadata
        for idx, block in enumerate(raw_blocks):
            block.metadata["index_position"] = idx

            if total_lines > 0 and block.source_line_numbers:
                avg_line = sum(block.source_line_numbers) / len(
                    block.source_line_numbers
                )
                block.position_in_document = min(avg_line / total_lines, 1.0)

        logger.debug(
            "Built %d blocks from %d sentences (max %d words/chunk)",
            len(raw_blocks), len(sentences), self._max_words,
        )
        return raw_blocks

    def _create_raw_blocks(
        self,
        sentences: list[Sentence],
        document_id: str,
        document_type: DocumentType,
        section_break_lines: set[int] | None = None,
    ) -> list[Block]:
        """
        Create initial blocks with hybrid chunking.

        - Groups consecutive sentences up to max_words limit.
        - Starts a new block on section breaks or large line gaps.
        - Splits oversized sentences at clause boundaries.
        """
        blocks: list[Block] = []
        current_sentences: list[Sentence] = []
        current_word_count = 0
        section_breaks = section_break_lines or set()
        is_topic_boundary = True  # First block is always a topic boundary

        for sentence in sentences:
            sentence_words = len(sentence.text.split())
            sentence_lines = set(sentence.source_line_numbers)

            # Check for section break
            is_section_break = bool(sentence_lines & section_breaks) if current_sentences else False

            # Check for paragraph break (gap of 2+ lines)
            is_gap_break = False
            if current_sentences and sentence.source_line_numbers and current_sentences[-1].source_line_numbers:
                prev_max = max(current_sentences[-1].source_line_numbers)
                curr_min = min(sentence.source_line_numbers)
                if curr_min - prev_max > 2:
                    is_gap_break = True

            # Would adding this sentence exceed word limit?
            would_exceed = (current_word_count + sentence_words) > self._max_words

            # Flush current block if needed
            if (is_section_break or is_gap_break or would_exceed) and current_sentences:
                block = self._make_block(
                    current_sentences, document_id, document_type,
                    topic_boundary=is_topic_boundary,
                )
                blocks.append(block)
                current_sentences = []
                current_word_count = 0
                is_topic_boundary = is_section_break or is_gap_break

            # Handle oversized sentences (> max_words)
            if sentence_words > self._max_words:
                # Split at clause boundaries
                sub_sents = self._split_oversized_sentence(sentence, document_id)
                for sub in sub_sents:
                    sub_words = len(sub.text.split())
                    if current_word_count + sub_words > self._max_words and current_sentences:
                        block = self._make_block(
                            current_sentences, document_id, document_type,
                            topic_boundary=is_topic_boundary,
                        )
                        blocks.append(block)
                        current_sentences = []
                        current_word_count = 0
                        is_topic_boundary = False

                    current_sentences.append(sub)
                    current_word_count += sub_words
            else:
                current_sentences.append(sentence)
                current_word_count += sentence_words

        # Flush remaining
        if current_sentences:
            block = self._make_block(
                current_sentences, document_id, document_type,
                topic_boundary=is_topic_boundary,
            )
            blocks.append(block)

        return blocks

    def _split_oversized_sentence(
        self, sentence: Sentence, document_id: str
    ) -> list[Sentence]:
        """
        Split an oversized sentence at clause boundaries.

        Tries splitting at semicolons, commas, conjunctions, or
        falls back to word-level splitting at max_words boundaries.
        """
        import re
        text = sentence.text

        # Try clause-level splitting: semicolons, " and ", " or ", commas
        parts: list[str] = []
        for delimiter_pattern in [
            r';\s*',                           # Semicolons
            r',\s+(?=and\b|or\b|but\b|which\b|where\b|while\b)',  # Commas before conjunctions
            r',\s+',                           # Commas
        ]:
            candidate_parts = re.split(delimiter_pattern, text)
            if len(candidate_parts) > 1:
                parts = [p.strip() for p in candidate_parts if p.strip()]
                break

        if not parts:
            # Last resort: split at word boundaries
            words = text.split()
            parts = []
            for i in range(0, len(words), self._max_words):
                chunk = " ".join(words[i:i + self._max_words])
                parts.append(chunk)

        return [
            Sentence(
                text=part,
                source_line_numbers=list(sentence.source_line_numbers),
                document_id=document_id,
            )
            for part in parts if part
        ]

    def _merge_short_blocks(
        self, blocks: list[Block], section_breaks: set[int] | None = None
    ) -> list[Block]:
        """
        Merge blocks that are too short (below min_chars) with neighbors.

        Short blocks are merged with the NEXT block if possible,
        otherwise with the PREVIOUS block. Blocks across section headings are never merged.
        """
        if len(blocks) <= 1:
            return blocks

        breaks = section_breaks or set()
        merged: list[Block] = []
        i = 0

        while i < len(blocks):
            block = blocks[i]

            if len(block.text) < self._min_chars and i + 1 < len(blocks):
                next_block = blocks[i + 1]
                # Never merge across a section break
                next_lines = set(next_block.source_line_numbers)
                if next_lines & breaks:
                    merged.append(block)
                    i += 1
                    continue

                # Check combined word count doesn't exceed limit
                combined_words = len(block.text.split()) + len(next_block.text.split())
                if combined_words <= self._max_words:
                    combined_sentences = block.sentences + next_block.sentences
                    merged_block = Block(
                        sentences=combined_sentences,
                        document_id=block.document_id,
                        document_type=block.document_type,
                    )
                    # Preserve topic_boundary from the first block
                    merged_block.metadata["topic_boundary"] = block.metadata.get(
                        "topic_boundary", False
                    )
                    merged.append(merged_block)
                    i += 2  # Skip next block
                else:
                    merged.append(block)
                    i += 1
            else:
                merged.append(block)
                i += 1

        return merged

    def _make_block(
        self,
        sentences: list[Sentence],
        document_id: str,
        document_type: DocumentType,
        topic_boundary: bool = False,
    ) -> Block:
        """Create a Block from a list of sentences with metadata."""
        block = Block(
            sentences=list(sentences),
            document_id=document_id,
            document_type=document_type,
        )
        block.metadata["topic_boundary"] = topic_boundary
        return block
