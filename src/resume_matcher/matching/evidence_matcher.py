"""
Evidence matcher for Resume Matcher.

For each JD block, retrieves the most similar resume blocks via FAISS
and constructs MatchEvidence with full audit trail:
- matched text, similarity scores, source sections, line numbers
- match strength classification, missing evidence, uncertainty flags
- human-readable explanations
"""

from __future__ import annotations

import logging

import numpy as np

from resume_matcher.config import RetrievalConfig, ScoringThresholds
from resume_matcher.domain.enums import DocumentType, MatchStrength, SectionType
from resume_matcher.domain.models import Block, MatchEvidence
from resume_matcher.embeddings.bge_embedder import BgeEmbedder
from resume_matcher.retrieval.faiss_index import FaissIndex
from resume_matcher.retrieval.metadata_filter import MetadataFilter

logger = logging.getLogger(__name__)


class EvidenceMatcher:
    """
    Matches JD blocks against resume blocks using FAISS retrieval.

    Produces MatchEvidence objects with complete traceability.
    """

    def __init__(
        self,
        embedder: BgeEmbedder,
        retrieval_config: RetrievalConfig,
        thresholds: ScoringThresholds,
    ) -> None:
        self._embedder = embedder
        self._retrieval_config = retrieval_config
        self._thresholds = thresholds
        self._metadata_filter = MetadataFilter(retrieval_config)

    def match(
        self,
        jd_blocks: list[Block],
        resume_blocks: list[Block],
    ) -> list[MatchEvidence]:
        """
        Match JD blocks against resume blocks.

        Args:
            jd_blocks: Blocks from the job description.
            resume_blocks: Blocks from the resume.

        Returns:
            List of MatchEvidence, one per JD block (best match).
            Also includes all matches for scoring purposes.
        """
        if not jd_blocks or not resume_blocks:
            return []

        # Embed resume blocks and build FAISS index
        resume_embeddings = self._embedder.embed_blocks(resume_blocks)
        resume_metadata = [
            {
                "block_id": b.block_id,
                "section": b.section.value,
                "document_type": b.document_type.value,
                "text": b.text,
                "line_numbers": b.source_line_numbers,
            }
            for b in resume_blocks
        ]

        faiss_index = FaissIndex(self._retrieval_config, self._embedder.dimension)
        faiss_index.build(resume_embeddings, resume_metadata)

        # Embed JD blocks and search
        jd_embeddings = self._embedder.embed_blocks(jd_blocks)
        search_results = faiss_index.search(jd_embeddings)

        # Build match evidence
        all_evidence: list[MatchEvidence] = []

        for jd_idx, jd_block in enumerate(jd_blocks):
            results = search_results[jd_idx] if jd_idx < len(search_results) else []

            # Apply metadata filtering
            filtered = self._metadata_filter.filter_results(
                results,
                query_section=jd_block.section,
                required_doc_type=DocumentType.RESUME,
            )

            if not filtered:
                # No match found → missing evidence
                all_evidence.append(
                    MatchEvidence(
                        jd_block_id=jd_block.block_id,
                        jd_text=jd_block.text,
                        jd_section=jd_block.section,
                        jd_line_numbers=jd_block.source_line_numbers,
                        match_strength=MatchStrength.NONE,
                        explanation=f"No matching resume content found for: {jd_block.text[:100]}",
                    )
                )
                continue

            # Create evidence for each match
            for result in filtered:
                meta = result.metadata
                score = result.similarity_score
                strength = self._classify_strength(score)

                evidence = MatchEvidence(
                    jd_block_id=jd_block.block_id,
                    resume_block_id=meta.get("block_id", ""),
                    jd_text=jd_block.text,
                    resume_text=meta.get("text", ""),
                    similarity_score=score,
                    jd_section=jd_block.section,
                    resume_section=SectionType(meta.get("section", "unknown")),
                    jd_line_numbers=jd_block.source_line_numbers,
                    resume_line_numbers=meta.get("line_numbers", []),
                    match_strength=strength,
                    explanation=self._build_explanation(
                        jd_block, meta, score, strength
                    ),
                    uncertainty=(strength == MatchStrength.WEAK),
                )
                all_evidence.append(evidence)

        logger.info(
            "Matched %d JD blocks → %d evidence items",
            len(jd_blocks),
            len(all_evidence),
        )
        return all_evidence

    def _classify_strength(self, score: float) -> MatchStrength:
        """Classify match strength based on similarity score thresholds."""
        if score >= self._thresholds.strong_match:
            return MatchStrength.STRONG
        elif score >= self._thresholds.moderate_match:
            return MatchStrength.MODERATE
        elif score >= self._thresholds.weak_match:
            return MatchStrength.WEAK
        else:
            return MatchStrength.NONE

    def _build_explanation(
        self,
        jd_block: Block,
        resume_meta: dict,
        score: float,
        strength: MatchStrength,
    ) -> str:
        """Build a human-readable explanation for the match."""
        parts = [
            f"Match strength: {strength.value} (similarity: {score:.3f})",
            f"JD section: {jd_block.section.value}",
            f"Resume section: {resume_meta.get('section', 'unknown')}",
        ]
        if jd_block.section.value == resume_meta.get("section"):
            parts.append("✓ Same section match (boosted)")
        else:
            parts.append("△ Cross-section match")
        return " | ".join(parts)
