"""
Metadata filter for post-retrieval filtering.

Applies business rules AFTER FAISS retrieval:
- Filter by section type
- Filter by document type
- Minimum similarity threshold
- Section-aware boosting
"""

from __future__ import annotations

import logging

from resume_matcher.config import RetrievalConfig
from resume_matcher.domain.enums import DocumentType, SectionType
from resume_matcher.retrieval.faiss_index import SearchResult

logger = logging.getLogger(__name__)


class MetadataFilter:
    """
    Filters and re-scores FAISS search results using metadata.

    Applied AFTER vector retrieval to enforce business rules that
    can't be expressed in vector space.
    """

    def __init__(self, config: RetrievalConfig) -> None:
        self._threshold = config.similarity_threshold
        self._section_boost = config.section_match_boost

    def filter_results(
        self,
        results: list[SearchResult],
        query_section: SectionType = SectionType.UNKNOWN,
        required_doc_type: DocumentType | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """
        Filter search results based on metadata criteria.

        Args:
            results: Raw search results from FAISS.
            query_section: Section of the query block (for boosting).
            required_doc_type: Only keep results of this document type.
            min_similarity: Minimum similarity threshold (overrides config).

        Returns:
            Filtered and re-scored results, sorted by score descending.
        """
        threshold = min_similarity if min_similarity is not None else self._threshold
        filtered: list[SearchResult] = []

        for result in results:
            # Filter by minimum similarity
            if result.similarity_score < threshold:
                continue

            # Filter by document type
            if required_doc_type is not None:
                result_doc_type = result.metadata.get("document_type")
                if result_doc_type and result_doc_type != required_doc_type.value:
                    continue

            # Section-aware boosting
            result_section = result.metadata.get("section", "")
            if (
                query_section != SectionType.UNKNOWN
                and result_section == query_section.value
            ):
                result.similarity_score = min(
                    result.similarity_score + self._section_boost, 1.0
                )

            filtered.append(result)

        # Sort by score descending
        filtered.sort(key=lambda r: r.similarity_score, reverse=True)

        logger.debug(
            "Filtered %d → %d results (threshold=%.2f)",
            len(results),
            len(filtered),
            threshold,
        )
        return filtered
