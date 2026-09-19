"""
FAISS index wrapper for Resume Matcher.

Uses IndexFlatIP (inner product) with L2-normalized vectors,
which is mathematically equivalent to cosine similarity.

Key operations:
- build(): Create index from embeddings + metadata
- search(): Find top-k similar vectors
- save() / load(): Persist/restore index
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from resume_matcher.config import RetrievalConfig
from resume_matcher.exceptions import ResumeMatcherError

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from FAISS."""

    index: int  # Position in the original dataset
    similarity_score: float  # Cosine similarity (0–1 for normalized vectors)
    metadata: dict[str, Any] = field(default_factory=dict)


class FaissIndex:
    """
    FAISS IndexFlatIP wrapper for cosine similarity search.

    Vectors MUST be L2-normalized before adding. Inner product on
    unit vectors equals cosine similarity.
    """

    def __init__(self, config: RetrievalConfig, dimension: int = 768) -> None:
        self._config = config
        self._dimension = dimension
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict[str, Any]] = []
        self._count = 0

    def build(
        self,
        vectors: np.ndarray,
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Build the FAISS index from vectors.

        Args:
            vectors: numpy array of shape (N, dimension), dtype float32,
                     MUST be L2-normalized.
            metadata_list: Optional list of metadata dicts (one per vector).
                           Stored alongside vectors for retrieval.

        Raises:
            ResumeMatcherError: If vectors have wrong shape or dtype.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise ResumeMatcherError(
                f"Expected vectors of shape (N, {self._dimension}), "
                f"got {vectors.shape}",
            )

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vectors)
        self._count = vectors.shape[0]

        if metadata_list:
            if len(metadata_list) != self._count:
                raise ResumeMatcherError(
                    f"Metadata count ({len(metadata_list)}) doesn't match "
                    f"vector count ({self._count})"
                )
            self._metadata = list(metadata_list)
        else:
            self._metadata = [{} for _ in range(self._count)]

        logger.info("FAISS index built: %d vectors of dim %d", self._count, self._dimension)

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int | None = None,
    ) -> list[list[SearchResult]]:
        """
        Search the index for similar vectors.

        Args:
            query_vectors: Query vectors, shape (Q, dimension), L2-normalized.
            top_k: Number of results per query. Defaults to config.top_k.

        Returns:
            List of lists of SearchResult (one list per query).
        """
        if self._index is None or self._count == 0:
            return [[] for _ in range(query_vectors.shape[0])]

        k = min(top_k or self._config.top_k, self._count)
        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

        distances, indices = self._index.search(query_vectors, k)

        results: list[list[SearchResult]] = []
        for q_idx in range(query_vectors.shape[0]):
            query_results: list[SearchResult] = []
            for rank in range(k):
                idx = int(indices[q_idx, rank])
                score = float(distances[q_idx, rank])

                if idx < 0:  # FAISS returns -1 for empty slots
                    continue

                query_results.append(
                    SearchResult(
                        index=idx,
                        similarity_score=score,
                        metadata=self._metadata[idx] if idx < len(self._metadata) else {},
                    )
                )
            results.append(query_results)

        return results

    def save(self, path: str | Path) -> None:
        """Save the FAISS index to disk."""
        if self._index is None:
            raise ResumeMatcherError("No index to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        logger.info("FAISS index saved to: %s", path)

    def load(self, path: str | Path) -> None:
        """Load a FAISS index from disk."""
        path = Path(path)
        if not path.exists():
            raise ResumeMatcherError(f"Index file not found: {path}")
        self._index = faiss.read_index(str(path))
        self._count = self._index.ntotal
        logger.info("FAISS index loaded: %d vectors", self._count)

    @property
    def count(self) -> int:
        """Number of vectors in the index."""
        return self._count
