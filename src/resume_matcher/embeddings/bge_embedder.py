"""
BGE embedder for Resume Matcher.

Wraps SentenceTransformer("BAAI/bge-base-en-v1.5") for:
- Block text → 768-dimensional embedding vectors
- Batch encoding with configurable batch size
- L2 normalization (for cosine similarity via FAISS IndexFlatIP)
- Singleton model loading (reuse across calls)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from resume_matcher.config import EmbeddingConfig
from resume_matcher.domain.models import Block
from resume_matcher.exceptions import EmbeddingError, ModelLoadError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Module-level model cache (singleton)
_model_cache: dict[str, "SentenceTransformer"] = {}


class BgeEmbedder:
    """
    Generates embeddings using BAAI/bge-base-en-v1.5.

    The model produces 768-dimensional vectors. Vectors are L2-normalized
    so that inner product (FAISS IndexFlatIP) equals cosine similarity.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> "SentenceTransformer":
        """Load or retrieve cached model (singleton)."""
        if self._model is not None:
            return self._model

        model_name = self._config.model_name
        if model_name in _model_cache:
            self._model = _model_cache[model_name]
            logger.debug("Reusing cached model: %s", model_name)
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", model_name)
            model = SentenceTransformer(model_name)
            _model_cache[model_name] = model
            self._model = model
            logger.info("Model loaded successfully: %s", model_name)
            return model
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load embedding model: {model_name}",
                details={"model_name": model_name, "error": str(exc)},
            ) from exc

    def embed_blocks(self, blocks: list[Block]) -> np.ndarray:
        """
        Generate embeddings for a list of blocks.

        Args:
            blocks: Blocks to embed. Uses block.text for encoding.

        Returns:
            numpy array of shape (N, 768), dtype float32, L2-normalized.

        Raises:
            EmbeddingError: If encoding fails.
        """
        if not blocks:
            return np.empty((0, self._config.dimension), dtype=np.float32)

        texts = [block.text for block in blocks]
        return self.embed_texts(texts)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: Strings to encode.

        Returns:
            numpy array of shape (N, 768), dtype float32, L2-normalized.
        """
        if not texts:
            return np.empty((0, self._config.dimension), dtype=np.float32)

        model = self._get_model()

        try:
            embeddings = model.encode(
                texts,
                batch_size=self._config.batch_size,
                show_progress_bar=self._config.show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self._config.normalize,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to encode {len(texts)} texts",
                details={"count": len(texts), "error": str(exc)},
            ) from exc

        # Ensure float32
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Double-check normalization
        if self._config.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        logger.debug(
            "Encoded %d texts → shape %s", len(texts), embeddings.shape
        )
        return embeddings

    @property
    def dimension(self) -> int:
        """Embedding dimension (768 for bge-base-en-v1.5)."""
        return self._config.dimension
