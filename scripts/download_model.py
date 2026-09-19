"""
Pre-download the BGE embedding model.

Run this before the first pipeline execution to download
the model (~440MB) to the local cache.

Usage:
    python -m scripts.download_model
"""

from __future__ import annotations

import logging
import sys


def main() -> None:
    """Download and cache the BGE model."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    model_name = "BAAI/bge-base-en-v1.5"
    logger.info("Downloading model: %s", model_name)
    logger.info("This may take a few minutes (~440MB)...")

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        # Quick test
        test_embedding = model.encode(["test"])
        logger.info("✅ Model downloaded and verified!")
        logger.info("   Embedding dimension: %d", test_embedding.shape[1])
        logger.info("   Model cached for future use")
    except Exception as exc:
        logger.error("❌ Failed to download model: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
