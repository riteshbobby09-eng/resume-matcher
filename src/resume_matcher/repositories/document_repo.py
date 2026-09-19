"""
In-memory document repository.
"""

from __future__ import annotations

import logging
import threading

from resume_matcher.domain.models import ProcessedDocument
from resume_matcher.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class InMemoryDocumentRepository(BaseRepository[ProcessedDocument]):
    """Thread-safe in-memory storage for processed documents."""

    def __init__(self) -> None:
        self._store: dict[str, ProcessedDocument] = {}
        self._lock = threading.Lock()

    def save(self, entity_id: str, entity: ProcessedDocument) -> None:
        with self._lock:
            self._store[entity_id] = entity
        logger.debug("Document saved: %s", entity_id)

    def get(self, entity_id: str) -> ProcessedDocument | None:
        with self._lock:
            return self._store.get(entity_id)

    def list_all(self) -> list[ProcessedDocument]:
        with self._lock:
            return list(self._store.values())

    def delete(self, entity_id: str) -> bool:
        with self._lock:
            if entity_id in self._store:
                del self._store[entity_id]
                return True
            return False

    def exists(self, entity_id: str) -> bool:
        with self._lock:
            return entity_id in self._store

    def clear(self) -> None:
        """Remove all stored documents."""
        with self._lock:
            self._store.clear()
