"""
Abstract repository interfaces for Resume Matcher.

Defines the contract for data access. Currently implemented
in-memory, designed for future PostgreSQL swap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract repository with CRUD operations."""

    @abstractmethod
    def save(self, entity_id: str, entity: T) -> None:
        """Persist an entity."""
        ...

    @abstractmethod
    def get(self, entity_id: str) -> T | None:
        """Retrieve an entity by ID."""
        ...

    @abstractmethod
    def list_all(self) -> list[T]:
        """List all entities."""
        ...

    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete an entity. Returns True if it existed."""
        ...

    @abstractmethod
    def exists(self, entity_id: str) -> bool:
        """Check if an entity exists."""
        ...
