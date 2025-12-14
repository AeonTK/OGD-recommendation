from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.vectorstore.data_store import DataVectorStore
from src.vectorstore.embeddings import Embedder
from src.vectorstore.retriever import Retriever


@dataclass(frozen=True)
class SearchServiceConfig:
    collection_name: str = "programs"


class SearchService:
    """Application-layer search service.

    Implements:
      1) Search by query text
      2) Search for items similar to an existing item by id

    Returns minimal results: {id, metadata}.

    Note: This service relies on the existing vectorstore layer and does not
    modify it.
    """

    def __init__(self, config: SearchServiceConfig | None = None):
        self.config = config or SearchServiceConfig()

        # Share the same embedder between store and retriever to guarantee dim consistency.
        self.embedder = Embedder()
        self.store = DataVectorStore(collection=self.config.collection_name, embedder=self.embedder)
        self.retriever = Retriever(store=self.store, embedder=self.embedder)

    def search_by_query(
        self,
        query: str,
        *,
        top_k: int = 10,
        types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search by a natural-language query string."""

        if not query or not query.strip():
            return []

        filter_expression = self._build_filter_expression(types=types)
        items = self.retriever.retrieve(
            query.strip(),
            limit=top_k,
            filter_expression=filter_expression,
        )

        return [self._to_result(item) for item in items]

    def search_similar_by_id(
        self,
        source_id: str,
        *,
        top_k: int = 10,
        types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        pass

    def _to_result(self, item) -> Dict[str, Any]:
        return {
            "id": str(item.id),
            "metadata": item.metadata or {},
        }

    def _build_filter_expression(self, *, types: Optional[List[str]]) -> Optional[str]:
        """Build a Milvus filter expression.

        We keep this intentionally small: only support filtering by object type.

        Assumption: you store object type in metadata under one of these keys:
          - metadata["type"]
          - metadata["object_type"]

        If the expression fails (e.g. metadata not present), vectorstore code may still work
        without the filter; callers can also choose to post-filter.
        """

        if not types:
            return None

        clean_types = [t.strip() for t in types if t and t.strip()]
        if not clean_types:
            return None

        # Milvus JSON field expr typically supports: metadata["key"] in ["a","b"]
        quoted = ", ".join([f'"{t}"' for t in clean_types])
        return f'(metadata["type"] in [{quoted}]) or (metadata["object_type"] in [{quoted}])'
