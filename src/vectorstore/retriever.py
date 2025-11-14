from __future__ import annotations

from typing import List
from uuid import UUID

from src.vectorstore.data_store import DataVectorStore
from src.vectorstore.embeddings import Embedder

from .schemas import SearchItem


class Retriever:
    def __init__(self, store: DataVectorStore | None = None, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.store = store or DataVectorStore(embedder=self.embedder)

    def _results_to_items(self, results) -> List[SearchItem]:
        """Convert Milvus hybrid_search results into our SearchItem list."""
        if not results:
            return []
        hits = results[0]
        items: List[SearchItem] = []
        for hit in hits:
            cid_str = str(hit.get("id")) if isinstance(hit, dict) else str(getattr(hit, "id", ""))
            text = hit.get("text") if isinstance(hit, dict) else getattr(hit, "text", "")
            metadata = hit.get("metadata") if isinstance(hit, dict) else getattr(hit, "metadata", None)
            distance = hit.get("distance") if isinstance(hit, dict) else getattr(hit, "distance", None)
            try:
                dist = float(distance) if distance is not None else None
            except Exception:
                dist = None
            pid = UUID(cid_str)
            meta_dict = metadata if isinstance(metadata, dict) else None
            items.append(SearchItem(id=pid, text=text or "", distance=dist, metadata=meta_dict))
        return items

    def retrieve(self, query: str, limit: int = 10) -> List[SearchItem]:
        """Synchronous retrieval using sync embeddings."""
        qv = self.embedder.embed_query(query)
        results = self.store.hybrid_search(query_text=query, query_dense=qv, limit=limit)
        return self._results_to_items(results)

    async def aretrieve(self, query: str, limit: int = 10) -> List[SearchItem]:
        """Asynchronous retrieval using async embeddings."""
        qv = await self.embedder.aembed_query(query)
        results = self.store.hybrid_search(query_text=query, query_dense=qv, limit=limit)
        return self._results_to_items(results)
