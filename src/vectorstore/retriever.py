from __future__ import annotations

import platform
from typing import List, overload
from uuid import UUID

from src.vectorstore.data_store import DataVectorStore
from src.vectorstore.embeddings import Embedder

from .schemas import SearchItem

# Platform-specific default batch sizes for hybrid search
# Can be overridden with MILVUS_BATCH_SIZE environment variable
import os
DEFAULT_BATCH_SIZE = int(os.getenv("MILVUS_BATCH_SIZE", 
                                  1024 if platform.system() == "Windows" else 32))


def _create_batches(items: List, batch_size: int = 32) -> List[List]:
    """Split a list into batches of specified size."""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


class Retriever:
    def __init__(
        self, store: DataVectorStore | None = None, embedder: Embedder | None = None
    ):
        self.embedder = embedder or Embedder()
        self.store = store or DataVectorStore(embedder=self.embedder)

    def _results_to_items(self, results) -> List[SearchItem]:
        """Convert Milvus hybrid_search results into our SearchItem list."""
        if not results:
            return []
        hits = results[0]
        items: List[SearchItem] = []
        for hit in hits:
            cid_str = (
                str(hit.get("id"))
                if isinstance(hit, dict)
                else str(getattr(hit, "id", ""))
            )
            text = (
                hit.get("text") if isinstance(hit, dict) else getattr(hit, "text", "")
            )
            metadata = (
                hit.get("metadata")
                if isinstance(hit, dict)
                else getattr(hit, "metadata", None)
            )
            distance = (
                hit.get("distance")
                if isinstance(hit, dict)
                else getattr(hit, "distance", None)
            )
            try:
                dist = float(distance) if distance is not None else None
            except Exception:
                dist = None
            pid = UUID(cid_str)
            meta_dict = metadata if isinstance(metadata, dict) else None
            items.append(
                SearchItem(id=pid, text=text or "", distance=dist, metadata=meta_dict)
            )
        return items
    
    def _batch_results_to_items(self, results) -> List[List[SearchItem]]:
        """Convert batch Milvus hybrid_search results into list of SearchItem lists."""
        if not results:
            return []
        
        batch_items: List[List[SearchItem]] = []
        for result_set in results:
            items: List[SearchItem] = []
            for hit in result_set:
                cid_str = (
                    str(hit.get("id"))
                    if isinstance(hit, dict)
                    else str(getattr(hit, "id", ""))
                )
                text = (
                    hit.get("text") if isinstance(hit, dict) else getattr(hit, "text", "")
                )
                metadata = (
                    hit.get("metadata")
                    if isinstance(hit, dict)
                    else getattr(hit, "metadata", None)
                )
                distance = (
                    hit.get("distance")
                    if isinstance(hit, dict)
                    else getattr(hit, "distance", None)
                )
                try:
                    dist = float(distance) if distance is not None else None
                except Exception:
                    dist = None
                pid = UUID(cid_str)
                meta_dict = metadata if isinstance(metadata, dict) else None
                items.append(
                    SearchItem(id=pid, text=text or "", distance=dist, metadata=meta_dict)
                )
            batch_items.append(items)
        return batch_items

    @overload
    def retrieve(
        self,
        query_or_queries: str,
        limit: int = 10,
        queries_as_documents: bool = True,
        *args,
        **kwargs,
    ) -> List[SearchItem]: ...

    @overload
    def retrieve(
        self,
        query_or_queries: List[str],
        limit: int = 10,
        queries_as_documents: bool = True,
        *args,
        **kwargs,
    ) -> List[List[SearchItem]]: ...

    def retrieve(
        self,
        query_or_queries: str | List[str],
        limit: int = 10,
        queries_as_documents: bool = True,
        *args,
        **kwargs,
    ):
        if isinstance(query_or_queries, list):
            # Embed all queries together first
            if queries_as_documents:
                qvs = self.embedder.embed_documents(query_or_queries)
            else:
                qvs = [self.embedder.embed_query(q) for q in query_or_queries]
            
            # Process hybrid_search in platform-appropriate batches
            all_results = []
            query_batches = _create_batches(query_or_queries, DEFAULT_BATCH_SIZE)
            dense_batches = _create_batches(qvs, DEFAULT_BATCH_SIZE)
            
            for query_batch, dense_batch in zip(query_batches, dense_batches):
                batch_results = self.store.hybrid_search(
                    query_texts=query_batch,
                    query_denses=dense_batch,
                    limit=limit,
                    *args,
                    **kwargs,
                )
                # Convert batch results and extend to all_results
                batch_items = self._batch_results_to_items(batch_results)
                all_results.extend(batch_items)
            
            return all_results
        elif isinstance(query_or_queries, str):
            qv = self.embedder.embed_query(query_or_queries)
            # Single query - wrap in lists for batch processing
            results = self.store.hybrid_search(
                query_texts=[query_or_queries],
                query_denses=[qv],
                limit=limit,
                *args,
                **kwargs,
            )
            # Extract first result from batch
            batch_items = self._batch_results_to_items(results)
            return batch_items[0] if batch_items else []
        else:
            raise ValueError("Input must be a string or a list of strings")

    @overload
    async def aretrieve(
        self,
        query_or_queries: List[str],
        limit: int = 10,
        queries_as_documents: bool = True,
        *args,
        **kwargs,
    ) -> List[List[SearchItem]]: ...

    @overload
    async def aretrieve(
        self,
        query_or_queries: str,
        limit: int = 10,
        queries_as_documents: bool = True,
        *args,
        **kwargs,
    ) -> List[SearchItem]: ...

    async def aretrieve(
        self,
        query_or_queries: str | List[str],
        limit: int = 10,
        queries_as_documents: bool = True,
        *args,
        **kwargs,
    ):
        """Asynchronous retrieval using async embeddings."""
        if isinstance(query_or_queries, list):
            # Embed all queries together first (async)
            if queries_as_documents:
                qvs = await self.embedder.aembed_documents(query_or_queries)
            else:
                qvs = [await self.embedder.aembed_query(q) for q in query_or_queries]
            
            # Process hybrid_search in platform-appropriate batches
            all_results = []
            query_batches = _create_batches(query_or_queries, DEFAULT_BATCH_SIZE)
            dense_batches = _create_batches(qvs, DEFAULT_BATCH_SIZE)
            
            for query_batch, dense_batch in zip(query_batches, dense_batches):
                batch_results = self.store.hybrid_search(
                    query_texts=query_batch,
                    query_denses=dense_batch,
                    limit=limit,
                    *args,
                    **kwargs,
                )
                # Convert batch results and extend to all_results
                batch_items = self._batch_results_to_items(batch_results)
                all_results.extend(batch_items)
            
            return all_results
        elif isinstance(query_or_queries, str):
            qv = await self.embedder.aembed_query(query_or_queries)
            # Single query - wrap in lists for batch processing
            results = self.store.hybrid_search(
                query_texts=[query_or_queries],
                query_denses=[qv],
                limit=limit,
                *args,
                **kwargs,
            )
            # Extract first result from batch
            batch_items = self._batch_results_to_items(results)
            return batch_items[0] if batch_items else []
        else:
            raise ValueError("Input must be a string or a list of strings")
