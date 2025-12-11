import logging
from typing import List, Optional

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from .milvus_client import get_milvus_client
from .embeddings import Embedder


logger = logging.getLogger(__name__)


class CollectionManager:
    """Manages Milvus collection lifecycle and schema for program vectors."""

    def __init__(
        self, client: MilvusClient, name: str = "programs", *, embedder: Embedder
    ) -> None:
        self.client = client
        self.name = name
        self.embedder = embedder

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.name):
            return

        schema = MilvusClient.create_schema(auto_id=False)
        # Use string UUIDs (36 chars with dashes) as primary key
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, max_length=256, is_primary=True
        )
        # Optional metadata stored as JSON (not analyzed/embedded)
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=8000,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="text_dense",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedder.dim,
        )
        schema.add_field(
            field_name="text_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        bm25_fn = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["text_sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_fn)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="text_dense",
            index_name="text_dense_index",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="text_sparse",
            index_name="text_sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

        self.client.create_collection(
            collection_name=self.name, schema=schema, index_params=index_params
        )

    def drop_collection(self) -> None:
        if self.client.has_collection(self.name):
            logger.info("Dropping collection '%s'.", self.name)
            self.client.drop_collection(collection_name=self.name)
            logger.info("Collection dropped successfully.")

    def reset_collection(self) -> None:
        self.drop_collection()
        self.ensure_collection()


class DataVectorStore:
    """CRUD and search operations; delegates lifecycle to a CollectionManager."""

    def __init__(
        self,
        client: Optional[MilvusClient] = None,
        collection: str = "programs",
        manager: Optional[CollectionManager] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.client = client or get_milvus_client()
        self.collection = collection
        self.embedder = embedder or Embedder()
        self.manager = manager or CollectionManager(
            self.client, name=self.collection, embedder=self.embedder
        )
        self.manager.ensure_collection()

    def reset(self) -> None:
        self.manager.reset_collection()

    def upsert(
        self,
        ids: List[str],
        texts: List[str],
        dense_vectors: List[List[float]],
        *,
        metadatas: Optional[List[Optional[dict]]] = None,
    ) -> None:
        if not (len(ids) == len(texts) == len(dense_vectors)):
            raise ValueError("ids, texts, dense_vectors must be the same length")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas (if provided) must be the same length as ids")
        # Validate vector dimensionality
        expected_dim = self.embedder.dim
        for i, v in enumerate(dense_vectors):
            if len(v) != expected_dim:
                raise ValueError(
                    f"dense_vectors[{i}] has dim {len(v)} but collection expects {expected_dim}"
                )
        base_rows = [
            {"id": ids[i], "text": texts[i], "text_dense": dense_vectors[i]}
            for i in range(len(ids))
        ]

        # Prefer metadata JSON; finally base only
        attempts: List[List[dict]] = []
        if metadatas is not None:
            rows = []
            for i, row in enumerate(base_rows):
                new_row = dict(row)
                new_row["metadata"] = metadatas[i] or {}
                rows.append(new_row)
            attempts.append(rows)
        attempts.append(base_rows)

        last_err: Optional[Exception] = None
        for payload in attempts:
            try:
                self.client.upsert(collection_name=self.collection, data=payload)
                return
            except Exception as e:  # pragma: no cover - depends on server schema
                last_err = e
                continue
        # If all attempts failed, raise the last error
        if last_err:
            raise last_err

    def hybrid_search(self, query_text: str, query_dense: List[float], limit: int = 5, **kwargs):
        # Validate query vector dimensionality
        if len(query_dense) != self.embedder.dim:
            raise ValueError(
                f"query_dense has dim {len(query_dense)} but collection expects {self.embedder.dim}"
            )
        
        filter_expression = kwargs.pop("filter_expression", None)

        search_params = None
        if filter_expression is not None:
            search_params = {"hints": "iterative_filter"}


        # Dense ANN on text_dense
        req_dense = AnnSearchRequest(
            data=[query_dense],
            anns_field="text_dense",
            param={"nprobe": 10},
            limit=int(min(16384, max(512, limit * 10))),
            expr=filter_expression,
        )
        # Sparse BM25 on text -> text_sparse
        req_sparse = AnnSearchRequest(
            data=[query_text],
            anns_field="text_sparse",
            param={"drop_ratio_search": 0.2},
            limit=int(min(16384, max(512, limit * 10))),
            expr=filter_expression,
        )
        ranker = RRFRanker(60)
        # Prefer returning metadata JSON if the field exists; gracefully fall back
        output_fields = ["id", "text", "text_dense", "metadata"]
        try:
            return self.client.hybrid_search(
                collection_name=self.collection,
                reqs=[req_dense, req_sparse],
                ranker=ranker,
                limit=limit,
                output_fields=output_fields,
                search_params=search_params,
            )
        except Exception:
            # Older collection without 'metadata'; retry without it
            return self.client.hybrid_search(
                collection_name=self.collection,
                reqs=[req_dense, req_sparse],
                ranker=ranker,
                limit=limit,
                output_fields=["id", "text", "text_dense"],
                search_params=search_params,
            )
