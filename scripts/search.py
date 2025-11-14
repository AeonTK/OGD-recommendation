"""Hybrid search script using Retriever abstraction.

Configuration via constants below (no CLI args). Run:
	uv run python scripts/search.py

Environment:
	COHERE_API_KEY  (embedding)
	MILVUS_URI      (default http://localhost:19530)
	MILVUS_TOKEN    (default root:Milvus)
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
import sys
from typing import List

# Ensure 'src' on path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from src.vectorstore.embeddings import Embedder  # type: ignore  # noqa: E402
from src.vectorstore.data_store import DataVectorStore  # type: ignore  # noqa: E402
from src.vectorstore.retriever import Retriever  # type: ignore  # noqa: E402
from src.vectorstore.schemas import SearchItem, format_search_item  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "programs"
QUERY_TEXT: str = "Gardens in Poland"
TOP_K: int = 5
LOG_LEVEL: str = "INFO"


def search(query: str, top_k: int = TOP_K) -> List[SearchItem]:
	"""Retrieve items from the vector store using Retriever.

	Returns a list of SearchItem and logs an aggregated multi-line block with results.
	"""
	logger = logging.getLogger(__name__)

	# Share the same embedder between retriever and store to ensure matching dims.
	embedder = Embedder()
	store = DataVectorStore(collection=COLLECTION_NAME, embedder=embedder)
	retriever = Retriever(store=store, embedder=embedder)

	items = retriever.retrieve(query, limit=top_k)
	header = f"Returned K={top_k} results. \nQuery: {query!r} \n"
	lines: List[str] = [header]
	for idx, item in enumerate(items, start=1):
		lines.append(f"{idx}. {format_search_item(item)}")
		# Also show metadata if present (prefer a concise dataset URL if available)
		if item.metadata:
			dataset = item.metadata.get("dataset") if isinstance(item.metadata, dict) else None
			if dataset:
				lines.append(f"    metadata.dataset: {dataset}")
			else:
				try:
					meta_str = json.dumps(item.metadata, ensure_ascii=False)
					lines.append(f"    metadata: {meta_str}")
				except Exception:
					lines.append("    metadata: <unserializable>")
	logger.info("\n".join(lines))
	return items


def main() -> int:
	logging.basicConfig(
		level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
		format="%(asctime)s %(levelname)s %(name)s - %(message)s",
	)
	try:
		search(QUERY_TEXT, TOP_K)
		return 0
	except Exception as e:  # pragma: no cover
		logging.exception("Search failed: %s", e)
		return 1

if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())
