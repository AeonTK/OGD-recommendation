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
from pathlib import Path
import sys
from typing import Sequence

# Ensure 'src' on path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from vectorstore.retriever import Retriever  # type: ignore  # noqa: E402
from vectorstore.schemas import SearchItem  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "programs"
QUERY_TEXT: str = "earth’s surface and all objects on it in the form of dots arranged in a regular lattice and georeferenced in position and height. In contrast to the digital terrain model (DGM), the vegetation and development surface is represented"
TOP_K: int = 5
LOG_LEVEL: str = "INFO"


def run_query(query: str, top_k: int) -> Sequence[SearchItem]:
	logger = logging.getLogger(__name__)
	retriever = Retriever()
	items = retriever.retrieve(query, limit=top_k)
	header = f"Top {top_k} results for query={query!r} (returned={len(items)})"
	lines: list[str] = [header]
	for idx, item in enumerate(items, start=1):
		dist_str = f"{item.distance:.4f}" if item.distance is not None else "?"
		lines.append(f"{idx}. dist={dist_str} id={item.id} text={item.snippet()}")
	logger.info("\n".join(lines))
	return items


def main() -> int:
	logging.basicConfig(
		level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
		format="%(asctime)s %(levelname)s %(name)s - %(message)s",
	)
	try:
		run_query(QUERY_TEXT, TOP_K)
		return 0
	except Exception as e:  # pragma: no cover
		logging.exception("Search failed: %s", e)
		return 1


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())
