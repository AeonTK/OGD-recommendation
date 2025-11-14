from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID
from typing import Optional, Dict, Any


@dataclass
class SearchItem:
    id: UUID
    text: str
    distance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def snippet(self, max_len: int = 160) -> str:
        """Return a centered snippet: beginning + ... + ending, length <= max_len.

        If text is shorter than or equal to max_len, returns full text.
        If max_len <= 3, returns leading max_len characters (no ellipsis logic).
        Otherwise, uses '...' as the separator and splits remaining budget
        between head and tail.
        """
        s = self.text or ""
        if len(s) <= max_len:
            return s
        if max_len <= 3:
            return s[:max_len]
        budget = max_len - 3
        head_len = budget // 2
        tail_len = budget - head_len
        return f"{s[:head_len]}...{s[-tail_len:]}"


def format_search_item(item: "SearchItem", max_len: int = 160) -> str:
    """Create a compact string representation for logs/printing.

    Example: "id=<uuid> dist=0.1234 text=<snippet>"
    """
    dist_str = f"{item.distance:.4f}" if item.distance is not None else "?"
    return f"id={item.id}; dist={dist_str}; text={item.snippet(max_len)}"
