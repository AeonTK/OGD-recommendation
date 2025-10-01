from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID
from typing import Optional


@dataclass
class SearchItem:
    id: UUID
    text: str
    distance: Optional[float] = None

    def snippet(self, max_len: int = 160) -> str:
        if len(self.text) <= max_len:
            return self.text
        return self.text[:max_len] + "..."
