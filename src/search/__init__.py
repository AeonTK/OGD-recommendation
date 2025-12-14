"""Search application layer.

This package provides small, reusable functions/classes that implement the two
search scenarios used by the API:
- Search by free-text query
- Search for items similar to an existing stored item (by id)

Implementation relies on the existing vectorstore module; it does not modify it.
"""

from .service import SearchService, SearchServiceConfig

__all__ = ["SearchService", "SearchServiceConfig"]
