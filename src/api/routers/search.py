from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.search import SearchService


router = APIRouter(prefix="/search", tags=["search"])


class ObjectType(str, Enum):
    catalogue = "catalogue"
    dataset = "dataset"
    dataservice = "dataservice"
    resource = "resource"
    vocabulary = "vocabulary"


class SearchFilters(BaseModel):
    types: Optional[List[ObjectType]] = Field(
        default=None,
        description="Restrict results to these object types (e.g. ['dataset']).",
    )


class SearchOptions(BaseModel):
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return.")


class QuerySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query.")
    filters: Optional[SearchFilters] = Field(
        default=None, description="Optional filters."
    )
    options: SearchOptions = Field(
        default_factory=SearchOptions, description="Search options."
    )


class SimilarSearchRequest(BaseModel):
    source_id: str = Field(
        ...,
        min_length=1,
        description="ID of an existing object already in the vectorstore.",
    )
    filters: Optional[SearchFilters] = Field(
        default=None, description="Optional filters."
    )
    options: SearchOptions = Field(
        default_factory=SearchOptions, description="Search options."
    )


class SearchResultItem(BaseModel):
    id: str = Field(
        ..., description="Primary key of the matched object in the vectorstore."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata stored alongside the vector.",
    )


@lru_cache(maxsize=1)
def _get_search_service() -> SearchService:
    return SearchService()


@router.post(
    "",
    summary="Semantic search by query",
    response_model=List[SearchResultItem],
)
async def search_by_query(request: QuerySearchRequest) -> List[SearchResultItem]:
    """Search by natural-language query."""

    types: Optional[List[str]] = None
    if request.filters and request.filters.types:
        types = [t.value for t in request.filters.types]

    try:
        results = _get_search_service().search_by_query(
            request.query,
            top_k=request.options.top_k,
            types=types,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return [
        SearchResultItem(id=str(item.get("id")), metadata=item.get("metadata") or {})
        for item in results
    ]


@router.post(
    "/similar",
    summary="Semantic search for similar items",
    response_model=List[SearchResultItem],
)
async def search_similar(request: SimilarSearchRequest) -> List[SearchResultItem]:
    """Search for items similar to an existing object.

    Implementation will retrieve the source object's text/embedding from the vectorstore and search for nearest neighbors if it exists.
    If the source object does not exist, it will be fetched from the main data store directly.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet")
