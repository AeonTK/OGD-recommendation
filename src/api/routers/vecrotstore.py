from __future__ import annotations


from fastapi import APIRouter, HTTPException
from pydantic import Field, BaseModel
import os
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection

load_dotenv(override=True)
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")


def ensure_connection():
    if not connections.has_connection("default"):
        connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)


router = APIRouter(prefix="/vectorstore", tags=["vectorstore"])


class CollectionInfo(BaseModel):
    name: str = Field(..., description="Collection name")
    state: str = Field(..., description="Collection state (LOADED or DISK)")
    state_code: int = Field(..., description="Numeric collection state code (Milvus)")
    n_entities: int | None = Field(
        None, description="Number of entities in the collection, if available"
    )
    aliases: list[str] = Field(
        default_factory=list, description="Aliases for the collection"
    )


class ListCollectionsResponse(BaseModel):
    collections: list[CollectionInfo] = Field(
        ..., description="A list of collections with their runtime status and metadata"
    )


class DropCollectionRequest(BaseModel):
    collection_name: str = Field(
        ..., description="The name of the collection to delete."
    )


class LoadCollectionRequest(BaseModel):
    collection_name: str = Field(
        ..., description="The name of the collection to load into memory."
    )


class ReleaseCollectionRequest(BaseModel):
    collection_name: str = Field(
        ..., description="The name of the collection to release from memory."
    )


class SchemaCollectionRequest(BaseModel):
    collection_name: str = Field(..., description="The name of the collection.")


class SchemaCollectionResponse(BaseModel):
    collection_schema: dict = Field(
        ..., description="The schema of the specified collection."
    )


class AliasCollectionRequest(BaseModel):
    alias_name: str = Field(
        ..., description="The alias name to assign to the collection."
    )


class RenameCollectionRequest(BaseModel):
    new_name: str = Field(..., description="The new name to rename the collection to.")


@router.get(
    "/collections",
    summary="List vector store collections",
    response_model=ListCollectionsResponse,
)
async def list_collections() -> ListCollectionsResponse:
    ensure_connection()
    try:
        collections = utility.list_collections()
        result = []
        for name in collections:
            try:
                # 3=Loaded, 2=Loading, 1=NotLoad, 0=NotExist
                state_code = utility.load_state(name)
                state_str = "LOADED" if state_code == 3 else "DISK"
                c = Collection(name)
                n_entities = c.num_entities
                aliases = list(utility.list_aliases(name))
            except Exception:
                state_code = -1
                state_str = "UNKNOWN"
                n_entities = None
                aliases = []
            info = CollectionInfo(
                name=name,
                state=state_str,
                state_code=state_code,
                n_entities=n_entities,
                aliases=aliases,
            )
            result.append(info)
        return ListCollectionsResponse(collections=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")


@router.post(
    "/collections/{collection_name}/load",
    summary="Load a vector store collection into memory",
)
async def load_collection(collection_name: str):
    ensure_connection()
    try:
        Collection(collection_name).load()
        return {"status": "loaded", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load collection: {e}")


@router.post(
    "/collections/{collection_name}/release",
    summary="Release a vector store collection from memory",
)
async def release_collection(collection_name: str):
    ensure_connection()
    try:
        Collection(collection_name).release()
        return {"status": "released", "collection": collection_name}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to release collection: {e}"
        )


@router.delete(
    "/collections/{collection_name}", summary="Delete a vector store collection"
)
async def drop_collection(collection_name: str):
    ensure_connection()
    try:
        utility.drop_collection(collection_name)
        return {"status": "dropped", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to drop collection: {e}")


@router.post(
    "/collections/{collection_name}/alias",
    summary="Create an alias for a vector store collection",
)
async def alias_collection(collection_name: str, request: AliasCollectionRequest):
    ensure_connection()
    try:
        utility.create_alias(collection_name, request.alias_name)
        aliases = utility.list_aliases(collection_name)
        return {
            "status": "aliased",
            "collection": collection_name,
            "aliases": aliases,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create alias: {e}")


@router.delete(
    "/collections/{collection_name}/aliases/{alias_name}",
    summary="Drop an alias for a vector store collection",
)
async def drop_alias(collection_name: str, alias_name: str):
    ensure_connection()
    try:
        # validate alias belongs to collection
        aliases = list(utility.list_aliases(collection_name))
        if alias_name not in aliases:
            raise HTTPException(
                status_code=404,
                detail=f"Alias '{alias_name}' not found for collection '{collection_name}'",
            )
        utility.drop_alias(alias_name)
        aliases = list(utility.list_aliases(collection_name))
        return {
            "status": "alias_dropped",
            "collection": collection_name,
            "aliases": aliases,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to drop alias: {e}")


@router.post(
    "/collections/{collection_name}/rename",
    summary="Rename a vector store collection",
)
async def rename_collection(collection_name: str, request: RenameCollectionRequest):
    ensure_connection()
    try:
        utility.rename_collection(collection_name, request.new_name)
        return {"status": "renamed", "collection": request.new_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename collection: {e}")


@router.get(
    "/collections/{collection_name}/schema",
    summary="Get the schema of a vector store collection",
    response_model=SchemaCollectionResponse,
)
async def get_collection_schema(collection_name: str) -> SchemaCollectionResponse:
    ensure_connection()
    try:
        c = Collection(collection_name)
        schema = c.schema.to_dict()
        return SchemaCollectionResponse(collection_schema=schema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {e}")
