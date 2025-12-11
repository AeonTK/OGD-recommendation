from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from .routers.vecrotstore import router as vectorstore_router


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # API will be slim and delegate runner work to search.conformance
    yield


"""
FastAPI application

Note on OpenAPI/Swagger docs:
Some recent combinations of FastAPI/Starlette serve the OpenAPI schema with
the vendor media type "application/vnd.oai.openapi+json". In certain client
environments (or with strict Accept headers), this can cause a 406 Not
Acceptable when the Swagger UI tries to fetch /openapi.json.

To avoid that, we disable the auto-registered OpenAPI/docs routes and add
explicit JSONResponse-based endpoints for the schema and Swagger UI.
"""

# Detect optional base path for deployments under a subpath (e.g., 
# https://example.com/conformance-checking-demo/...). If set, this will be
# used as the ASGI root_path and advertised via OpenAPI "servers" so that
# Swagger UI "Try it out" sends requests to the correct prefixed URLs.
_env_base_path = os.getenv("API_BASE_PATH", "").strip()
if _env_base_path and not _env_base_path.startswith("/"):
    _env_base_path = "/" + _env_base_path
if _env_base_path.endswith("/") and _env_base_path != "/":
    _env_base_path = _env_base_path.rstrip("/")

# Disable built-in docs/openapi routes; we'll provide explicit JSON-based ones
app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    root_path=_env_base_path or "",
)


# CORS: allow browser apps hosted on other origins to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount routers
app.include_router(vectorstore_router)


@app.get("/health", tags=["ops"], summary="Health check")
async def health():
    return {"status": "ok"}


def _with_servers(base_path: str | None):
    """Return OpenAPI schema optionally annotated with servers -> [{url: base_path}].

    This ensures Swagger UI "Try it out" respects a deployment subpath.
    """
    schema = app.openapi()
    # Only add servers if we actually have a non-root base path
    if base_path and base_path != "/":
        # Copy-on-write to avoid mutating a cached schema in place across requests
        # (FastAPI caches app.openapi())
        schema = {**schema, "servers": [{"url": base_path}]}
    return schema


# Explicit OpenAPI JSON (forces application/json, avoids 406 with strict Accept)
@app.get("/openapi.json", include_in_schema=False)
def openapi_json():
    return JSONResponse(_with_servers(_env_base_path or None))


# Swagger UI served manually and pointed at our JSON OpenAPI endpoint
# Use a relative URL for openapi_url so it works when the app is served under a
# subpath (e.g., /conformance-checking-demo) behind a reverse proxy.
@app.get("/docs", include_in_schema=False)
def swagger_ui():
    return get_swagger_ui_html(openapi_url="openapi.json", title="API Docs")
