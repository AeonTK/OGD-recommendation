from typing import Iterable, List, Optional, Dict, Any
from pathlib import Path
from langchain.embeddings.base import init_embeddings
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from functools import cached_property
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict
import yaml


# The function now requires a path. No more magic defaults.
def load_config(config_path: Path | str) -> Dict[str, Any]:
    """Load configuration from a given YAML file path.

    Args:
        config_path: Explicit path to the configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If there is an error parsing the YAML file.
    """
    path = Path(config_path)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {path}") from e

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{path}': {e}") from e

    return data or {}


CONFIG_FILE_PATH = Path(__file__).parent / "config.yaml"

logger = logging.getLogger(__name__)


class Embedder:
    """Generic embedding wrapper backed by LangChain's init_embeddings.

    Credentials are read from environment as required by the chosen provider
    (e.g., COHERE_API_KEY, OPENAI_API_KEY, etc.).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        provider: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Start from explicit config dict if provided, else load from vectorstore/config.yaml
        if config is not None:
            cfg: Dict[str, Any] = dict(config)
        else:
            try:
                cfg = load_config(CONFIG_FILE_PATH)
            except FileNotFoundError:
                cfg = {}

        embedding_cfg = cfg.get("embedding_model", {}) if isinstance(cfg, dict) else {}

        # Override with explicit args if provided
        if provider is not None:
            embedding_cfg["provider"] = provider
        if model is not None:
            embedding_cfg["model"] = model

        # Validate final config
        if not embedding_cfg.get("provider") or not embedding_cfg.get("model"):
            raise ValueError(
                "Embedding configuration missing 'provider' and/or 'model'. "
                "Set them in src/vectorstore/config.yaml under 'embedding_model', or pass them to Embedder()."
            )
        self._cfg = cfg
        if self._cfg.get("dim"):
            logger.info(
                "Using configured embedding dimension: %s", self._cfg.get("dim")
            )
            self.__dict__["dim"] = self._cfg.get("dim")

        provider_name = embedding_cfg.get("provider", "").lower()
        if provider_name == "spacy":
            model_name = embedding_cfg.get("model", "en_core_web_lg")
            logger.info("Initializing spaCy embeddings with model '%s'", model_name)
            self._emb = SpacyEmbeddings(model_name=model_name)
        else:
            logger.info(
                "Initializing embeddings via init_embeddings provider=%s model=%s", 
                embedding_cfg.get("provider"), 
                embedding_cfg.get("model"),
            )
            self._emb = init_embeddings(**embedding_cfg)

    def _cache_dim(self, new_dim: int, source: str) -> None:
        """Cache embedding dimension once; warn on mismatches across calls."""
        cached = self.__dict__.get("dim")
        if cached is None:
            # Seed cached_property storage to avoid separate probe later
            self.__dict__["dim"] = new_dim
            logger.info(f"Cached embedding dimension from {source}: {new_dim}")
        elif cached != new_dim:
            logger.warning(
                f"Embedding dimension mismatch detected: cached={cached}, new={new_dim}."
            )

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        # LangChain embeddings implement embed_documents for batch embedding
        if not hasattr(self._emb, "embed_documents"):
            raise NotImplementedError(
                "The embedding provider does not implement 'embed_documents'. "
                "Please choose a provider/model that supports batch embedding."
            )
        vecs = self._emb.embed_documents(texts)  # type: ignore[attr-defined]
        # Opportunistically cache dimension if not already set
        if vecs and vecs[0]:
            self._cache_dim(len(vecs[0]), "embed_documents()")
        return vecs

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string synchronously.

        Uses provider embed_query when available; otherwise falls back to batch embed().
        Caches dimension and warns on mismatch similarly to embed().
        """
        emb = self._emb
        if hasattr(emb, "embed_query"):
            vec: List[float] = emb.embed_query(text)  # type: ignore[attr-defined]
        else:
            res = self.embed_documents([text])
            vec = res[0] if res else []

        if vec:
            self._cache_dim(len(vec), "embed_query()")
        return vec

    async def aembed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        """Async variant of embed(); uses provider aembed_documents if available.

        Falls back to running sync methods in a worker thread to avoid blocking the event loop.
        """
        items = list(texts)
        if not items:
            return []

        emb = self._emb
        vecs: List[List[float]]

        if hasattr(emb, "aembed_documents"):
            vecs = await emb.aembed_documents(items)  # type: ignore[attr-defined]
        else:
            if hasattr(emb, "embed_documents"):
                logger.debug("Using sync embed_documents in async aembed_documents()")
                vecs = await asyncio.to_thread(emb.embed_documents, items)  # type: ignore[attr-defined]
            else:
                raise NotImplementedError(
                    "The embedding provider does not implement 'embed_documents' or 'aembed_documents'. "
                    "Please choose a provider/model that supports batch embedding."
                )

        # Opportunistically cache dimension if not already set
        if vecs and vecs[0]:
            self._cache_dim(len(vecs[0]), "aembed_documents()")
        return vecs

    async def aembed_query(self, text: str) -> List[float]:
        """Async single-text embedding; prefers provider aembed_query if available."""
        emb = self._emb

        if hasattr(emb, "aembed_query"):
            vec: List[float] = await emb.aembed_query(text)  # type: ignore[attr-defined]
        elif hasattr(emb, "embed_query"):
            logger.debug("Using sync embed_query in async aembed_query()")
            vec = await asyncio.to_thread(emb.embed_query, text)  # type: ignore[attr-defined]
        else:
            res = await self.aembed_documents([text])
            vec = res[0] if res else []

        # Cache dimension if not already set
        if vec:
            self._cache_dim(len(vec), "aembed_query()")
        return vec

    @cached_property
    def dim(self) -> int:
        """Embedding vector dimension, computed once by probing the model.

        Falls back to embed_query if embed_documents isn't available.
        """
        # 0) Prefer an explicitly configured dimension when provided
        cfg = getattr(self, "_cfg", {})
        if isinstance(cfg, dict):
            cfg_dim = cfg.get("dim") or cfg.get("dimensions")
            # Coerce to int if it's a string/float-like value
            if isinstance(cfg_dim, (int,)) and cfg_dim > 0:
                logger.info(f"Using configured embedding dimension: {cfg_dim}")
                return int(cfg_dim)
            if isinstance(cfg_dim, (str, float)):
                try:
                    val = int(float(cfg_dim))
                    if val > 0:
                        logger.info(f"Using configured embedding dimension: {val}")
                        return val
                except Exception:
                    pass

        # Prefer batch API to match runtime usage
        try:
            vecs = self._emb.embed_documents(["Hello World!"])  # type: ignore[attr-defined]
            if not vecs or not vecs[0]:
                raise RuntimeError(
                    "Embedding model returned empty vector when probing dimension."
                )
            logger.info(f"Probed embedding dimension: {len(vecs[0])}")
            return len(vecs[0])
        except AttributeError:
            # Some LC embeddings expose only embed_query
            if hasattr(self._emb, "embed_query"):
                vec = self._emb.embed_query("Hello World!")  # type: ignore[attr-defined]
                if not vec:
                    raise RuntimeError(
                        "Embedding model returned empty vector when probing dimension."
                    )
                return len(vec)
            raise
