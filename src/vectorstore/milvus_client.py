import os
import time
from typing import Optional
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv(override=True)


def get_milvus_client(
    uri: Optional[str] = None,
    token: Optional[str] = None,
    wait_ready: bool = True,
    retries: int = 10,
    backoff_sec: float = 1.0,
) -> MilvusClient:
    """Create a Milvus client using env defaults if not provided and optionally wait for readiness.

    Env overrides:
      - MILVUS_URI (default http://localhost:19530)
      - MILVUS_TOKEN (default root:Milvus)
    """
    uri = uri or os.environ.get("MILVUS_URI", "http://localhost:19530")
    token = token or os.environ.get("MILVUS_TOKEN", "root:Milvus")
    client = MilvusClient(uri=uri, token=token)

    if wait_ready:
        for i in range(max(1, retries)):
            try:
                # A light call to verify connectivity
                client.list_collections()
                break
            except Exception:  # pragma: no cover
                if i == retries - 1:
                    raise
                time.sleep(backoff_sec)
    return client
