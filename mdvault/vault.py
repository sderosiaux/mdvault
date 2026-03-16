from collections.abc import Callable
from pathlib import Path

import numpy as np

from mdvault.db import get_connection, init_db
from mdvault.indexer import index_directory
from mdvault.memory import delete_memory as _delete_memory
from mdvault.memory import store_memory as _store_memory
from mdvault.memory import update_memory as _update_memory
from mdvault.retriever import hybrid_search


class Vault:
    """Facade for mdvault: memory store + file indexer + hybrid search."""

    def __init__(self, db_path: str | Path, embedder: Callable[[list[str]], np.ndarray] | None = None):
        self._db_path = Path(db_path)
        self._embedder = embedder
        init_db(self._db_path)
        self._conn = get_connection(self._db_path)

    def _get_embedder(self) -> Callable[[list[str]], np.ndarray]:
        if self._embedder is None:
            import os

            model_id = "minishlab/potion-base-8M"
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
            if cache_dir.exists():
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from model2vec import StaticModel

            model = StaticModel.from_pretrained(model_id)
            self._embedder = model.encode
        return self._embedder

    def store(
        self,
        content: str,
        namespace: str = "",
        source: str = "api",
        metadata: dict | None = None,
    ) -> dict:
        """Store a memory. Auto-chunks if content > 400 words."""
        result = _store_memory(self._conn, content, self._get_embedder(), namespace, source, metadata)
        self._conn.commit()
        return result

    def update(self, id: str, content: str | None = None, metadata: dict | None = None) -> None:
        """Update a memory's content and/or metadata."""
        _update_memory(self._conn, id, self._get_embedder(), content, metadata)
        self._conn.commit()

    def delete(self, id: str | None = None, namespace: str | None = None) -> int:
        """Delete memories by id or namespace. Returns count deleted."""
        count = _delete_memory(self._conn, id, namespace)
        self._conn.commit()
        return count

    def search(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
        namespace: str | None = None,
        expand: bool = False,
        expand_model: str = "qwen3:0.6b",
    ) -> list[dict]:
        """Hybrid search across files and memories."""
        return hybrid_search(
            self._conn,
            query,
            self._get_embedder(),
            top_k=top_k,
            source=source,
            namespace=namespace,
            expand=expand,
            expand_model=expand_model,
        )

    def index(self, vault_path: str | Path, full: bool = False) -> None:
        """Index a directory of markdown files."""
        vault_root = Path(vault_path).resolve()
        index_directory(self._conn, vault_root, self._get_embedder(), full=full)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
