import functools
import os
from pathlib import Path

import platformdirs
from mcp.server.fastmcp import FastMCP

from mdvault.db import get_connection, init_db
from mdvault.memory import delete_memory as _delete_memory
from mdvault.memory import store_memory as _store_memory
from mdvault.memory import update_memory as _update_memory
from mdvault.retriever import get_total_chunks, hybrid_search
from mdvault.retriever import related_notes as _related_notes

mcp_app = FastMCP("mdvault")


def _resolve_db() -> Path:
    env_db = os.environ.get("VAULT_DB")
    if env_db:
        return Path(env_db)
    return Path(platformdirs.user_data_dir("mdvault")) / "vault.db"


_MODEL_ID = "minishlab/potion-base-8M"


@functools.lru_cache(maxsize=1)
def _get_embedder():
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{_MODEL_ID.replace('/', '--')}"
    if cache_dir.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(_MODEL_ID)

    def embed(texts: list[str]):
        return model.encode(texts)

    return embed


@mcp_app.tool()
def search_vault(
    query: str,
    top_k: int = 5,
    expand: bool = False,
    source: str | None = None,
    namespace: str | None = None,
    vault: str | None = None,
) -> dict:
    """Search the local markdown vault using hybrid BM25 + semantic search.
    Set expand=True to use local LLM (Ollama) for query expansion.
    Filter by source ('files' or 'memories'), namespace, or vault name."""
    db_path = _resolve_db()
    if not db_path.exists():
        return {
            "error": "Database not found. Run 'mdvault index' first.",
            "results": [],
            "query": query,
            "total_chunks": 0,
        }
    embedder = _get_embedder()
    conn = get_connection(db_path)
    total = get_total_chunks(conn)
    results = hybrid_search(conn, query, embedder, top_k=top_k, expand=expand, source=source, namespace=namespace)
    if vault:
        prefix = f"{vault}/"
        results = [r for r in results if r["file_path"].startswith(prefix)]
    conn.close()

    return {
        "results": [
            {
                "file_path": r["file_path"],
                "chunk_idx": r["chunk_idx"],
                "content": r["raw_content"],
                "score": round(r["score"], 4),
            }
            for r in results
        ],
        "query": query,
        "total_chunks": total,
    }


@mcp_app.tool()
def related_notes(file_path: str, top_k: int = 5) -> dict:
    """Find related notes: direct links, backlinks, and semantically similar files."""
    db_path = _resolve_db()
    if not db_path.exists():
        return {
            "error": "Database not found. Run 'mdvault index' first.",
            "file_path": file_path,
            "links": [],
            "backlinks": [],
            "similar": [],
        }
    embedder = _get_embedder()
    conn = get_connection(db_path)
    result = _related_notes(conn, file_path, embedder, top_k=top_k)
    conn.close()
    return result


@mcp_app.tool()
def store_memory(
    content: str,
    namespace: str = "",
    source: str = "agent",
    metadata: dict | None = None,
) -> dict:
    """Store a memory in the vault. Auto-chunks long content. Searchable via search_vault."""
    db_path = _resolve_db()
    from mdvault.db import init_db

    init_db(db_path)
    embedder = _get_embedder()
    conn = get_connection(db_path)
    result = _store_memory(conn, content, embedder, namespace, source, metadata)
    conn.commit()
    conn.close()
    return result


@mcp_app.tool()
def update_memory(id: str, content: str | None = None, metadata: dict | None = None) -> dict:
    """Update an existing memory's content and/or metadata."""
    db_path = _resolve_db()
    if not db_path.exists():
        return {"error": "Database not found.", "id": id, "updated": False}
    embedder = _get_embedder()
    conn = get_connection(db_path)
    _update_memory(conn, id, embedder, content, metadata)
    conn.commit()
    conn.close()
    return {"id": id, "updated": True}


@mcp_app.tool()
def delete_memory(id: str | None = None, namespace: str | None = None) -> dict:
    """Delete memories by id or namespace."""
    db_path = _resolve_db()
    if not db_path.exists():
        return {"error": "Database not found.", "deleted": 0}
    conn = get_connection(db_path)
    count = _delete_memory(conn, id, namespace)
    conn.commit()
    conn.close()
    return {"deleted": count}


def run():
    db_path = _resolve_db()
    if db_path.exists():
        init_db(db_path)  # apply migrations for older DBs
    mcp_app.run()
