import functools
import os
from pathlib import Path

import platformdirs
from mcp.server.fastmcp import FastMCP

from mdvault.db import get_connection
from mdvault.retriever import get_total_chunks, hybrid_search
from mdvault.retriever import related_notes as _related_notes

mcp_app = FastMCP("mdvault")


def _resolve_db() -> Path:
    env_db = os.environ.get("VAULT_DB")
    if env_db:
        return Path(env_db)
    return Path(platformdirs.user_data_dir("mdvault")) / "vault.db"


@functools.lru_cache(maxsize=1)
def _get_embedder():
    from model2vec import StaticModel

    model = StaticModel.from_pretrained("minishlab/potion-base-8M")

    def embed(texts: list[str]):
        return model.encode(texts)

    return embed


@mcp_app.tool()
def search_vault(query: str, top_k: int = 5, expand: bool = False) -> dict:
    """Search the local markdown vault using hybrid BM25 + semantic search.
    Set expand=True to use local LLM (Ollama) for query expansion."""
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
    results = hybrid_search(conn, query, embedder, top_k=top_k, expand=expand)
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


def run():
    mcp_app.run()
