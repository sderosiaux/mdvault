import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import platformdirs

from mdvault.db import get_connection
from mdvault.retriever import hybrid_search, get_total_chunks

mcp_app = FastMCP("mdvault")


def _resolve_db() -> Path:
    env_db = os.environ.get("VAULT_DB")
    if env_db:
        return Path(env_db)
    return Path(platformdirs.user_data_dir("mdvault")) / "vault.db"


def _get_embedder():
    from model2vec import StaticModel

    model = StaticModel.from_pretrained("minishlab/potion-base-8M")

    def embed(texts: list[str]):
        return model.encode(texts)

    return embed


@mcp_app.tool()
def search_vault(query: str, top_k: int = 5) -> dict:
    """Search the local markdown vault using hybrid BM25 + semantic search."""
    db_path = _resolve_db()
    embedder = _get_embedder()
    conn = get_connection(db_path)
    total = get_total_chunks(conn)
    results = hybrid_search(conn, query, embedder, top_k=top_k)
    conn.close()

    return {
        "results": [
            {
                "file_path": r["file_path"],
                "chunk_idx": r["chunk_idx"],
                "content": r["content"],
                "score": round(r["score"], 4),
            }
            for r in results
        ],
        "query": query,
        "total_chunks": total,
    }


def run():
    mcp_app.run()
