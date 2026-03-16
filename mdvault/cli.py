import os
from pathlib import Path

import platformdirs
import typer

from mdvault.db import get_connection, init_db
from mdvault.indexer import index_directory
from mdvault.retriever import get_total_chunks, hybrid_search
from mdvault.retriever import related_notes as _related_notes

app = typer.Typer(add_completion=False)


def _resolve_db(db: str | None = None) -> Path:
    """Resolve DB path: --db flag > VAULT_DB env > platformdirs default."""
    if db:
        return Path(db)
    env_db = os.environ.get("VAULT_DB")
    if env_db:
        return Path(env_db)
    data_dir = Path(platformdirs.user_data_dir("mdvault"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "vault.db"


_MODEL_ID = "minishlab/potion-base-8M"


def _get_embedder():
    """Load the real model2vec embedder."""
    # Suppress HF Hub auth warnings when model is already cached
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{_MODEL_ID.replace('/', '--')}"
    if cache_dir.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(_MODEL_ID)

    def embed(texts: list[str]):
        return model.encode(texts)

    return embed


@app.command()
def index(
    vault_path: str = typer.Argument(..., help="Path to markdown vault directory"),
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    full: bool = typer.Option(False, "--full", help="Force full re-index of this vault (default: additive)"),
):
    """Index a directory of markdown files. Additive by default — multiple vaults can share one DB."""
    vault_root = Path(vault_path).resolve()
    db_path = _resolve_db(db)
    embedder = _get_embedder()

    init_db(db_path)
    conn = get_connection(db_path)

    index_directory(conn, vault_root, embedder, full=full)
    conn.commit()

    mode = "full re-index" if full else "indexed"
    typer.echo(f"{vault_root.name}: {mode}")

    file_count = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    chunk_count = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
    typer.echo(f"Files: {file_count}, Chunks: {chunk_count}")
    conn.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    top_k: int = typer.Option(5, "--top-k", help="Number of results to return"),
    expand: bool = typer.Option(False, "--expand", help="Expand query via local LLM (requires Ollama)"),
    expand_model: str = typer.Option("qwen3:0.6b", "--expand-model", help="Ollama model for query expansion"),
):
    """Search the vault using hybrid BM25 + vector search."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo("Error: Database not found. Run 'mdvault index' first.")
        raise typer.Exit(1)

    embedder = _get_embedder()
    conn = get_connection(db_path)
    total = get_total_chunks(conn)
    results = hybrid_search(conn, query, embedder, top_k=top_k, expand=expand, expand_model=expand_model)
    conn.close()

    typer.echo(f"Query: {query}")
    typer.echo(f"Chunks searched: {total}")
    typer.echo("")

    for i, r in enumerate(results, start=1):
        typer.echo(f"[{i}] {r['file_path']} (chunk {r['chunk_idx']}) — score {r['score']:.3f}")
        typer.echo("─" * 42)
        content_preview = r["raw_content"][:500]
        typer.echo(content_preview)
        typer.echo("─" * 42)
        typer.echo("")


@app.command()
def stats(
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
):
    """Show index statistics."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo("Error: Database not found. Run 'mdvault index' first.")
        raise typer.Exit(1)

    conn = get_connection(db_path)
    vault_roots = conn.execute(
        "SELECT key, value FROM vault_config WHERE key LIKE 'vault_root:%' ORDER BY key"
    ).fetchall()
    file_count = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    chunk_count = get_total_chunks(conn)
    conn.close()

    db_size = db_path.stat().st_size
    if db_size > 1024 * 1024:
        size_str = f"{db_size / (1024 * 1024):.0f} MB"
    else:
        size_str = f"{db_size / 1024:.0f} KB"

    if vault_roots:
        for vr in vault_roots:
            name = vr["key"].removeprefix("vault_root:")
            typer.echo(f"Vault         : {name} → {vr['value']}")
    else:
        typer.echo("Vault         : (none)")
    typer.echo(f"DB path       : {db_path}")
    typer.echo(f"Files indexed : {file_count:,}")
    typer.echo(f"Total chunks  : {chunk_count:,}")
    typer.echo(f"DB size       : {size_str}")


@app.command()
def related(
    file_path: str = typer.Argument(..., help="Relative path of the file in the vault"),
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    top_k: int = typer.Option(5, "--top-k", help="Number of similar files to return"),
):
    """Show related notes: links, backlinks, and semantically similar files."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo("Error: Database not found. Run 'mdvault index' first.")
        raise typer.Exit(1)

    embedder = _get_embedder()
    conn = get_connection(db_path)
    result = _related_notes(conn, file_path, embedder, top_k=top_k)
    conn.close()

    typer.echo(f"Related notes for: {result['file_path']}")
    typer.echo("")

    if result["links"]:
        typer.echo("Links →")
        for link in result["links"]:
            typer.echo(f"  {link}")
    else:
        typer.echo("Links → (none)")

    typer.echo("")

    if result["backlinks"]:
        typer.echo("Backlinks ←")
        for bl in result["backlinks"]:
            typer.echo(f"  {bl}")
    else:
        typer.echo("Backlinks ← (none)")

    typer.echo("")

    if result["similar"]:
        typer.echo("Similar (vector)")
        for s in result["similar"]:
            typer.echo(f"  {s}")
    else:
        typer.echo("Similar → (none)")


@app.command()
def serve(
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
):
    """Start the MCP server for Claude Code integration."""
    db_path = _resolve_db(db)
    # Set env so mcp_server can find the DB
    os.environ["VAULT_DB"] = str(db_path)
    from mdvault.mcp_server import run

    run()
