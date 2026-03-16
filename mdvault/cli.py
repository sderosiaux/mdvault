import os
from pathlib import Path
from typing import Optional

import typer
import platformdirs

from mdvault.db import get_connection, init_db
from mdvault.indexer import index_directory, incremental_index
from mdvault.retriever import hybrid_search, get_total_chunks, related_notes as _related_notes

app = typer.Typer(add_completion=False)


def _resolve_db(db: Optional[str] = None) -> Path:
    """Resolve DB path: --db flag > VAULT_DB env > platformdirs default."""
    if db:
        return Path(db)
    env_db = os.environ.get("VAULT_DB")
    if env_db:
        return Path(env_db)
    data_dir = Path(platformdirs.user_data_dir("mdvault"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "vault.db"


def _get_embedder():
    """Load the real model2vec embedder."""
    from model2vec import StaticModel

    model = StaticModel.from_pretrained("minishlab/potion-base-8M")

    def embed(texts: list[str]):
        return model.encode(texts)

    return embed


@app.command()
def index(
    vault_path: str = typer.Argument(..., help="Path to markdown vault directory"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to database file"),
    incremental: bool = typer.Option(False, "--incremental", help="Incremental index"),
):
    """Index a directory of markdown files."""
    vault_root = Path(vault_path).resolve()
    db_path = _resolve_db(db)
    embedder = _get_embedder()

    init_db(db_path, vault_root=str(vault_root))
    conn = get_connection(db_path)

    if incremental:
        incremental_index(conn, vault_root, embedder)
        conn.commit()
        typer.echo(f"Incremental index complete.")
    else:
        index_directory(conn, vault_root, embedder, full=True)
        conn.commit()
        typer.echo(f"Indexed {vault_root}")

    file_count = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    chunk_count = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
    typer.echo(f"Files: {file_count}, Chunks: {chunk_count}")
    conn.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to database file"),
    top_k: int = typer.Option(5, "--top-k", help="Number of results to return"),
):
    """Search the vault using hybrid BM25 + vector search."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo("Error: Database not found. Run 'mdvault index' first.")
        raise typer.Exit(1)

    embedder = _get_embedder()
    conn = get_connection(db_path)
    total = get_total_chunks(conn)
    results = hybrid_search(conn, query, embedder, top_k=top_k)
    conn.close()

    typer.echo(f"Query: {query}")
    typer.echo(f"Chunks searched: {total}")
    typer.echo("")

    for i, r in enumerate(results, start=1):
        typer.echo(
            f"[{i}] {r['file_path']} (chunk {r['chunk_idx']}) — score {r['score']:.3f}"
        )
        typer.echo("─" * 42)
        content_preview = r["raw_content"][:500]
        typer.echo(content_preview)
        typer.echo("─" * 42)
        typer.echo("")


@app.command()
def stats(
    db: Optional[str] = typer.Option(None, "--db", help="Path to database file"),
):
    """Show index statistics."""
    db_path = _resolve_db(db)
    if not db_path.exists():
        typer.echo("Error: Database not found. Run 'mdvault index' first.")
        raise typer.Exit(1)

    conn = get_connection(db_path)
    vault_root = conn.execute(
        "SELECT value FROM vault_config WHERE key = 'vault_root'"
    ).fetchone()
    file_count = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    chunk_count = get_total_chunks(conn)
    conn.close()

    db_size = db_path.stat().st_size
    if db_size > 1024 * 1024:
        size_str = f"{db_size / (1024 * 1024):.0f} MB"
    else:
        size_str = f"{db_size / 1024:.0f} KB"

    typer.echo(f"Vault root    : {vault_root['value'] if vault_root else 'N/A'}")
    typer.echo(f"DB path       : {db_path}")
    typer.echo(f"Files indexed : {file_count:,}")
    typer.echo(f"Total chunks  : {chunk_count:,}")
    typer.echo(f"DB size       : {size_str}")


@app.command()
def related(
    file_path: str = typer.Argument(..., help="Relative path of the file in the vault"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to database file"),
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
    db: Optional[str] = typer.Option(None, "--db", help="Path to database file"),
):
    """Start the MCP server for Claude Code integration."""
    db_path = _resolve_db(db)
    # Set env so mcp_server can find the DB
    os.environ["VAULT_DB"] = str(db_path)
    from mdvault.mcp_server import run

    run()
