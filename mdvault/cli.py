import contextlib
import json as _json
import os
import sqlite3
from datetime import UTC
from pathlib import Path

import platformdirs
import typer

from mdvault.db import get_connection, init_db
from mdvault.indexer import index_directory
from mdvault.memory import delete_memory as _delete_mem
from mdvault.memory import store_memory as _store_mem
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
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{_MODEL_ID.replace('/', '--')}"
    if cache_dir.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(_MODEL_ID)

    def embed(texts: list[str]):
        return model.encode(texts)

    return embed


def _vault_roots(conn: sqlite3.Connection) -> dict[str, str]:
    """Return {vault_name: absolute_path} from vault_config."""
    rows = conn.execute("SELECT key, value FROM vault_config WHERE key LIKE 'vault_root:%'").fetchall()
    return {row["key"].removeprefix("vault_root:"): row["value"] for row in rows}


def _disk_path(file_path: str, roots: dict[str, str]) -> str | None:
    """Resolve a vault-relative file_path (e.g. '.claude/foo.md') to its absolute disk path."""
    parts = file_path.split("/", 1)
    if len(parts) < 2:
        return None
    vault_name, rel = parts
    root = roots.get(vault_name)
    if not root:
        return None
    return str(Path(root) / rel)


def _require_db(db_path: Path) -> None:
    if not db_path.exists():
        typer.echo("Error: Database not found. Run 'mdvault index' first.", err=True)
        raise typer.Exit(1)


@app.command()
def index(
    vault_path: str = typer.Argument(..., help="Path to markdown vault directory"),
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    full: bool = typer.Option(False, "--full", help="Force full re-index of this vault (default: additive)"),
    no_gitignore: bool = typer.Option(False, "--no-gitignore", help="Ignore .gitignore rules (index all .md files)"),
    json: bool = typer.Option(False, "--json", "-j", help="Output JSON"),
):
    """Index a directory of markdown files. Additive by default — multiple vaults can share one DB."""
    vault_root = Path(vault_path).resolve()
    if not vault_root.exists():
        typer.echo(f"Error: path does not exist: {vault_root}", err=True)
        raise typer.Exit(1)
    if not vault_root.is_dir():
        typer.echo(f"Error: path is not a directory: {vault_root}", err=True)
        raise typer.Exit(1)

    db_path = _resolve_db(db)
    embedder = _get_embedder()

    init_db(db_path)
    conn = get_connection(db_path)

    index_directory(conn, vault_root, embedder, full=full, no_gitignore=no_gitignore)
    conn.commit()

    file_count = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    chunk_count = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
    conn.close()

    if json:
        typer.echo(
            _json.dumps(
                {
                    "vault": vault_root.name,
                    "path": str(vault_root),
                    "mode": "full" if full else "incremental",
                    "files": file_count,
                    "chunks": chunk_count,
                }
            )
        )
    else:
        mode = "full re-index" if full else "indexed"
        typer.echo(f"{vault_root.name}: {mode}")
        typer.echo(f"Files: {file_count}, Chunks: {chunk_count}")


@app.command()
def reindex(
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    full: bool = typer.Option(False, "--full", help="Force full re-index of all vaults"),
    json: bool = typer.Option(False, "--json", "-j", help="Output JSON"),
):
    """Re-index all known vaults using their original paths and options."""
    db_path = _resolve_db(db)
    _require_db(db_path)

    conn = get_connection(db_path)
    roots = _vault_roots(conn)
    if not roots:
        conn.close()
        typer.echo("No vaults registered. Run 'mdvault index <path>' first.", err=True)
        raise typer.Exit(1)

    # Read per-vault options
    opts_rows = conn.execute("SELECT key, value FROM vault_config WHERE key LIKE 'vault_opts:%'").fetchall()
    vault_opts = {row["key"].removeprefix("vault_opts:"): row["value"] for row in opts_rows}

    embedder = _get_embedder()
    results = []
    for name, path in sorted(roots.items()):
        vault_root = Path(path)
        if not vault_root.exists():
            typer.echo(f"Warning: {name} ({path}) no longer exists, skipping.", err=True)
            continue
        no_gitignore = "no_gitignore" in vault_opts.get(name, "")
        index_directory(conn, vault_root, embedder, full=full, no_gitignore=no_gitignore)
        conn.commit()
        v_files = conn.execute("SELECT COUNT(*) as c FROM files WHERE file_path LIKE ?", (f"{name}/%",)).fetchone()["c"]
        v_chunks = conn.execute(
            "SELECT COUNT(*) as c FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.file_path LIKE ?",
            (f"{name}/%",),
        ).fetchone()["c"]
        results.append({"vault": name, "path": path, "files": v_files, "chunks": v_chunks})
        if not json:
            typer.echo(f"{name}: {v_files:,} files, {v_chunks:,} chunks")

    conn.close()
    if json:
        typer.echo(_json.dumps(results))


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    top_k: int = typer.Option(5, "--top-k", help="Number of results to return"),
    expand: bool = typer.Option(False, "--expand", help="Expand query via local LLM (requires Ollama)"),
    expand_model: str = typer.Option("qwen3:0.6b", "--expand-model", help="Ollama model for query expansion"),
    source: str | None = typer.Option(None, "--source", help="Filter: 'files' or 'memories'"),
    vault: str | None = typer.Option(None, "--vault", help="Filter results by vault name"),
    paths_only: bool = typer.Option(False, "--paths-only", help="Output unique file paths only"),
    no_truncate: bool = typer.Option(False, "--no-truncate", help="Show full content (no 500-char limit)"),
    json: bool = typer.Option(False, "--json", "-j", help="Output JSON"),
):
    """Search the vault using hybrid BM25 + vector search."""
    db_path = _resolve_db(db)
    _require_db(db_path)

    if source and source not in ("files", "memories"):
        typer.echo("Error: --source must be 'files' or 'memories'", err=True)
        raise typer.Exit(1)

    embedder = _get_embedder()
    conn = get_connection(db_path)
    total = get_total_chunks(conn)
    roots = _vault_roots(conn)
    results = hybrid_search(conn, query, embedder, top_k=top_k, expand=expand, expand_model=expand_model, source=source)
    conn.close()

    if vault:
        prefix = f"{vault}/"
        results = [r for r in results if r["file_path"].startswith(prefix)]

    if paths_only:
        seen: set[str] = set()
        paths: list[str] = []
        for r in results:
            fp = r["file_path"]
            if fp not in seen:
                seen.add(fp)
                paths.append(fp)
        if json:
            typer.echo(_json.dumps(paths))
        else:
            for p in paths:
                typer.echo(p)
        return

    if json:
        typer.echo(
            _json.dumps(
                {
                    "query": query,
                    "total_chunks": total,
                    "results": [
                        {
                            "file_path": r["file_path"],
                            "disk_path": _disk_path(r["file_path"], roots),
                            "chunk_idx": r["chunk_idx"],
                            "score": round(r["score"], 4),
                            "content": r["raw_content"],
                        }
                        for r in results
                    ],
                }
            )
        )
    else:
        if not results:
            typer.echo("No results.")
            return

        for i, r in enumerate(results, start=1):
            if no_truncate:
                typer.echo(f"[{i}] {r['score']:.3f}  {r['file_path']}:{r['chunk_idx']}")
                typer.echo(r["raw_content"])
                typer.echo("")
            else:
                flat = " ".join(r["raw_content"].split())[:200]
                typer.echo(f"{r['file_path']}:{r['chunk_idx']}")
                typer.echo(flat)
                typer.echo("")


@app.command()
def stats(
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    json: bool = typer.Option(False, "--json", "-j", help="Output JSON"),
):
    """Show index statistics."""
    db_path = _resolve_db(db)
    _require_db(db_path)

    conn = get_connection(db_path)
    roots = _vault_roots(conn)
    file_count = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    chunk_count = get_total_chunks(conn)

    vaults = []
    for name, path in sorted(roots.items()):
        v_files = conn.execute("SELECT COUNT(*) as c FROM files WHERE file_path LIKE ?", (f"{name}/%",)).fetchone()["c"]
        v_chunks = conn.execute(
            "SELECT COUNT(*) as c FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.file_path LIKE ?",
            (f"{name}/%",),
        ).fetchone()["c"]
        vaults.append({"name": name, "path": path, "files": v_files, "chunks": v_chunks})

    mem_count = conn.execute("SELECT COUNT(*) as c FROM files WHERE file_path LIKE 'memory://%'").fetchone()["c"]

    # Extra stats — wrapped for backward compat with older DBs
    query_count = cluster_count = gap_count = 0
    src_counts: list[sqlite3.Row] = []
    with contextlib.suppress(sqlite3.OperationalError):
        query_count = conn.execute(
            "SELECT COUNT(*) as c FROM query_log WHERE created_at > datetime('now', '-30 days')"
        ).fetchone()["c"]
    with contextlib.suppress(sqlite3.OperationalError):
        cluster_count = conn.execute("SELECT COUNT(*) as c FROM query_clusters").fetchone()["c"]
    with contextlib.suppress(sqlite3.OperationalError):
        gap_count = conn.execute("SELECT COUNT(*) as c FROM files WHERE file_path LIKE 'memory://gaps/%'").fetchone()[
            "c"
        ]
    with contextlib.suppress(sqlite3.OperationalError):
        src_counts = conn.execute("SELECT source, COUNT(*) as c FROM memory_meta GROUP BY source").fetchall()

    conn.close()

    db_size = db_path.stat().st_size

    if json:
        data = {
            "db_path": str(db_path),
            "db_size_bytes": db_size,
            "total_files": file_count,
            "total_chunks": chunk_count,
            "memories": mem_count,
            "memory_by_source": {r["source"]: r["c"] for r in src_counts},
            "queries_30d": query_count,
            "query_clusters": cluster_count,
            "knowledge_gaps": gap_count,
            "vaults": vaults,
        }
        typer.echo(_json.dumps(data))
    else:
        if vaults:
            for v in vaults:
                typer.echo(f"Vault         : {v['name']} → {v['path']} ({v['files']:,} files, {v['chunks']:,} chunks)")
        else:
            typer.echo("Vault         : (none)")
        if mem_count:
            if src_counts:
                parts = ", ".join(f"{r['c']} {r['source']}" for r in src_counts)
                typer.echo(f"Memories      : {mem_count} ({parts})")
            else:
                typer.echo(f"Memories      : {mem_count}")
        if query_count:
            typer.echo(f"Queries (30d) : {query_count}")
        if cluster_count:
            typer.echo(f"Query clusters: {cluster_count}")
        if gap_count:
            typer.echo(f"Knowledge gaps: {gap_count}")
        typer.echo(f"DB path       : {db_path}")
        typer.echo(f"Files indexed : {file_count:,}")
        typer.echo(f"Total chunks  : {chunk_count:,}")
        size_str = f"{db_size / (1024 * 1024):.0f} MB" if db_size > 1024 * 1024 else f"{db_size / 1024:.0f} KB"
        typer.echo(f"DB size       : {size_str}")


@app.command()
def related(
    file_path: str = typer.Argument(..., help="Relative path of the file in the vault"),
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    top_k: int = typer.Option(5, "--top-k", help="Number of similar files to return"),
    json: bool = typer.Option(False, "--json", "-j", help="Output JSON"),
):
    """Show related notes: links, backlinks, and semantically similar files."""
    db_path = _resolve_db(db)
    _require_db(db_path)

    embedder = _get_embedder()
    conn = get_connection(db_path)
    result = _related_notes(conn, file_path, embedder, top_k=top_k)
    conn.close()

    if json:
        typer.echo(_json.dumps(result))
    else:
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
def remember(
    content: str = typer.Argument(..., help="Memory content to store"),
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    namespace: str = typer.Option("", "--namespace", "-n", help="Namespace (e.g. user/prefs)"),
    source: str = typer.Option("cli", "--source", "-s", help="Source identifier"),
    json: bool = typer.Option(False, "--json", "-j", help="Output JSON"),
):
    """Store a memory in the vault."""
    db_path = _resolve_db(db)
    init_db(db_path)
    embedder = _get_embedder()
    conn = get_connection(db_path)
    result = _store_mem(conn, content, embedder, namespace=namespace, source=source)
    conn.commit()
    conn.close()
    if json:
        typer.echo(_json.dumps(result))
    else:
        typer.echo(f"Stored {result['id']} ({result['chunks']} chunks)")


@app.command()
def forget(
    id: str | None = typer.Option(None, "--id", help="Memory ID to delete"),
    namespace: str | None = typer.Option(None, "--namespace", "-n", help="Delete all memories in namespace"),
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    json: bool = typer.Option(False, "--json", "-j", help="Output JSON"),
):
    """Delete memories by ID or namespace."""
    if not id and not namespace:
        typer.echo("Error: provide --id or --namespace", err=True)
        raise typer.Exit(1)
    db_path = _resolve_db(db)
    _require_db(db_path)
    conn = get_connection(db_path)
    count = _delete_mem(conn, id=id, namespace=namespace)
    conn.commit()
    conn.close()
    if json:
        typer.echo(_json.dumps({"deleted": count}))
    else:
        typer.echo(f"Deleted {count} memories")


def _last_hit_display(last_hit_at: str | None) -> str:
    if not last_hit_at:
        return "never"
    from datetime import datetime

    dt = datetime.fromisoformat(last_hit_at)
    days = (datetime.now(tz=UTC) - dt.replace(tzinfo=UTC)).days
    if days == 0:
        return "today"
    if days == 1:
        return "1d ago"
    return f"{days}d ago"


@app.command()
def memories(
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    namespace: str | None = typer.Option(None, "--namespace", "-n", help="Filter by namespace"),
    json: bool = typer.Option(False, "--json", "-j", help="Output JSON"),
):
    """List stored memories."""
    db_path = _resolve_db(db)
    _require_db(db_path)
    conn = get_connection(db_path)

    sql = (
        "SELECT f.file_path, mm.namespace, mm.source, mm.created_at,"
        " mm.confidence, mm.hit_count, mm.last_hit_at,"
        " (SELECT COUNT(*) FROM chunks c WHERE c.file_id = f.id)"
        " as chunk_count"
        " FROM memory_meta mm"
        " JOIN files f ON f.id = mm.file_id"
    )
    params: list = []
    if namespace:
        sql += " WHERE mm.namespace = ?"
        params.append(namespace)
    sql += " ORDER BY mm.created_at DESC"

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    items = []
    for row in rows:
        path = row["file_path"]
        items.append(
            {
                "id": path.rsplit("/", 1)[-1],
                "file_path": path,
                "namespace": row["namespace"] or "",
                "source": row["source"],
                "confidence": row["confidence"],
                "hit_count": row["hit_count"],
                "last_hit_at": row["last_hit_at"],
                "chunks": row["chunk_count"],
                "created_at": row["created_at"],
            }
        )

    if json:
        typer.echo(_json.dumps(items))
    else:
        if not items:
            typer.echo("No memories stored.")
            return
        typer.echo(f"Memories: {len(items)}")
        typer.echo("")
        for m in items:
            ns = m["namespace"] or "(none)"
            last = _last_hit_display(m.get("last_hit_at"))
            typer.echo(
                f"  {m['id']}  ns={ns}  src={m['source']}"
                f"  conf={m['confidence']:.2f}  hits={m['hit_count']}"
                f"  last={last}  {m['chunks']} chunks"
            )


@app.command()
def gaps(
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output JSON",
    ),
):
    """Show knowledge gaps -- topics frequently searched with poor results."""
    db_path = _resolve_db(db)
    _require_db(db_path)
    conn = get_connection(db_path)

    try:
        rows = conn.execute(
            """SELECT f.file_path, c.raw_content, mm.metadata,
                mm.created_at, qc.query_count, qc.avg_score
            FROM memory_meta mm
            JOIN files f ON f.id = mm.file_id
            JOIN chunks c ON c.file_id = f.id AND c.chunk_idx = 0
            LEFT JOIN query_clusters qc
                ON qc.id = CAST(
                    json_extract(mm.metadata, '$.cluster_id') AS INTEGER
                )
            WHERE mm.namespace = 'gaps'
            ORDER BY qc.query_count DESC"""
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    if json:
        items = [
            {
                "id": row["file_path"].rsplit("/", 1)[-1],
                "query": _json.loads(row["metadata"]).get("from_query", ""),
                "query_count": row["query_count"],
                "avg_score": row["avg_score"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        typer.echo(_json.dumps(items))
    else:
        if not rows:
            typer.echo("No knowledge gaps detected.")
            return
        typer.echo(f"Knowledge gaps: {len(rows)}")
        typer.echo("")
        for row in rows:
            mid = row["file_path"].rsplit("/", 1)[-1]
            meta = _json.loads(row["metadata"])
            query = meta.get("from_query", "unknown")
            qc = row["query_count"] or "?"
            score = f"{row['avg_score']:.2f}" if row["avg_score"] else "?"
            typer.echo(f'  {mid}  "{query}"  (queried {qc} times, best score {score})')


@app.command()
def read(
    file_path: str = typer.Argument(..., help="Vault-relative file path (e.g. 'myvault/notes/foo.md')"),
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    json: bool = typer.Option(False, "--json", "-j", help="Output JSON"),
):
    """Read the full content of an indexed file."""
    db_path = _resolve_db(db)
    _require_db(db_path)

    conn = get_connection(db_path)
    roots = _vault_roots(conn)

    # Check file exists in index
    row = conn.execute("SELECT id FROM files WHERE file_path = ?", (file_path,)).fetchone()
    if not row:
        conn.close()
        typer.echo(f"Error: '{file_path}' not found in index.", err=True)
        raise typer.Exit(1)

    # Get all chunks (raw_content) in order
    chunks = conn.execute(
        "SELECT chunk_idx, raw_content FROM chunks WHERE file_id = ? ORDER BY chunk_idx",
        (row["id"],),
    ).fetchall()
    conn.close()

    disk = _disk_path(file_path, roots)
    content = "\n\n".join(c["raw_content"] for c in chunks)

    if json:
        typer.echo(
            _json.dumps(
                {
                    "file_path": file_path,
                    "disk_path": disk,
                    "chunks": len(chunks),
                    "content": content,
                }
            )
        )
    else:
        typer.echo(f"File: {file_path}")
        if disk:
            typer.echo(f"Disk: {disk}")
        typer.echo(f"Chunks: {len(chunks)}")
        typer.echo("")
        typer.echo(content)


@app.command(name="list")
def list_files(
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
    vault: str | None = typer.Option(None, "--vault", help="Filter by vault name"),
    pattern: str | None = typer.Option(
        None, "--pattern", help="Glob pattern to filter file paths (e.g. '*.md', 'notes/*')"
    ),
    json: bool = typer.Option(False, "--json", "-j", help="Output JSON"),
):
    """List indexed files. Filterable by vault and glob pattern."""
    import fnmatch

    db_path = _resolve_db(db)
    _require_db(db_path)

    conn = get_connection(db_path)
    roots = _vault_roots(conn)

    sql = "SELECT file_path FROM files WHERE file_path NOT LIKE 'memory://%'"
    params: list = []
    if vault:
        sql += " AND file_path LIKE ?"
        params.append(f"{vault}/%")
    sql += " ORDER BY file_path"

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    files = [r["file_path"] for r in rows]
    if pattern:
        files = [
            f
            for f in files
            if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(f.split("/", 1)[-1] if "/" in f else f, pattern)
        ]

    if json:
        items = []
        for fp in files:
            parts = fp.split("/", 1)
            items.append(
                {
                    "file_path": fp,
                    "disk_path": _disk_path(fp, roots),
                    "vault": parts[0] if len(parts) > 1 else None,
                }
            )
        typer.echo(_json.dumps(items))
    else:
        typer.echo(f"Files: {len(files)}")
        for fp in files:
            typer.echo(f"  {fp}")


@app.command()
def serve(
    db: str | None = typer.Option(None, "--db", help="Path to database file"),
):
    """Start the MCP server for Claude Code integration."""
    db_path = _resolve_db(db)
    os.environ["VAULT_DB"] = str(db_path)
    from mdvault.mcp_server import run

    run()
