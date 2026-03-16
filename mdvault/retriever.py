import json
import posixpath
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Callable

import numpy as np

from mdvault.db import serialize_f32


def bm25_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 50,
    source: str | None = None,
    namespace: str | None = None,
) -> list[dict]:
    """FTS5 BM25 search. Returns ranked results (best first)."""
    # Build FTS5 query: bigrams (boosted via NEAR) + unigrams for partial matches
    tokens = [token.replace('"', '""') for token in query.split() if len(token) > 2]
    parts = [f'NEAR("{tokens[i]}" "{tokens[i + 1]}", 3)' for i in range(len(tokens) - 1)]
    parts.extend(f'"{t}"' for t in tokens)
    # Add focused AND clause: top 3 longest terms (FTS5 AND has higher precedence than OR)
    sorted_toks = sorted(tokens, key=len, reverse=True)
    if len(sorted_toks) >= 3:
        focused = " ".join(f'"{t}"' for t in sorted_toks[:3])
        parts.insert(0, focused)
    fts_query = " OR ".join(parts) if parts else query

    source_clause = ""
    params: list = [fts_query]
    if source == "memories":
        source_clause += " AND f.file_path LIKE 'memory://%'"
    elif source == "files":
        source_clause += " AND f.file_path NOT LIKE 'memory://%'"
    if namespace is not None:
        source_clause += " AND f.file_path LIKE ?"
        params.append(f"memory://{namespace}/%")
    params.append(top_k)

    try:
        rows = conn.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                f.file_path,
                c.chunk_idx,
                c.content,
                c.raw_content,
                fts.rank AS bm25_rank
            FROM chunks_fts fts
            JOIN chunks c ON c.id = fts.rowid
            JOIN files f ON f.id = c.file_id
            WHERE chunks_fts MATCH ?{source_clause}
            ORDER BY fts.rank ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    return [dict(row) for row in rows]


def vector_search(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    top_k: int = 50,
    source: str | None = None,
    namespace: str | None = None,
) -> list[dict]:
    """sqlite-vec exact nearest neighbor search. Returns ranked by distance (closest first)."""
    # Fetch extra results when filtering, since sqlite-vec doesn't support WHERE clauses
    fetch_k = top_k * 3 if (source or namespace) else top_k
    blob = serialize_f32(query_vec)
    rows = conn.execute(
        """
        SELECT
            v.rowid AS chunk_id,
            v.distance
        FROM chunks_vec v
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (blob, fetch_k),
    ).fetchall()

    if not rows:
        return []

    chunk_ids = [row["chunk_id"] for row in rows]
    placeholders = ",".join("?" * len(chunk_ids))
    details = conn.execute(
        f"""
        SELECT c.id, c.chunk_idx, c.content, c.raw_content, f.file_path
        FROM chunks c
        JOIN files f ON f.id = c.file_id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    detail_map = {d["id"]: d for d in details}

    results = []
    for row in rows:
        d = detail_map.get(row["chunk_id"])
        if d:
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "file_path": d["file_path"],
                    "chunk_idx": d["chunk_idx"],
                    "content": d["content"],
                    "raw_content": d["raw_content"],
                    "distance": row["distance"],
                }
            )

    # Post-filter by source/namespace (sqlite-vec doesn't support JOINs in WHERE)
    if source == "memories":
        results = [r for r in results if r["file_path"].startswith("memory://")]
    elif source == "files":
        results = [r for r in results if not r["file_path"].startswith("memory://")]
    if namespace is not None:
        prefix = f"memory://{namespace}/"
        results = [r for r in results if r["file_path"].startswith(prefix)]
    return results[:top_k]


def rrf_fusion(
    bm25_results: list[dict],
    vec_results: list[dict],
    top_k: int = 5,
    k: int = 10,
    bm25_weight: float = 4.0,
    vec_weight: float = 1.0,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion. score = w_bm25/(k+rank_bm25) + w_vec/(k+rank_vec)."""
    scores: dict[int, float] = {}
    metadata: dict[int, dict] = {}

    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + bm25_weight / (k + rank)
        metadata[cid] = r

    for rank, r in enumerate(vec_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + vec_weight / (k + rank)
        if cid not in metadata:
            metadata[cid] = r

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    results = []
    for cid in sorted_ids[:top_k]:
        m = metadata[cid]
        entry = {
            "chunk_id": m["chunk_id"],
            "file_path": m["file_path"],
            "chunk_idx": m["chunk_idx"],
            "content": m["content"],
            "raw_content": m["raw_content"],
            "score": scores[cid],
        }
        results.append(entry)
    return results


def expand_query_llm(query: str, model: str = "qwen3:0.6b") -> str | None:
    """Expand query via local Ollama LLM. Returns expanded text or None on failure."""
    prompt = (
        f"Expand this search query into a short paragraph (2-3 sentences) that a relevant "
        f"document might contain. Do not explain, just write the paragraph.\n\n"
        f"Query: {query}\n\nParagraph:"
    )
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 100},
        }
    ).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip() or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _dedup_results(results: list[dict], conn: sqlite3.Connection, top_k: int) -> list[dict]:
    """Deduplicate: one result per unique file content (by file_hash), then one per file_path."""
    # Build file_hash lookup for all file_paths in results
    file_paths = list({r["file_path"] for r in results})
    hash_map: dict[str, str] = {}
    if file_paths:
        placeholders = ",".join("?" * len(file_paths))
        rows = conn.execute(
            f"SELECT file_path, file_hash FROM files WHERE file_path IN ({placeholders})",
            file_paths,
        ).fetchall()
        hash_map = {row["file_path"]: row["file_hash"] for row in rows}

    seen_hashes: set[str] = set()
    seen_paths: set[str] = set()
    seen_suffixes: set[str] = set()
    deduped: list[dict] = []
    for r in results:
        fp = r["file_path"]
        fh = hash_map.get(fp, fp)  # fallback to path if hash not found
        # Dedup by content hash, exact path, or same 2-segment suffix (versioned copies)
        parts = fp.rsplit("/", 3)
        suffix = "/".join(parts[-3:]) if len(parts) >= 3 else fp
        if fh in seen_hashes or fp in seen_paths or suffix in seen_suffixes:
            continue
        seen_hashes.add(fh)
        seen_paths.add(fp)
        seen_suffixes.add(suffix)
        deduped.append(r)
        if len(deduped) >= top_k:
            break
    return deduped


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    embedder: Callable[[list[str]], np.ndarray],
    top_k: int = 5,
    expand: bool = False,
    expand_model: str = "qwen3:0.6b",
    source: str | None = None,
    namespace: str | None = None,
) -> list[dict]:
    """Full hybrid search: BM25 + vector + RRF fusion. Optional LLM query expansion."""
    # BM25 always uses original query (lexical match)
    bm25_results = bm25_search(conn, query, top_k=50, source=source, namespace=namespace)

    # Vector search: optionally use expanded query for richer embedding
    vec_query = query
    if expand:
        expanded = expand_query_llm(query, model=expand_model)
        if expanded:
            vec_query = f"{query} {expanded}"

    query_vec = embedder([vec_query])[0]
    vec_results = vector_search(conn, query_vec, top_k=50, source=source, namespace=namespace)
    # Fuse with extra headroom, then dedup by content hash to avoid duplicated files
    fused = rrf_fusion(bm25_results, vec_results, top_k=top_k * 15)

    # File-level aggregation: files with multiple matching chunks get a bonus
    file_counts: dict[str, int] = {}
    for r in fused:
        fp = r["file_path"]
        file_counts[fp] = file_counts.get(fp, 0) + 1
    for r in fused:
        extra = file_counts.get(r["file_path"], 1) - 1
        if extra > 0:
            r["score"] = r.get("score", 0.0) + extra * 0.01

    deduped = _dedup_results(fused, conn, top_k * 8)

    # Re-rank: boost by query term coverage (union across all file chunks in fused)
    query_terms = {t.lower() for t in query.split() if len(t) > 2}
    file_raw: dict[str, str] = {}
    for r in fused:
        fp = r["file_path"]
        if fp not in file_raw:
            file_raw[fp] = r.get("raw_content", "").lower()
        else:
            file_raw[fp] += " " + r.get("raw_content", "").lower()
    for r in deduped:
        raw = file_raw.get(r["file_path"], "")
        covered = sum(1 for t in query_terms if t in raw)
        coverage = covered / len(query_terms) if query_terms else 0
        r["score"] = r.get("score", 0.0) + coverage * 0.15

    # Re-rank: boost by title match (H1 heading from context prefix)
    for r in deduped:
        content = r.get("content", "")
        if content.startswith("[") and "]\n" in content:
            prefix = content.split("]\n", 1)[0][1:]
            prefix_parts = [p.strip() for p in prefix.split(">")]
            if len(prefix_parts) >= 2:
                title = prefix_parts[1]
                title_tokens = {
                    tok for tok in title.lower().replace("-", " ").replace("_", " ").split() if len(tok) > 2
                }
                if title_tokens:
                    title_matches = sum(
                        1 for qt in query_terms if any(tt.startswith(qt) or qt.startswith(tt) for tt in title_tokens)
                    )
                    if title_matches:
                        ratio = title_matches / len(query_terms) if query_terms else 0
                        r["score"] = r.get("score", 0.0) + ratio * 0.20

    # Re-rank: boost by path segment match (filename + parent dirs)
    for r in deduped:
        fp = r["file_path"]
        # Extract all path segments as tokens (dirs + filename without extension)
        parts = fp.replace("\\", "/").split("/")
        path_tokens: set[str] = set()
        for part in parts:
            seg = part.rsplit(".", 1)[0] if "." in part else part
            for tok in seg.lower().replace("-", " ").replace("_", " ").split():
                if len(tok) > 2:
                    path_tokens.add(tok)
        if not path_tokens:
            continue
        matches = sum(1 for qt in query_terms if any(pt.startswith(qt) or qt.startswith(pt) for pt in path_tokens))
        if matches:
            ratio = matches / len(query_terms) if query_terms else 0
            r["score"] = r.get("score", 0.0) + ratio * 0.30

    deduped.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return deduped[:top_k]


def get_total_chunks(conn: sqlite3.Connection) -> int:
    """Return total number of indexed chunks."""
    return conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]


def related_notes(
    conn: sqlite3.Connection,
    file_path: str,
    embedder: Callable[[list[str]], np.ndarray],
    top_k: int = 5,
) -> dict:
    """Find related notes: direct links, backlinks, and semantically similar files."""
    filename = posixpath.basename(file_path)

    # Direct links (outgoing)
    links = [
        row["target_path"]
        for row in conn.execute(
            """
            SELECT DISTINCT l.target_path
            FROM links l
            JOIN files f ON f.id = l.source_file_id
            WHERE f.file_path = ?
            """,
            (file_path,),
        ).fetchall()
    ]

    # Backlinks (incoming) — match by full path OR filename (for wikilinks)
    backlinks = [
        row["file_path"]
        for row in conn.execute(
            """
            SELECT DISTINCT f.file_path
            FROM links l
            JOIN files f ON f.id = l.source_file_id
            WHERE l.target_path = ? OR l.target_path = ?
            """,
            (file_path, filename),
        ).fetchall()
    ]

    # Semantically similar files via vector search on first chunk
    chunk = conn.execute(
        """
        SELECT c.content FROM chunks c
        JOIN files f ON f.id = c.file_id
        WHERE f.file_path = ?
        ORDER BY c.chunk_idx
        LIMIT 1
        """,
        (file_path,),
    ).fetchone()

    similar: list[str] = []
    if chunk:
        query_vec = embedder([chunk["content"]])[0]
        vec_results = vector_search(conn, query_vec, top_k=50)
        seen: set[str] = set()
        for r in vec_results:
            fp = r["file_path"]
            if fp != file_path and fp not in seen:
                seen.add(fp)
                similar.append(fp)
                if len(similar) >= top_k:
                    break

    return {
        "file_path": file_path,
        "links": links,
        "backlinks": backlinks,
        "similar": similar,
    }
