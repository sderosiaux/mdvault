import sqlite3
from collections.abc import Callable

import numpy as np


def cluster_recent_queries(
    conn: sqlite3.Connection,
    embedder: Callable[[list[str]], np.ndarray],
    similarity_threshold: float = 0.85,
) -> None:
    """Cluster recent queries by embedding similarity. Updates query_clusters."""
    # Get existing cluster canonicals and their embeddings
    existing = conn.execute("SELECT id, canonical FROM query_clusters").fetchall()
    cluster_vecs: list[tuple[int, np.ndarray]] = []
    if existing:
        canonicals = [row["canonical"] for row in existing]
        vecs = embedder(canonicals)
        for row, vec in zip(existing, vecs, strict=True):
            cluster_vecs.append((row["id"], vec))

    # Get recent queries (last 100)
    rows = conn.execute(
        """SELECT ql.id, ql.query, ql.top_score
        FROM query_log ql
        ORDER BY ql.created_at DESC
        LIMIT 100""",
    ).fetchall()

    if not rows:
        return

    query_texts = [r["query"] for r in rows]
    query_vecs = embedder(query_texts)

    for row, qvec in zip(rows, query_vecs, strict=True):
        best_cluster_id = None
        best_sim = 0.0

        for cid, cvec in cluster_vecs:
            norm = np.linalg.norm(qvec) * np.linalg.norm(cvec) + 1e-9
            sim = float(np.dot(qvec, cvec) / norm)
            if sim > best_sim:
                best_sim = sim
                best_cluster_id = cid

        if best_sim >= similarity_threshold and best_cluster_id is not None:
            conn.execute(
                """UPDATE query_clusters
                SET query_count = query_count + 1,
                    avg_score = (avg_score * (query_count - 1) + ?) / query_count,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?""",
                (row["top_score"], best_cluster_id),
            )
            # Update canonical if this query scored better
            current = conn.execute(
                "SELECT canonical, avg_score FROM query_clusters WHERE id = ?",
                (best_cluster_id,),
            ).fetchone()
            if row["top_score"] and current["avg_score"] and row["top_score"] > current["avg_score"]:
                conn.execute(
                    "UPDATE query_clusters SET canonical = ? WHERE id = ?",
                    (row["query"], best_cluster_id),
                )
        else:
            conn.execute(
                "INSERT INTO query_clusters (canonical, query_count, avg_score) VALUES (?, 1, ?)",
                (row["query"], row["top_score"]),
            )
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            cluster_vecs.append((new_id, qvec))


def maybe_promote(
    conn: sqlite3.Connection,
    embedder: Callable[[list[str]], np.ndarray],
    max_promotions: int = 3,
    min_query_count: int = 5,
    good_score_threshold: float = 0.3,
    gap_score_threshold: float = 0.15,
) -> int:
    """Promote recurring query clusters into memories. Returns count."""
    from mdvault.memory import store_memory
    from mdvault.retriever import hybrid_search

    candidates = conn.execute(
        """SELECT id, canonical, query_count, avg_score
        FROM query_clusters
        WHERE promoted = FALSE AND query_count >= ?
        ORDER BY query_count DESC""",
        (min_query_count,),
    ).fetchall()

    promoted = 0
    for row in candidates:
        if promoted >= max_promotions:
            break

        if row["avg_score"] and row["avg_score"] >= good_score_threshold:
            # Case 1: recurring query with good results
            results = hybrid_search(
                conn,
                row["canonical"],
                embedder,
                top_k=1,
            )
            if results:
                store_memory(
                    conn,
                    results[0]["raw_content"],
                    embedder,
                    namespace="auto",
                    source="promoted",
                    metadata={
                        "from_query": row["canonical"],
                        "cluster_id": row["id"],
                    },
                )
                conn.execute(
                    "UPDATE query_clusters SET promoted = TRUE, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (row["id"],),
                )
                promoted += 1

        elif row["avg_score"] is not None and row["avg_score"] < gap_score_threshold:
            # Case 2: knowledge gap
            gap_content = f"[Knowledge gap] Recurring query with no good results: {row['canonical']}"
            store_memory(
                conn,
                gap_content,
                embedder,
                namespace="gaps",
                source="promoted",
                metadata={
                    "from_query": row["canonical"],
                    "cluster_id": row["id"],
                    "gap": True,
                },
            )
            conn.execute(
                "UPDATE query_clusters SET promoted = TRUE, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (row["id"],),
            )
            promoted += 1

    return promoted
