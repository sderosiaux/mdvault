from mdvault.db import get_connection, serialize_f32
from mdvault.memory import store_memory
from mdvault.promoter import cluster_recent_queries, maybe_promote


def test_cluster_groups_similar_queries(db_path, mock_embedder):
    """Queries with similar embeddings are grouped into the same cluster."""
    conn = get_connection(db_path)

    # Insert 5 identical queries (same text = same embedding from mock)
    for _i in range(5):
        conn.execute(
            "INSERT INTO query_log (query, top_score, result_count) VALUES (?, ?, ?)",
            ("kafka timeout", 0.4, 3),
        )
        log_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        vec = mock_embedder(["kafka timeout"])[0]
        conn.execute(
            "INSERT INTO query_vec (rowid, embedding) VALUES (?, ?)",
            (log_id, serialize_f32(vec)),
        )

    # Insert 1 different query
    conn.execute(
        "INSERT INTO query_log (query, top_score, result_count) VALUES (?, ?, ?)",
        ("nginx proxy", 0.5, 3),
    )
    log_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    vec = mock_embedder(["nginx proxy"])[0]
    conn.execute(
        "INSERT INTO query_vec (rowid, embedding) VALUES (?, ?)",
        (log_id, serialize_f32(vec)),
    )
    conn.commit()

    cluster_recent_queries(conn, mock_embedder)

    clusters = conn.execute(
        "SELECT * FROM query_clusters ORDER BY query_count DESC",
    ).fetchall()
    assert len(clusters) >= 1
    assert clusters[0]["query_count"] >= 5
    assert clusters[0]["canonical"] == "kafka timeout"
    conn.close()


def test_promote_recurring_good_query(db_path, mock_embedder):
    """Cluster with count >= 5 and avg_score >= 0.3 creates a promoted memory."""
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO query_clusters (canonical, query_count, avg_score, promoted) VALUES (?, ?, ?, ?)",
        ("kafka consumer timeout", 6, 0.45, False),
    )
    conn.commit()

    # Index a real file so source="files" promotion can find it
    from mdvault.indexer import index_directory
    from tests.conftest import FIXTURES_DIR

    index_directory(conn, FIXTURES_DIR, mock_embedder, full=True)
    conn.commit()

    promoted = maybe_promote(conn, mock_embedder, max_promotions=3)
    assert promoted >= 1

    cluster = conn.execute(
        "SELECT promoted FROM query_clusters WHERE canonical = 'kafka consumer timeout'",
    ).fetchone()
    assert cluster["promoted"] == 1

    mem = conn.execute(
        "SELECT * FROM files WHERE file_path LIKE 'memory://auto/%'",
    ).fetchone()
    assert mem is not None
    conn.close()


def test_promote_knowledge_gap(db_path, mock_embedder):
    """Cluster with count >= 5 and avg_score < 0.15 creates a gap memory."""
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO query_clusters (canonical, query_count, avg_score, promoted) VALUES (?, ?, ?, ?)",
        ("terraform state locking", 7, 0.08, False),
    )
    conn.commit()

    promoted = maybe_promote(conn, mock_embedder, max_promotions=3)
    assert promoted >= 1

    cluster = conn.execute(
        "SELECT promoted FROM query_clusters WHERE canonical = 'terraform state locking'",
    ).fetchone()
    assert cluster["promoted"] == 1

    mem = conn.execute(
        "SELECT * FROM files WHERE file_path LIKE 'memory://gaps/%'",
    ).fetchone()
    assert mem is not None
    conn.close()


def test_promote_respects_max_promotions(db_path, mock_embedder):
    """At most max_promotions memories are created per cycle."""
    conn = get_connection(db_path)
    for i in range(10):
        conn.execute(
            "INSERT INTO query_clusters (canonical, query_count, avg_score, promoted) VALUES (?, ?, ?, ?)",
            (f"gap query {i}", 6, 0.05, False),
        )
    conn.commit()

    promoted = maybe_promote(conn, mock_embedder, max_promotions=3)
    assert promoted <= 3
    conn.close()


def test_full_lifecycle(db_path, mock_embedder):
    """End-to-end: store, search, hit tracking, query logging, clustering."""
    from mdvault.retriever import hybrid_search

    conn = get_connection(db_path)

    store_memory(
        conn,
        "Kafka timeout is controlled by max.poll.interval.ms setting",
        mock_embedder,
        namespace="kb",
        source="user",
    )
    conn.commit()

    # Search 20 times to trigger promotion cycle
    for _ in range(20):
        hybrid_search(
            conn,
            "kafka timeout configuration",
            mock_embedder,
            top_k=3,
        )

    # Verify query logging
    q_count = conn.execute("SELECT COUNT(*) as c FROM query_log").fetchone()["c"]
    assert q_count >= 20

    # Verify clustering happened
    clusters = conn.execute("SELECT * FROM query_clusters").fetchall()
    assert len(clusters) >= 1

    # Verify hit tracking on the memory
    meta = conn.execute("SELECT hit_count FROM memory_meta LIMIT 1").fetchone()
    assert meta["hit_count"] > 0

    conn.close()


def test_promote_skips_already_promoted(db_path, mock_embedder):
    """Clusters already promoted are skipped."""
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO query_clusters (canonical, query_count, avg_score, promoted) VALUES (?, ?, ?, ?)",
        ("already done", 10, 0.5, True),
    )
    conn.commit()

    promoted = maybe_promote(conn, mock_embedder, max_promotions=3)
    assert promoted == 0
    conn.close()
