import json

from mdvault.db import get_connection
from mdvault.memory import delete_memory, store_memory


def test_store_memory_creates_file_and_meta(db_path, mock_embedder):
    """store_memory inserts into files, memory_meta, chunks, chunks_fts, chunks_vec."""
    conn = get_connection(db_path)
    result = store_memory(conn, "L'utilisateur préfère Python", mock_embedder, namespace="user/prefs", source="agent")
    conn.commit()

    assert "id" in result
    assert result["chunks"] == 1

    file_row = conn.execute("SELECT * FROM files WHERE file_path LIKE 'memory://%'").fetchone()
    assert file_row is not None
    assert file_row["file_path"].startswith("memory://user/prefs/")

    meta = conn.execute("SELECT * FROM memory_meta WHERE file_id = ?", (file_row["id"],)).fetchone()
    assert meta["namespace"] == "user/prefs"
    assert meta["source"] == "agent"

    chunks = conn.execute("SELECT * FROM chunks WHERE file_id = ?", (file_row["id"],)).fetchall()
    assert len(chunks) == 1
    assert "Python" in chunks[0]["raw_content"]

    fts = conn.execute("SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'Python'").fetchone()
    assert fts is not None
    conn.close()


def test_store_memory_short_content_no_chunking(db_path, mock_embedder):
    """Content under 400 words is stored as a single chunk."""
    conn = get_connection(db_path)
    result = store_memory(conn, "Short fact about something interesting enough", mock_embedder)
    conn.commit()
    assert result["chunks"] == 1
    conn.close()


def test_store_memory_long_content_auto_chunks(db_path, mock_embedder):
    """Content over 400 words is automatically chunked."""
    conn = get_connection(db_path)
    long_content = "## Section A\n\n" + " ".join(["word"] * 300) + "\n\n## Section B\n\n" + " ".join(["other"] * 300)
    result = store_memory(conn, long_content, mock_embedder, namespace="docs")
    conn.commit()
    assert result["chunks"] > 1
    conn.close()


def test_store_memory_default_namespace_empty(db_path, mock_embedder):
    """Default namespace is empty string."""
    conn = get_connection(db_path)
    result = store_memory(conn, "No namespace memory content here", mock_embedder)
    conn.commit()

    file_row = conn.execute("SELECT * FROM files WHERE id = ?", (result["file_id"],)).fetchone()
    assert file_row["file_path"].startswith("memory:///")

    meta = conn.execute("SELECT * FROM memory_meta WHERE file_id = ?", (result["file_id"],)).fetchone()
    assert meta["namespace"] == ""
    conn.close()


def test_store_memory_metadata_json(db_path, mock_embedder):
    """Custom metadata is stored as JSON."""
    conn = get_connection(db_path)
    store_memory(conn, "With meta content here", mock_embedder, metadata={"confidence": 0.9, "tag": "test"})
    conn.commit()

    meta = conn.execute("SELECT metadata FROM memory_meta").fetchone()
    parsed = json.loads(meta["metadata"])
    assert parsed["confidence"] == 0.9
    assert parsed["tag"] == "test"
    conn.close()


def test_store_memory_context_prefix(db_path, mock_embedder):
    """Chunk content (not raw_content) has memory:// context prefix."""
    conn = get_connection(db_path)
    store_memory(conn, "Some fact about preferences", mock_embedder, namespace="user/prefs")
    conn.commit()

    chunk = conn.execute("SELECT content, raw_content FROM chunks").fetchone()
    assert chunk["content"].startswith("[memory://user/prefs")
    assert not chunk["raw_content"].startswith("[")
    conn.close()


def test_store_memory_returns_unique_ids(db_path, mock_embedder):
    """Each store_memory call returns a unique id."""
    conn = get_connection(db_path)
    r1 = store_memory(conn, "fact one content here", mock_embedder)
    r2 = store_memory(conn, "fact two content here", mock_embedder)
    conn.commit()
    assert r1["id"] != r2["id"]
    conn.close()


def test_delete_memory_by_id(db_path, mock_embedder):
    """delete_memory by id removes file, meta, chunks, fts, vec."""
    conn = get_connection(db_path)
    result = store_memory(conn, "To delete this memory", mock_embedder, namespace="tmp")
    conn.commit()

    deleted = delete_memory(conn, id=result["id"])
    conn.commit()

    assert deleted == 1
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) as c FROM memory_meta").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"] == 0
    conn.close()


def test_delete_memory_by_namespace(db_path, mock_embedder):
    """delete_memory by namespace removes all memories in that namespace."""
    conn = get_connection(db_path)
    store_memory(conn, "Fact A in project", mock_embedder, namespace="project/x")
    store_memory(conn, "Fact B in project", mock_embedder, namespace="project/x")
    store_memory(conn, "Fact C in other ns", mock_embedder, namespace="other")
    conn.commit()

    deleted = delete_memory(conn, namespace="project/x")
    conn.commit()

    assert deleted == 2
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 1
    conn.close()


def test_delete_memory_nonexistent_returns_zero(db_path):
    """delete_memory with unknown id returns 0."""
    conn = get_connection(db_path)
    assert delete_memory(conn, id="nonexist") == 0
    conn.close()


def test_delete_memory_requires_id_or_namespace(db_path):
    """delete_memory with neither id nor namespace raises ValueError."""
    conn = get_connection(db_path)
    import pytest as _pt

    with _pt.raises(ValueError):
        delete_memory(conn)
    conn.close()
