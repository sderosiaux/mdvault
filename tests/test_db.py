import sqlite3
import struct

import pytest

from mdvault.db import get_connection


def test_init_db_creates_all_tables(db_path):
    """After init_db, tables vault_config, files, chunks, chunks_fts, chunks_vec exist."""
    conn = get_connection(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow') ORDER BY name")
    names = {row["name"] for row in cursor.fetchall()}
    conn.close()
    for table in ("vault_config", "files", "chunks", "links"):
        assert table in names, f"Missing table: {table}"
    # FTS5 creates shadow tables; verify the virtual table works
    conn = get_connection(db_path)
    conn.execute("SELECT * FROM chunks_fts LIMIT 1")
    conn.execute("SELECT * FROM chunks_vec LIMIT 1")
    conn.close()


def test_vault_config_table_exists(db_path):
    """vault_config table exists and is initially empty."""
    conn = get_connection(db_path)
    count = conn.execute("SELECT COUNT(*) as c FROM vault_config").fetchone()["c"]
    conn.close()
    assert count == 0


def test_foreign_keys_enabled(db_path):
    """Inserting a chunk with nonexistent file_id raises IntegrityError."""
    conn = get_connection(db_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO chunks (file_id, chunk_idx, content, raw_content) VALUES (9999, 0, 'test', 'test')")
    conn.close()


def test_cascade_delete_removes_chunks(db_path):
    """Deleting a file cascades to its chunks."""
    conn = get_connection(db_path)
    conn.execute("INSERT INTO files (file_path, file_hash) VALUES ('test.md', 'abc123')")
    file_id = conn.execute("SELECT id FROM files WHERE file_path = 'test.md'").fetchone()["id"]
    conn.execute(
        "INSERT INTO chunks (file_id, chunk_idx, content, raw_content) VALUES (?, 0, 'hello', 'hello')", (file_id,)
    )
    conn.execute(
        "INSERT INTO chunks (file_id, chunk_idx, content, raw_content) VALUES (?, 1, 'world', 'world')", (file_id,)
    )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"] == 2
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"] == 0
    conn.close()


def test_chunks_fts_exists(db_path):
    """Can INSERT into chunks_fts with explicit rowid."""
    conn = get_connection(db_path)
    # Insert a file and chunk first so we have a valid context
    conn.execute("INSERT INTO files (file_path, file_hash) VALUES ('t.md', 'h')")
    file_id = conn.execute("SELECT id FROM files WHERE file_path = 't.md'").fetchone()["id"]
    conn.execute(
        "INSERT INTO chunks (file_id, chunk_idx, content, raw_content) VALUES (?, 0, 'test content', 'test content')",
        (file_id,),
    )
    chunk_id = conn.execute("SELECT id FROM chunks").fetchone()["id"]
    conn.execute("INSERT INTO chunks_fts(rowid, content) VALUES (?, 'test content')", (chunk_id,))
    conn.commit()
    row = conn.execute("SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'test'").fetchone()
    assert row is not None
    conn.close()


def test_chunks_vec_exists(db_path):
    """Can INSERT into chunks_vec with explicit rowid + float[256] embedding."""
    conn = get_connection(db_path)
    embedding = [0.1] * 256
    blob = struct.pack(f"{len(embedding)}f", *embedding)
    conn.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (1, ?)", (blob,))
    conn.commit()
    row = conn.execute("SELECT rowid FROM chunks_vec").fetchone()
    assert row is not None
    conn.close()


def test_get_connection_enables_foreign_keys(db_path):
    """get_connection() returns conn with FK enabled."""
    conn = get_connection(db_path)
    result = conn.execute("PRAGMA foreign_keys").fetchone()
    assert result[0] == 1
    conn.close()


def test_memory_meta_table_exists(db_path):
    """memory_meta table exists after init_db."""
    conn = get_connection(db_path)
    conn.execute("SELECT * FROM memory_meta LIMIT 1")
    conn.close()


def test_memory_meta_foreign_key(db_path):
    """memory_meta.file_id references files.id with CASCADE delete."""
    conn = get_connection(db_path)
    conn.execute("INSERT INTO files (file_path, file_hash) VALUES ('memory://test/abc', 'h1')")
    file_id = conn.execute("SELECT id FROM files WHERE file_path = 'memory://test/abc'").fetchone()["id"]
    conn.execute(
        "INSERT INTO memory_meta (file_id, namespace, source, metadata) VALUES (?, 'test', 'api', '{}')",
        (file_id,),
    )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM memory_meta").fetchone()["c"] == 1
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM memory_meta").fetchone()["c"] == 0
    conn.close()


def test_memory_meta_namespace_index(db_path):
    """idx_memory_ns index exists on memory_meta.namespace."""
    conn = get_connection(db_path)
    indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memory_meta'").fetchall()
    names = {row["name"] for row in indexes}
    conn.close()
    assert "idx_memory_ns" in names
