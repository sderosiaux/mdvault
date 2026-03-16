import sqlite3
import struct
from pathlib import Path

import sqlite_vec


def serialize_f32(vector) -> bytes:
    """Serialize a float vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    conn = get_connection(db_path)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS vault_config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            id         INTEGER PRIMARY KEY,
            file_path  TEXT NOT NULL UNIQUE,
            file_hash  TEXT NOT NULL,
            indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY,
            file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            chunk_idx   INTEGER NOT NULL,
            content     TEXT NOT NULL,
            raw_content TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);

        CREATE TABLE IF NOT EXISTS links (
            id             INTEGER PRIMARY KEY,
            source_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            target_path    TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_file_id);
        CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_path);

        CREATE TABLE IF NOT EXISTS memory_meta (
            file_id    INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
            namespace  TEXT NOT NULL DEFAULT '',
            source     TEXT NOT NULL DEFAULT 'api',
            metadata   TEXT NOT NULL DEFAULT '{}',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_memory_ns ON memory_meta(namespace);
    """)

    # FTS5 virtual table (external content)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            content='chunks',
            content_rowid='id'
        )
    """)

    # sqlite-vec virtual table
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            embedding FLOAT[256]
        )
    """)

    conn.commit()

    # Migrate older DBs that lack raw_content column
    cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    if "raw_content" not in cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN raw_content TEXT NOT NULL DEFAULT ''")
        conn.commit()

    # Re-enable foreign keys (executescript resets pragmas)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.close()
