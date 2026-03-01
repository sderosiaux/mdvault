import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np

from mdvault.db import serialize_f32


def chunk_file(content: str) -> list[str]:
    """Split markdown content into chunks by ## headings, with overlap and merging."""
    # Split by ## or ### headings
    heading_pattern = re.compile(r"^(#{2,3}\s)", re.MULTILINE)
    parts = heading_pattern.split(content)

    # Reassemble into blocks: before first heading is preamble, then heading+content pairs
    blocks: list[str] = []
    i = 0
    # parts[0] is text before first heading match
    preamble = parts[0].strip()
    if preamble:
        # Only keep preamble if it has substance (skip top-level # Title lines)
        preamble_lines = preamble.split("\n")
        non_title_lines = [l for l in preamble_lines if not l.startswith("# ")]
        preamble_text = "\n".join(non_title_lines).strip()
        if preamble_text and len(preamble_text.split()) >= 20:
            blocks.append(preamble_text)
    i = 1
    while i < len(parts):
        if heading_pattern.match(parts[i]):
            heading_marker = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ""
            blocks.append(heading_marker + body.strip())
            i += 2
        else:
            i += 1

    # If no headings found, treat entire content as one block
    if not blocks:
        blocks = [content.strip()]

    # Now split oversized blocks
    final_chunks: list[str] = []
    for block in blocks:
        sub_chunks = _split_oversized(block, max_words=400)
        final_chunks.extend(sub_chunks)

    # Merge tiny chunks (<20 words) into previous BEFORE overlap
    merged: list[str] = []
    for chunk in final_chunks:
        if len(chunk.split()) < 20 and merged:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)

    # Final check: if last chunk is too small, merge it
    if len(merged) > 1 and len(merged[-1].split()) < 20:
        merged[-2] = merged[-2] + "\n\n" + merged[-1]
        merged.pop()

    # Apply overlap between chunks AFTER merge
    if len(merged) > 1:
        overlapped = [merged[0]]
        for i in range(1, len(merged)):
            prev_words = merged[i - 1].split()
            overlap_words = prev_words[-50:] if len(prev_words) >= 50 else prev_words
            overlap_text = " ".join(overlap_words)
            current = merged[i]
            overlapped.append(overlap_text + " " + current)
        merged = overlapped

    return merged


def _split_oversized(block: str, max_words: int = 400) -> list[str]:
    """Split a block that exceeds max_words by paragraphs, then hard-split."""
    words = block.split()
    if len(words) <= max_words:
        return [block]

    # Try splitting by paragraphs first
    paragraphs = re.split(r"\n\n+", block)
    # Keep heading paragraph attached to the next paragraph
    if len(paragraphs) > 1 and paragraphs[0].startswith("##") and len(paragraphs[0].split()) < 20:
        paragraphs[0] = paragraphs[0] + "\n\n" + paragraphs[1]
        paragraphs.pop(1)
    if len(paragraphs) > 1:
        result: list[str] = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            para_words = para.split()
            if current_len + len(para_words) > max_words and current:
                result.append("\n\n".join(current))
                current = []
                current_len = 0
            # If single paragraph is still oversized, hard-split it
            if len(para_words) > max_words:
                if current:
                    result.append("\n\n".join(current))
                    current = []
                    current_len = 0
                result.extend(_hard_split(para, max_words))
            else:
                current.append(para)
                current_len += len(para_words)
        if current:
            result.append("\n\n".join(current))
        return result

    # Single paragraph -- hard split
    return _hard_split(block, max_words)


def _hard_split(text: str, max_words: int = 400) -> list[str]:
    """Hard-split text at word boundaries."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def compute_sha256(file_path: Path) -> str:
    """Return hex SHA256 of file content."""
    content = file_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def index_file(
    conn: sqlite3.Connection,
    file_path: Path,
    vault_root: Path,
    embedder: Callable[[list[str]], np.ndarray],
) -> None:
    """Index a single markdown file: insert into files, chunks, chunks_fts, chunks_vec."""
    rel_path = str(file_path.relative_to(vault_root))
    file_hash = compute_sha256(file_path)
    content = file_path.read_text(encoding="utf-8")
    chunks = chunk_file(content)

    if not chunks:
        return

    conn.execute(
        "INSERT INTO files (file_path, file_hash) VALUES (?, ?)",
        (rel_path, file_hash),
    )
    file_id = conn.execute(
        "SELECT id FROM files WHERE file_path = ?", (rel_path,)
    ).fetchone()["id"]

    # Embed in batches of 32
    all_embeddings = []
    for batch_start in range(0, len(chunks), 32):
        batch = chunks[batch_start : batch_start + 32]
        batch_embeddings = embedder(batch)
        all_embeddings.append(batch_embeddings)
    embeddings = np.concatenate(all_embeddings, axis=0)

    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        conn.execute(
            "INSERT INTO chunks (file_id, chunk_idx, content) VALUES (?, ?, ?)",
            (file_id, idx, chunk_text),
        )
        chunk_id = conn.execute(
            "SELECT last_insert_rowid() as id"
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
            (chunk_id, chunk_text),
        )
        conn.execute(
            "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
            (chunk_id, serialize_f32(embedding)),
        )


def _remove_file(conn: sqlite3.Connection, file_path: str) -> None:
    """Remove a file and its chunks from all tables (FTS/vec first, then cascade)."""
    chunk_ids = [
        row["id"]
        for row in conn.execute(
            "SELECT c.id FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.file_path = ?",
            (file_path,),
        ).fetchall()
    ]
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(
            f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", chunk_ids
        )
        conn.execute(
            f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})", chunk_ids
        )
    conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))


def index_directory(
    conn: sqlite3.Connection,
    vault_root: Path,
    embedder: Callable[[list[str]], np.ndarray],
    full: bool = True,
) -> None:
    """Index all .md files under vault_root."""
    if not vault_root.exists():
        raise ValueError(f"Vault path does not exist: {vault_root}")
    if not vault_root.is_dir():
        raise ValueError(f"Vault path is not a directory: {vault_root}")
    md_files = sorted(vault_root.rglob("*.md"))

    if full:
        # Wipe existing data
        # Get all existing chunk ids for FTS/vec cleanup
        existing_chunks = [
            row["id"] for row in conn.execute("SELECT id FROM chunks").fetchall()
        ]
        if existing_chunks:
            placeholders = ",".join("?" * len(existing_chunks))
            conn.execute(
                f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})",
                existing_chunks,
            )
            conn.execute(
                f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})",
                existing_chunks,
            )
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM files")
        conn.commit()

    # Process in batches of 100
    for batch_start in range(0, len(md_files), 100):
        batch = md_files[batch_start : batch_start + 100]
        for file_path in batch:
            index_file(conn, file_path, vault_root, embedder)
        conn.commit()


def incremental_index(
    conn: sqlite3.Connection,
    vault_root: Path,
    embedder: Callable[[list[str]], np.ndarray],
) -> None:
    """Incremental index: skip unchanged, update modified, add new, remove deleted."""
    disk_files = sorted(vault_root.rglob("*.md"))
    disk_paths = {str(f.relative_to(vault_root)) for f in disk_files}

    # Get DB state
    db_files = {
        row["file_path"]: row["file_hash"]
        for row in conn.execute("SELECT file_path, file_hash FROM files").fetchall()
    }
    db_paths = set(db_files.keys())

    # Deleted files: in DB but not on disk
    for deleted_path in db_paths - disk_paths:
        _remove_file(conn, deleted_path)

    # New or modified files
    for file_path in disk_files:
        rel_path = str(file_path.relative_to(vault_root))
        current_hash = compute_sha256(file_path)

        if rel_path not in db_files:
            # New file
            index_file(conn, file_path, vault_root, embedder)
        elif db_files[rel_path] != current_hash:
            # Modified file -- remove old, re-index
            _remove_file(conn, rel_path)
            index_file(conn, file_path, vault_root, embedder)
        # else: unchanged, skip

    conn.commit()
