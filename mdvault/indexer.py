import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from mdvault.db import serialize_f32


@dataclass
class Chunk:
    content: str
    heading: str | None = None


def _extract_title(content: str) -> str | None:
    """Extract the first # heading as document title."""
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].rstrip("#").strip()
    return None


def _context_prefix(rel_path: str, title: str | None, heading: str | None) -> str:
    """Build context prefix like [path > title > heading]."""
    parts = [rel_path]
    if title:
        parts.append(title)
    if heading and heading != title:
        parts.append(heading)
    return f"[{' > '.join(parts)}]\n"


def chunk_file(content: str) -> list[Chunk]:
    """Split markdown content into chunks by ## headings, with overlap and merging."""
    heading_pattern = re.compile(r"^(#{2,3}\s)", re.MULTILINE)
    parts = heading_pattern.split(content)

    # Reassemble into blocks: (text, heading_text)
    blocks: list[tuple[str, str | None]] = []
    preamble = parts[0].strip()
    if preamble:
        preamble_lines = preamble.split("\n")
        non_title_lines = [l for l in preamble_lines if not l.startswith("# ")]
        preamble_text = "\n".join(non_title_lines).strip()
        if preamble_text and len(preamble_text.split()) >= 20:
            blocks.append((preamble_text, None))
    i = 1
    while i < len(parts):
        if heading_pattern.match(parts[i]):
            heading_marker = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ""
            block_text = heading_marker + body.strip()
            heading_text = block_text.split("\n", 1)[0].lstrip("#").strip()
            blocks.append((block_text, heading_text))
            i += 2
        else:
            i += 1

    if not blocks:
        blocks = [(content.strip(), None)]

    # Split oversized blocks (preserve heading association)
    final_chunks: list[tuple[str, str | None]] = []
    for text, heading in blocks:
        sub_chunks = _split_oversized(text, max_words=400)
        for sc in sub_chunks:
            final_chunks.append((sc, heading))

    # Merge tiny chunks (<20 words) into previous
    merged: list[tuple[str, str | None]] = []
    for text, heading in final_chunks:
        if len(text.split()) < 20 and merged:
            prev_text, prev_heading = merged[-1]
            merged[-1] = (prev_text + "\n\n" + text, prev_heading)
        else:
            merged.append((text, heading))

    if len(merged) > 1 and len(merged[-1][0].split()) < 20:
        prev_text, prev_heading = merged[-2]
        last_text, _ = merged[-1]
        merged[-2] = (prev_text + "\n\n" + last_text, prev_heading)
        merged.pop()

    # Apply overlap between chunks
    if len(merged) > 1:
        overlapped = [merged[0]]
        for i in range(1, len(merged)):
            prev_words = merged[i - 1][0].split()
            overlap_words = prev_words[-50:] if len(prev_words) >= 50 else prev_words
            overlap_text = " ".join(overlap_words)
            current_text, current_heading = merged[i]
            overlapped.append((overlap_text + " " + current_text, current_heading))
        merged = overlapped

    return [Chunk(content=text, heading=heading) for text, heading in merged]


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

    title = _extract_title(content)

    conn.execute(
        "INSERT INTO files (file_path, file_hash) VALUES (?, ?)",
        (rel_path, file_hash),
    )
    file_id = conn.execute(
        "SELECT id FROM files WHERE file_path = ?", (rel_path,)
    ).fetchone()["id"]

    # Build contextualized texts for embedding and FTS
    ctx_texts = []
    for chunk in chunks:
        prefix = _context_prefix(rel_path, title, chunk.heading)
        ctx_texts.append(prefix + chunk.content)

    # Embed in batches of 32
    all_embeddings = []
    for batch_start in range(0, len(ctx_texts), 32):
        batch = ctx_texts[batch_start : batch_start + 32]
        batch_embeddings = embedder(batch)
        all_embeddings.append(batch_embeddings)
    embeddings = np.concatenate(all_embeddings, axis=0)

    for idx, (chunk, ctx_text, embedding) in enumerate(zip(chunks, ctx_texts, embeddings)):
        cursor = conn.execute(
            "INSERT INTO chunks (file_id, chunk_idx, content, raw_content) VALUES (?, ?, ?, ?)",
            (file_id, idx, ctx_text, chunk.content),
        )
        chunk_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
            (chunk_id, ctx_text),
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
