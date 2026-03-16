import hashlib
import posixpath
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
        non_title_lines = [line for line in preamble_lines if not line.startswith("# ")]
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
        final_chunks.extend((sc, heading) for sc in sub_chunks)

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
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def compute_sha256(file_path: Path) -> str:
    """Return hex SHA256 of file content."""
    content = file_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def _extract_links(content: str, source_rel_path: str) -> list[str]:
    """Extract link targets from markdown content. Returns target paths."""
    targets: set[str] = set()
    source_dir = posixpath.dirname(source_rel_path)

    # Strip fenced code blocks and inline code before parsing
    cleaned = re.sub(r"```[\s\S]*?```", "", content)
    cleaned = re.sub(r"`[^`]+`", "", cleaned)

    # Standard Markdown links (not images): [text](path.md) or [text](path.md#anchor)
    for m in re.finditer(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)", cleaned):
        href = m.group(2).split("#")[0].split()[0]
        if not href or "://" in href or href.startswith("mailto:"):
            continue
        if not href.endswith(".md"):
            continue
        resolved = posixpath.normpath(posixpath.join(source_dir, href))
        targets.add(resolved)

    # Wikilinks: [[target]] or [[target|display text]]
    for m in re.finditer(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", cleaned):
        name = m.group(1).strip().split("#")[0]
        if not name:
            continue
        if not name.endswith(".md"):
            name += ".md"
        targets.add(name)

    # Remove self-links
    targets.discard(source_rel_path)
    targets.discard(posixpath.basename(source_rel_path))

    return sorted(targets)


def index_file(
    conn: sqlite3.Connection,
    file_path: Path,
    vault_root: Path,
    embedder: Callable[[list[str]], np.ndarray],
) -> None:
    """Index a single markdown file: insert into files, chunks, chunks_fts, chunks_vec."""
    rel_path = str(file_path.relative_to(vault_root))

    # Single read: hash + decode
    raw = file_path.read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return  # skip non-UTF-8 files

    chunks = chunk_file(content)
    if not chunks:
        return

    title = _extract_title(content)

    # Build contextualized texts for embedding and FTS
    ctx_texts = []
    for chunk in chunks:
        prefix = _context_prefix(rel_path, title, chunk.heading)
        ctx_texts.append(prefix + chunk.content)

    # Embed before DB writes (most likely failure point)
    all_embeddings = []
    for batch_start in range(0, len(ctx_texts), 32):
        batch = ctx_texts[batch_start : batch_start + 32]
        batch_embeddings = embedder(batch)
        all_embeddings.append(batch_embeddings)
    embeddings = np.concatenate(all_embeddings, axis=0)

    # All DB writes in a savepoint for atomicity
    conn.execute("SAVEPOINT sp_index_file")
    try:
        conn.execute(
            "INSERT INTO files (file_path, file_hash) VALUES (?, ?)",
            (rel_path, file_hash),
        )
        file_id = conn.execute("SELECT id FROM files WHERE file_path = ?", (rel_path,)).fetchone()["id"]

        for idx, (chunk, ctx_text, embedding) in enumerate(zip(chunks, ctx_texts, embeddings, strict=True)):
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

        # Extract and store links
        link_targets = _extract_links(content, rel_path)
        for target_path in link_targets:
            conn.execute(
                "INSERT INTO links (source_file_id, target_path) VALUES (?, ?)",
                (file_id, target_path),
            )

        conn.execute("RELEASE SAVEPOINT sp_index_file")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT sp_index_file")
        raise


def _remove_file(conn: sqlite3.Connection, file_path: str) -> None:
    """Remove a file and its chunks from all tables (FTS/vec via subquery, then cascade)."""
    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN ("
        "  SELECT c.id FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.file_path = ?"
        ")",
        (file_path,),
    )
    conn.execute(
        "DELETE FROM chunks_vec WHERE rowid IN ("
        "  SELECT c.id FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.file_path = ?"
        ")",
        (file_path,),
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
        # Wipe existing data via subqueries (no variable limit)
        conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks)")
        conn.execute("DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks)")
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
        row["file_path"]: row["file_hash"] for row in conn.execute("SELECT file_path, file_hash FROM files").fetchall()
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
