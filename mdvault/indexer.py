import fnmatch
import hashlib
import json
import posixpath
import re
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mdvault.db import serialize_f32


@dataclass
class Chunk:
    content: str
    heading: str | None = None
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


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

    # No overlap — each chunk is independent for cleaner scoring

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


_INDEXABLE_EXTENSIONS = {".md", ".jsonl"}


def _list_files(vault_root: Path, no_gitignore: bool = False) -> list[Path]:
    """List indexable files (.md, .jsonl) under vault_root, respecting .gitignore if in a git repo."""
    if no_gitignore:
        files = []
        for ext in _INDEXABLE_EXTENSIONS:
            files.extend(vault_root.rglob(f"*{ext}"))
        return sorted(files)
    try:
        git_args = [
            "git",
            "-C",
            str(vault_root),
            "-c",
            "core.quotepath=false",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ]
        patterns = [f"*{ext}" for ext in _INDEXABLE_EXTENSIONS]
        result = subprocess.run(
            [*git_args, *patterns],
            capture_output=True,
            text=True,
            check=True,
        )
        return sorted(vault_root / line for line in result.stdout.splitlines() if line)
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = []
        for ext in _INDEXABLE_EXTENSIONS:
            files.extend(vault_root.rglob(f"*{ext}"))
        return sorted(files)


def _extract_jsonl_chunks(raw: bytes, session_id: str) -> list[Chunk]:
    """Extract structured chunks from a Claude Code session JSONL file.

    One chunk per message turn (user or assistant). Skips thinking/tool_use/progress.
    Each chunk carries metadata: role, message_id, model, session_id.
    """
    chunks: list[Chunk] = []
    for raw_line in raw.split(b"\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):  # noqa: S112
            continue

        msg_type = rec.get("type", "")
        msg = rec.get("message", {})
        if not isinstance(msg, dict):
            continue

        if msg_type == "user":
            content = msg.get("content", "")
            text = ""
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text", "").strip()
                        if t:
                            parts.append(t)
                text = "\n".join(parts)
            if text:
                chunks.append(
                    Chunk(
                        content=text,
                        heading="User",
                        metadata={
                            "role": "user",
                            "message_id": msg.get("id", ""),
                            "session_id": session_id,
                        },
                    )
                )

        elif msg_type == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                texts = [
                    block.get("text", "").strip()
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                text = "\n".join(t for t in texts if t)
                if text:
                    chunks.append(
                        Chunk(
                            content=text,
                            heading="Assistant",
                            metadata={
                                "role": "assistant",
                                "message_id": msg.get("id", ""),
                                "model": msg.get("model", ""),
                                "session_id": session_id,
                            },
                        )
                    )

    return chunks


def _parse_mdvault_search_output(output: str) -> list[str]:
    """Extract file_path:chunk_idx identifiers from mdvault search CLI output.

    mdvault search output format:
      [1] 0.983  vault/path/to/file.md:2
      content snippet...
    """
    results = []
    for line in output.splitlines():
        m = re.match(r"\[\d+\]\s+[\d.]+\s+(\S+:\d+)", line.strip())
        if m:
            results.append(m.group(1))
    return results


def analyze_session_feedback(
    conn: sqlite3.Connection,
    raw: bytes,
    session_id: str,
    embedder,
) -> None:
    """Detect mdvault search CLI calls in a JSONL session and score chunk utility.

    Looks for: assistant Bash tool_use with 'mdvault search' → user tool_result (stdout)
    → next assistant text. Computes cosine similarity between assistant response and
    retrieved chunks. Stores signals in chunk_feedback.
    """
    import numpy as np

    records = []
    for raw_line in raw.split(b"\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except (json.JSONDecodeError, ValueError):  # noqa: S112
            continue

    # Index tool_use id → search results (file_path:chunk_idx strings)
    pending: dict[str, list[str]] = {}  # tool_use_id → [file:idx, ...]

    for rec in records:
        msg_type = rec.get("type", "")
        msg = rec.get("message", {})
        if not isinstance(msg, dict):
            continue

        # Detect Bash tool_use containing "mdvault search"
        if msg_type == "assistant":
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") == "Bash":
                    cmd = block.get("input", {}).get("command", "")
                    if "mdvault" in cmd and "search" in cmd:
                        pending[block["id"]] = []

        # Collect tool_result for pending tool_use ids
        elif msg_type == "user":
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    tool_id = block.get("tool_use_id", "")
                    if tool_id in pending:
                        output = ""
                        c = block.get("content", "")
                        if isinstance(c, str):
                            output = c
                        elif isinstance(c, list):
                            output = "\n".join(
                                b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
                            )
                        pending[tool_id] = _parse_mdvault_search_output(output)

            # After collecting results, look for next assistant text to score
            # (we process this as we find the next assistant message below)

        # Next assistant text after a completed tool_result
        elif msg_type == "assistant" and pending:
            texts = [
                block.get("text", "").strip()
                for block in msg.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            response_text = "\n".join(t for t in texts if t)
            if not response_text:
                continue

            # Score against all pending search results that have been resolved
            completed = {tid: paths for tid, paths in pending.items() if paths}
            if not completed:
                continue

            response_vec = embedder([response_text])[0]

            for paths in completed.values():
                for path_idx in paths:
                    # Resolve chunk in DB: file_path and chunk_idx
                    parts = path_idx.rsplit(":", 1)
                    if len(parts) != 2:
                        continue
                    file_path, chunk_idx_str = parts
                    if not chunk_idx_str.isdigit():
                        continue
                    chunk_idx = int(chunk_idx_str)

                    row = conn.execute(
                        """
                        SELECT c.id, c.raw_content FROM chunks c
                        JOIN files f ON c.file_id = f.id
                        WHERE f.file_path = ? AND c.chunk_idx = ?
                        """,
                        (file_path, chunk_idx),
                    ).fetchone()
                    if not row:
                        continue

                    chunk_vec = embedder([row["raw_content"]])[0]
                    sim = float(
                        np.dot(response_vec, chunk_vec)
                        / (np.linalg.norm(response_vec) * np.linalg.norm(chunk_vec) + 1e-9)
                    )

                    if sim > 0.7:
                        conn.execute(
                            "INSERT INTO chunk_feedback (chunk_id, score, session_id) VALUES (?, ?, ?)",
                            (row["id"], sim, session_id),
                        )

            # Clear completed entries from pending
            for tid in list(completed.keys()):
                del pending[tid]


def _extract_links(content: str, source_rel_path: str) -> list[str]:
    """Extract link targets from markdown content. Returns target paths."""
    targets: set[str] = set()
    source_dir = posixpath.dirname(source_rel_path)

    # Strip fenced code blocks and inline code before parsing
    cleaned = re.sub(r"```[\s\S]*?```", "", content)
    cleaned = re.sub(r"`[^`]+`", "", cleaned)

    # Standard Markdown links (not images): [text](path.md) or [text](path.md#anchor)
    for m in re.finditer(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)", cleaned):
        href_parts = m.group(2).split("#")[0].split()
        if not href_parts:
            continue
        href = href_parts[0]
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
    raw: bytes | None = None,
) -> bool:
    """Index a single file (.md or .jsonl): insert into files, chunks, chunks_fts, chunks_vec.

    Returns True when a row was inserted, False when the file was skipped
    (unreadable, non-UTF-8, no chunks). Pass ``raw`` to reuse already-read
    bytes and avoid a redundant disk read.
    """
    rel_path = f"{vault_root.name}/{file_path.relative_to(vault_root)}"

    if raw is None:
        try:
            raw = file_path.read_bytes()
        except (FileNotFoundError, OSError):
            return False  # skip broken symlinks or unreadable files
    file_hash = hashlib.sha256(raw).hexdigest()

    if file_path.suffix == ".jsonl":
        session_id = file_path.stem
        chunks = _extract_jsonl_chunks(raw, session_id)
        chunks = [c for c in chunks if c.content.strip()]
        if not chunks:
            return False
        content = "\n\n".join(f"## {c.heading}\n{c.content}" for c in chunks)
        title = None
    else:
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return False  # skip non-UTF-8 files
        chunks = chunk_file(content)
        chunks = [c for c in chunks if c.content.strip()]
        if not chunks:
            return False
        title = _extract_title(content)

    # Build contextualized texts for FTS (with path/title prefix)
    ctx_texts = []
    raw_texts = []
    for chunk in chunks:
        prefix = _context_prefix(rel_path, title, chunk.heading)
        ctx_texts.append(prefix + chunk.content)
        raw_texts.append(chunk.content)

    # Embed raw_content (without prefix) — prefix hurts vector similarity
    all_embeddings = []
    for batch_start in range(0, len(raw_texts), 32):
        batch = raw_texts[batch_start : batch_start + 32]
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
                "INSERT INTO chunks (file_id, chunk_idx, content, raw_content, metadata) VALUES (?, ?, ?, ?, ?)",
                (file_id, idx, ctx_text, chunk.content, json.dumps(chunk.metadata)),
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

    # Post-index feedback analysis for JSONL sessions (best-effort).
    # No explicit commit here: the caller's batch loop commits, and an inner
    # commit would release outer savepoints used by index_directory's
    # atomic replace path.
    if file_path.suffix == ".jsonl":
        import contextlib

        with contextlib.suppress(Exception):
            analyze_session_feedback(conn, raw, file_path.stem, embedder)

    return True


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


def _remove_vault_files(
    conn: sqlite3.Connection,
    vault_name: str,
    keep_patterns: list[str] | None = None,
) -> None:
    """Remove all files (and their chunks/links) belonging to a vault prefix.

    When ``keep_patterns`` is provided, rows whose path matches any pattern
    are preserved — this lets ``--full`` rebuild from disk while keeping
    transient-source entries (e.g. rotated session logs) intact.
    """
    pattern = f"{vault_name}/%"
    if not keep_patterns:
        conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN ("
            "  SELECT c.id FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.file_path LIKE ?"
            ")",
            (pattern,),
        )
        conn.execute(
            "DELETE FROM chunks_vec WHERE rowid IN ("
            "  SELECT c.id FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.file_path LIKE ?"
            ")",
            (pattern,),
        )
        conn.execute("DELETE FROM files WHERE file_path LIKE ?", (pattern,))
        return

    rows = conn.execute(
        "SELECT file_path FROM files WHERE file_path LIKE ?",
        (pattern,),
    ).fetchall()
    for row in rows:
        rel_in_vault = row["file_path"].split("/", 1)[1] if "/" in row["file_path"] else row["file_path"]
        if not _matches_keep_deleted(rel_in_vault, keep_patterns):
            _remove_file(conn, row["file_path"])


def _matches_keep_deleted(rel_in_vault: str, patterns: list[str]) -> bool:
    """True if a path matches any keep-deleted pattern.

    Pattern semantics:
    - Bare prefix (e.g. ``projects`` or ``projects/``) matches anything under that directory.
    - Trailing ``*`` / ``**`` is stripped, then prefix-matched.
    - Otherwise, ``fnmatch`` glob (e.g. ``*.jsonl``).
    """
    for raw in patterns:
        p = raw.strip()
        if not p:
            continue
        prefix = p.rstrip("*").rstrip("/")
        if prefix and (rel_in_vault == prefix or rel_in_vault.startswith(prefix + "/")):
            return True
        if fnmatch.fnmatchcase(rel_in_vault, p):
            return True
    return False


def index_directory(
    conn: sqlite3.Connection,
    vault_root: Path,
    embedder: Callable[[list[str]], np.ndarray],
    full: bool = False,
    no_gitignore: bool = False,
    keep_deleted: list[str] | None = None,
) -> None:
    """Index all .md files under vault_root. Additive by default (per-vault incremental)."""
    if not vault_root.exists():
        raise ValueError(f"Vault path does not exist: {vault_root}")
    if not vault_root.is_dir():
        raise ValueError(f"Vault path is not a directory: {vault_root}")

    vault_name = vault_root.name
    resolved = str(vault_root.resolve())

    # Check for vault name collision with different absolute path
    existing = conn.execute(
        "SELECT value FROM vault_config WHERE key = ?",
        (f"vault_root:{vault_name}",),
    ).fetchone()
    if existing and existing["value"] != resolved:
        msg = (
            f"Vault name '{vault_name}' already used by {existing['value']}. Use --db to specify a different database."
        )
        raise ValueError(msg)

    # Register vault root and options
    conn.execute(
        "INSERT OR REPLACE INTO vault_config (key, value) VALUES (?, ?)",
        (f"vault_root:{vault_name}", resolved),
    )
    conn.execute(
        "INSERT OR REPLACE INTO vault_config (key, value) VALUES (?, ?)",
        (f"vault_opts:{vault_name}", "no_gitignore" if no_gitignore else ""),
    )
    keep_patterns = [p for p in (keep_deleted or []) if p.strip()]
    conn.execute(
        "INSERT OR REPLACE INTO vault_config (key, value) VALUES (?, ?)",
        (f"keep_deleted:{vault_name}", json.dumps(keep_patterns)),
    )

    md_files = _list_files(vault_root, no_gitignore=no_gitignore)

    if full:
        # Honor keep_deleted on --full so transient-source entries survive
        # a forced rebuild. To wipe everything, call without keep_deleted.
        _remove_vault_files(conn, vault_name, keep_patterns=keep_patterns)
        conn.commit()

    # Build set of expected disk paths (prefixed)
    disk_paths = {f"{vault_name}/{f.relative_to(vault_root)}" for f in md_files}

    # Get DB state for this vault (empty after full wipe)
    db_files = {
        row["file_path"]: row["file_hash"]
        for row in conn.execute(
            "SELECT file_path, file_hash FROM files WHERE file_path LIKE ?",
            (f"{vault_name}/%",),
        ).fetchall()
    }

    # Prune files that disappeared from disk, unless they match a keep-deleted
    # pattern. Use case: ~/.claude/projects/*.jsonl session logs are rotated
    # away by Claude Code, but we want their indexed content to remain
    # searchable. Skills/commands/knowledge are user-owned and prune normally.
    for deleted_path in set(db_files.keys()) - disk_paths:
        rel_in_vault = deleted_path.split("/", 1)[1] if "/" in deleted_path else deleted_path
        if not _matches_keep_deleted(rel_in_vault, keep_patterns):
            _remove_file(conn, deleted_path)

    total = len(md_files)
    show_progress = sys.stderr.isatty() and total > 50

    for batch_start in range(0, total, 100):
        batch = md_files[batch_start : batch_start + 100]
        for fp in batch:
            rel_path = f"{vault_name}/{fp.relative_to(vault_root)}"
            rel_in_vault = str(fp.relative_to(vault_root))
            if rel_path not in db_files:
                index_file(conn, fp, vault_root, embedder)
                continue

            # Hash the on-disk file. If the read fails, the existing DB row
            # is preserved (gated by keep_deleted, so non-transient stale
            # rows still get pruned).
            try:
                raw = fp.read_bytes()
            except (FileNotFoundError, OSError):
                if not _matches_keep_deleted(rel_in_vault, keep_patterns):
                    _remove_file(conn, rel_path)
                continue

            if hashlib.sha256(raw).hexdigest() == db_files[rel_path]:
                continue

            # Replace atomically: only delete the old row once we've read the
            # new bytes successfully. The savepoint guards the swap so an
            # embedder failure mid-replace doesn't leave the file unindexed.
            conn.execute("SAVEPOINT sp_replace")
            try:
                _remove_file(conn, rel_path)
                index_file(conn, fp, vault_root, embedder, raw=raw)
                conn.execute("RELEASE SAVEPOINT sp_replace")
            except Exception:
                conn.execute("ROLLBACK TO SAVEPOINT sp_replace")
                raise
        conn.commit()
        if show_progress:
            done = min(batch_start + len(batch), total)
            sys.stderr.write(f"\r  {done:,}/{total:,} files indexed")
            sys.stderr.flush()

    if show_progress:
        sys.stderr.write("\n")
