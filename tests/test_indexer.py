import hashlib
import shutil
import pytest
from pathlib import Path
from mdvault.db import get_connection, init_db
from mdvault.indexer import (
    Chunk,
    chunk_file,
    compute_sha256,
    index_file,
    index_directory,
    incremental_index,
    _extract_links,
)
from tests.conftest import FIXTURES_DIR


# ---------- chunk_file tests ----------

def test_chunk_file_by_headings():
    """File with 3 ## sections -> 3 Chunk objects, each contains its heading."""
    content = (
        "# Title\n\nIntro paragraph here.\n\n"
        "## Section One\n\nContent of section one with enough words to be valid "
        "and this is filler text to exceed twenty words minimum for this chunk.\n\n"
        "## Section Two\n\nContent of section two with enough words to be valid "
        "and this is filler text to exceed twenty words minimum for this chunk.\n\n"
        "## Section Three\n\nContent of section three with enough words to be valid "
        "and this is filler text to exceed twenty words minimum for this chunk.\n"
    )
    chunks = chunk_file(content)
    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    # First chunk starts with its heading (no overlap prefix)
    assert chunks[0].content.startswith("## Section One")
    assert chunks[0].heading == "Section One"
    # Subsequent chunks have overlap prefix, but must contain their heading
    assert "## Section Two" in chunks[1].content
    assert chunks[1].heading == "Section Two"
    assert "## Section Three" in chunks[2].content
    assert chunks[2].heading == "Section Three"


def test_chunk_file_no_headings():
    """File without ## -> chunked by paragraph/word limit."""
    content = "This is a paragraph with enough words. " * 30
    chunks = chunk_file(content)
    assert len(chunks) >= 1


def test_chunk_max_400_words():
    """A single paragraph > 400 words -> split into multiple chunks."""
    content = "## Big Section\n\n" + ("word " * 900)
    chunks = chunk_file(content)
    for chunk in chunks:
        word_count = len(chunk.content.split())
        # Allow overlap words (50) on top of 400
        assert word_count <= 460, f"Chunk too long: {word_count} words"


def test_chunk_overlap_50_words():
    """Chunk N+1 starts with last 50 words of chunk N."""
    # Create content large enough that chunks have 50+ words each
    content = "## Big Section\n\n" + ("word " * 900)
    chunks = chunk_file(content)
    assert len(chunks) >= 2
    words_0 = chunks[0].content.split()
    last_50_of_0 = words_0[-50:]
    words_1 = chunks[1].content.split()
    first_50_of_1 = words_1[:50]
    assert last_50_of_0 == first_50_of_1


def test_chunk_minimum_20_words():
    """Tiny section < 20 words merged into previous chunk."""
    content = (
        "## Section One\n\n"
        "This section has enough words to be a valid chunk on its own and exceeds the minimum. " * 3 + "\n\n"
        "## Tiny\n\nJust a few words.\n"
    )
    chunks = chunk_file(content)
    # "Tiny" section (<20 words) should be merged, so we get 1 chunk not 2
    assert len(chunks) == 1
    assert "Just a few words." in chunks[0].content


# ---------- compute_sha256 ----------

def test_sha256_file(tmp_path):
    """Returns hex SHA256 of file content."""
    f = tmp_path / "test.md"
    f.write_text("hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert compute_sha256(f) == expected


# ---------- index_file ----------

def test_index_file_inserts_rows(db_path, mock_embedder):
    """After indexing one file: 1 row in files, N rows in chunks, same N in fts and vec."""
    nginx_path = FIXTURES_DIR / "infra" / "nginx.md"
    conn = get_connection(db_path)
    index_file(conn, nginx_path, FIXTURES_DIR, mock_embedder)
    conn.commit()

    file_count = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    chunk_count = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
    fts_count = conn.execute(
        "SELECT COUNT(*) as c FROM chunks_fts"
    ).fetchone()["c"]
    vec_count = conn.execute("SELECT COUNT(*) as c FROM chunks_vec").fetchone()["c"]
    conn.close()

    assert file_count == 1
    assert chunk_count > 0
    assert fts_count == chunk_count
    assert vec_count == chunk_count


def test_index_file_contextual_prefix(db_path, mock_embedder):
    """content has context prefix, raw_content has original text."""
    nginx_path = FIXTURES_DIR / "infra" / "nginx.md"
    conn = get_connection(db_path)
    index_file(conn, nginx_path, FIXTURES_DIR, mock_embedder)
    conn.commit()

    row = conn.execute("SELECT content, raw_content FROM chunks LIMIT 1").fetchone()
    conn.close()
    # content starts with context prefix [path > ...]
    assert row["content"].startswith("[infra/nginx.md")
    # raw_content does NOT have the prefix
    assert not row["raw_content"].startswith("[")


def test_index_file_fts_searchable_via_context(db_path, mock_embedder):
    """FTS5 can find chunks by file path thanks to context prefix."""
    nginx_path = FIXTURES_DIR / "infra" / "nginx.md"
    conn = get_connection(db_path)
    index_file(conn, nginx_path, FIXTURES_DIR, mock_embedder)
    conn.commit()

    # Search by path component — wouldn't work without context prefix
    rows = conn.execute(
        "SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'nginx'"
    ).fetchall()
    conn.close()
    assert len(rows) > 0


def test_index_file_fts_searchable(db_path, mock_embedder):
    """After indexing nginx.md, FTS5 query 'reverse proxy' returns a result."""
    nginx_path = FIXTURES_DIR / "infra" / "nginx.md"
    conn = get_connection(db_path)
    index_file(conn, nginx_path, FIXTURES_DIR, mock_embedder)
    conn.commit()

    rows = conn.execute(
        "SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'reverse proxy'"
    ).fetchall()
    conn.close()
    assert len(rows) > 0


# ---------- index_directory ----------

def test_index_directory_indexes_all_md_files(db_path, mock_embedder):
    """Indexing fixtures/ creates rows for all .md files."""
    conn = get_connection(db_path)
    index_directory(conn, FIXTURES_DIR, mock_embedder)
    conn.commit()

    file_count = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    md_files = list(FIXTURES_DIR.rglob("*.md"))
    conn.close()
    assert file_count == len(md_files)


def test_full_reindex_wipes_and_rebuilds(db_path, mock_embedder):
    """Index twice -> same count (no duplicates)."""
    conn = get_connection(db_path)
    index_directory(conn, FIXTURES_DIR, mock_embedder, full=True)
    conn.commit()
    count1 = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]

    index_directory(conn, FIXTURES_DIR, mock_embedder, full=True)
    conn.commit()
    count2 = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    conn.close()

    assert count1 == count2
    assert count1 > 0


# ---------- incremental_index ----------

def test_incremental_skip_unchanged_file(tmp_path, mock_embedder):
    """Index -> re-run incremental -> same hash -> file not reindexed (indexed_at unchanged)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    f = vault / "note.md"
    f.write_text("## Hello\n\nThis is a test note with enough words to pass the minimum chunk size requirement easily.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p, vault_root=str(vault))
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    ts1 = conn.execute("SELECT indexed_at FROM files").fetchone()["indexed_at"]

    # Incremental -- no changes
    incremental_index(conn, vault, mock_embedder)
    conn.commit()

    ts2 = conn.execute("SELECT indexed_at FROM files").fetchone()["indexed_at"]
    conn.close()
    assert ts1 == ts2


def test_incremental_reindex_modified_file(tmp_path, mock_embedder):
    """Index -> modify file content -> incremental -> chunks updated."""
    vault = tmp_path / "vault"
    vault.mkdir()
    f = vault / "note.md"
    f.write_text("## Original\n\nOriginal content that is long enough to form a valid chunk for indexing purposes here.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p, vault_root=str(vault))
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    old_content = conn.execute("SELECT raw_content FROM chunks LIMIT 1").fetchone()["raw_content"]

    # Modify file
    f.write_text("## Modified\n\nModified content that is long enough to form a valid chunk for indexing purposes here.\n")

    incremental_index(conn, vault, mock_embedder)
    conn.commit()

    new_content = conn.execute("SELECT raw_content FROM chunks LIMIT 1").fetchone()["raw_content"]
    conn.close()
    assert old_content != new_content


def test_incremental_removes_deleted_file(tmp_path, mock_embedder):
    """Index -> delete file -> incremental -> file+chunks removed from DB."""
    vault = tmp_path / "vault"
    vault.mkdir()
    f = vault / "note.md"
    f.write_text("## Test\n\nContent that is long enough to form a valid chunk for indexing purposes minimum words.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p, vault_root=str(vault))
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 1

    f.unlink()
    incremental_index(conn, vault, mock_embedder)
    conn.commit()

    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"] == 0
    conn.close()


def test_incremental_adds_new_file(tmp_path, mock_embedder):
    """Index -> add new .md -> incremental -> new file indexed."""
    vault = tmp_path / "vault"
    vault.mkdir()
    f1 = vault / "note1.md"
    f1.write_text("## First\n\nFirst note content with enough words to pass the minimum chunk threshold easily.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p, vault_root=str(vault))
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 1

    f2 = vault / "note2.md"
    f2.write_text("## Second\n\nSecond note content with enough words to pass the minimum chunk threshold easily.\n")

    incremental_index(conn, vault, mock_embedder)
    conn.commit()

    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 2
    conn.close()


def test_batch_processing_large_vault(tmp_path, mock_embedder):
    """Create 150 temp markdown files -> index -> all indexed correctly."""
    vault = tmp_path / "vault"
    vault.mkdir()
    for i in range(150):
        (vault / f"note_{i:03d}.md").write_text(
            f"## Note {i}\n\nThis is note number {i} with enough content words to satisfy the minimum chunk size requirement.\n"
        )

    db_p = tmp_path / "test.db"
    init_db(db_p, vault_root=str(vault))
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    file_count = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    conn.close()
    assert file_count == 150


# ---------- _extract_links tests ----------

def test_extract_links_standard_markdown():
    """Parses [text](path.md) links, resolves relative to source."""
    content = "See [config](../infra/nginx.md) and [guide](./setup.md) for details."
    links = _extract_links(content, "docs/howto.md")
    assert "infra/nginx.md" in links
    assert "docs/setup.md" in links


def test_extract_links_wikilinks():
    """Parses [[wikilink]] and [[wikilink|display]]."""
    content = "Check [[kubernetes]] and [[docker|Docker Guide]] for more."
    links = _extract_links(content, "notes/index.md")
    assert "kubernetes.md" in links
    assert "docker.md" in links


def test_extract_links_ignores_external_urls():
    """Skips http/https/mailto links."""
    content = "See [docs](https://example.com/foo.md) and [mail](mailto:x@y.md)."
    links = _extract_links(content, "notes/index.md")
    assert links == []


def test_extract_links_ignores_non_md():
    """Skips links to non-.md files."""
    content = "See [image](photo.png) and [pdf](doc.pdf)."
    links = _extract_links(content, "notes/index.md")
    assert links == []


def test_extract_links_removes_self_links():
    """Does not include self-referencing links."""
    content = "See [myself](index.md) and [[index]]."
    links = _extract_links(content, "index.md")
    assert links == []


def test_extract_links_strips_anchors():
    """Strips #anchor fragments from links."""
    content = "See [section](other.md#setup) for details."
    links = _extract_links(content, "notes/index.md")
    assert "notes/other.md" in links


def test_extract_links_wikilink_with_heading():
    """[[note#heading]] strips the anchor and resolves to note.md."""
    content = "See [[kubernetes#pods]] for details."
    links = _extract_links(content, "notes/index.md")
    assert "kubernetes.md" in links
    assert "kubernetes#pods.md" not in links


def test_extract_links_ignores_image_links():
    """![alt](file.md) image syntax is not parsed as a document link."""
    content = "![diagram](overview.md) and [real link](other.md)."
    links = _extract_links(content, "notes/index.md")
    assert "notes/other.md" in links
    assert "notes/overview.md" not in links


def test_extract_links_ignores_code_blocks():
    """Links inside fenced code blocks and inline code are skipped."""
    content = (
        "Real [link](real.md) here.\n\n"
        "```markdown\n[fake](fake.md)\n```\n\n"
        "And `[inline](inline.md)` code."
    )
    links = _extract_links(content, "index.md")
    assert "real.md" in links
    assert "fake.md" not in links
    assert "inline.md" not in links


def test_extract_links_with_title_attribute():
    """[text](path.md 'title') correctly extracts path.md."""
    content = 'See [guide](setup.md "Setup Guide") for details.'
    links = _extract_links(content, "index.md")
    assert "setup.md" in links


def test_extract_links_ignores_ftp_file_protocols():
    """ftp:// and file:// links are filtered out."""
    content = "See [ftp](ftp://server/notes.md) and [local](file:///tmp/notes.md)."
    links = _extract_links(content, "index.md")
    assert links == []


# ---------- links table integration ----------

def test_index_file_stores_links(tmp_path, mock_embedder):
    """Links extracted from markdown are stored in the links table."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text(
        "# Note A\n\n## Content\n\nSee [B](b.md) and [[c]] for reference with enough words here.\n"
    )
    (vault / "b.md").write_text(
        "# Note B\n\n## Content\n\nThis is note B with enough words to form a valid chunk.\n"
    )
    (vault / "c.md").write_text(
        "# Note C\n\n## Content\n\nThis is note C with enough words to form a valid chunk.\n"
    )

    db_p = tmp_path / "test.db"
    init_db(db_p, vault_root=str(vault))
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    links = [
        row["target_path"]
        for row in conn.execute(
            "SELECT target_path FROM links l JOIN files f ON f.id = l.source_file_id WHERE f.file_path = 'a.md'"
        ).fetchall()
    ]
    conn.close()
    assert "b.md" in links
    assert "c.md" in links


def test_links_removed_on_file_delete(tmp_path, mock_embedder):
    """When a file is deleted (incremental), its links are cascade-deleted."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text(
        "# Note A\n\n## Links\n\nSee [B](b.md) for reference with enough words to chunk.\n"
    )
    (vault / "b.md").write_text(
        "# Note B\n\n## Content\n\nEnough words to form a valid chunk for indexing purposes.\n"
    )

    db_p = tmp_path / "test.db"
    init_db(db_p, vault_root=str(vault))
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM links").fetchone()["c"] > 0

    (vault / "a.md").unlink()
    incremental_index(conn, vault, mock_embedder)
    conn.commit()

    # Links from a.md should be gone (CASCADE)
    assert conn.execute("SELECT COUNT(*) as c FROM links").fetchone()["c"] == 0
    conn.close()
