import hashlib

from mdvault.db import get_connection, init_db
from mdvault.indexer import (
    Chunk,
    _extract_links,
    _list_files,
    chunk_file,
    compute_sha256,
    index_directory,
    index_file,
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
    fts_count = conn.execute("SELECT COUNT(*) as c FROM chunks_fts").fetchone()["c"]
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
    # content starts with context prefix [vault_name/path > ...]
    assert row["content"].startswith("[fixtures/infra/nginx.md")
    # raw_content does NOT have the prefix
    assert not row["raw_content"].startswith("[")


def test_index_file_fts_searchable_via_context(db_path, mock_embedder):
    """FTS5 can find chunks by file path thanks to context prefix."""
    nginx_path = FIXTURES_DIR / "infra" / "nginx.md"
    conn = get_connection(db_path)
    index_file(conn, nginx_path, FIXTURES_DIR, mock_embedder)
    conn.commit()

    # Search by path component — wouldn't work without context prefix
    rows = conn.execute("SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'nginx'").fetchall()
    conn.close()
    assert len(rows) > 0


def test_index_file_fts_searchable(db_path, mock_embedder):
    """After indexing nginx.md, FTS5 query 'reverse proxy' returns a result."""
    nginx_path = FIXTURES_DIR / "infra" / "nginx.md"
    conn = get_connection(db_path)
    index_file(conn, nginx_path, FIXTURES_DIR, mock_embedder)
    conn.commit()

    rows = conn.execute("SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'reverse proxy'").fetchall()
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


def test_additive_reindex_no_duplicates(db_path, mock_embedder):
    """Indexing same vault twice (additive) -> same count (no duplicates)."""
    conn = get_connection(db_path)
    index_directory(conn, FIXTURES_DIR, mock_embedder)
    conn.commit()
    count1 = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]

    index_directory(conn, FIXTURES_DIR, mock_embedder)
    conn.commit()
    count2 = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    conn.close()

    assert count1 == count2
    assert count1 > 0


def test_full_reindex_wipes_vault_and_rebuilds(db_path, mock_embedder):
    """full=True wipes this vault's files, then re-indexes."""
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


# ---------- additive indexing ----------


def test_additive_skip_unchanged_file(tmp_path, mock_embedder):
    """Index -> re-run additive -> same hash -> file not reindexed (indexed_at unchanged)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    f = vault / "note.md"
    f.write_text("## Hello\n\nThis is a test note with enough words to pass the chunk size requirement.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    ts1 = conn.execute("SELECT indexed_at FROM files").fetchone()["indexed_at"]

    # Additive re-index -- no changes
    index_directory(conn, vault, mock_embedder)
    conn.commit()

    ts2 = conn.execute("SELECT indexed_at FROM files").fetchone()["indexed_at"]
    conn.close()
    assert ts1 == ts2


def test_additive_reindex_modified_file(tmp_path, mock_embedder):
    """Index -> modify file content -> additive -> chunks updated."""
    vault = tmp_path / "vault"
    vault.mkdir()
    f = vault / "note.md"
    f.write_text("## Original\n\nOriginal content long enough to form a valid chunk for indexing.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    old_content = conn.execute("SELECT raw_content FROM chunks LIMIT 1").fetchone()["raw_content"]

    # Modify file
    f.write_text("## Modified\n\nModified content long enough to form a valid chunk for indexing.\n")

    index_directory(conn, vault, mock_embedder)
    conn.commit()

    new_content = conn.execute("SELECT raw_content FROM chunks LIMIT 1").fetchone()["raw_content"]
    conn.close()
    assert old_content != new_content


def test_additive_removes_deleted_file(tmp_path, mock_embedder):
    """Index -> delete file -> additive -> file+chunks removed from DB."""
    vault = tmp_path / "vault"
    vault.mkdir()
    f = vault / "note.md"
    f.write_text("## Test\n\nContent that is long enough to form a valid chunk for indexing purposes minimum words.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 1

    f.unlink()
    index_directory(conn, vault, mock_embedder)
    conn.commit()

    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"] == 0
    conn.close()


def test_keep_deleted_retains_pruned_file(tmp_path, mock_embedder):
    """Files matching --keep-deleted survive disappearance from disk."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "projects").mkdir()
    transient = vault / "projects" / "session.md"
    transient.write_text("## Session\n\nSession log content with enough words to form a valid indexable chunk.\n")
    persistent = vault / "skills" / "skill.md"
    persistent.parent.mkdir()
    persistent.write_text("## Skill\n\nSkill body with enough words to form a valid indexable chunk for testing.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True, keep_deleted=["projects"])
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 2

    # Both files vanish; only the non-matching one should be pruned.
    transient.unlink()
    persistent.unlink()
    index_directory(conn, vault, mock_embedder, keep_deleted=["projects"])
    conn.commit()

    paths = {row["file_path"] for row in conn.execute("SELECT file_path FROM files").fetchall()}
    assert paths == {"vault/projects/session.md"}
    conn.close()


def test_keep_deleted_pattern_matching():
    """Pattern matcher: prefix semantics + fnmatch fallback."""
    from mdvault.indexer import _matches_keep_deleted

    assert _matches_keep_deleted("projects/uuid/log.jsonl", ["projects"])
    assert _matches_keep_deleted("projects", ["projects"])
    assert not _matches_keep_deleted("skills/foo.md", ["projects"])
    assert _matches_keep_deleted("foo.jsonl", ["*.jsonl"])
    assert _matches_keep_deleted("a/b/c.jsonl", ["a/b"])
    assert not _matches_keep_deleted("ab/c.jsonl", ["a"])


def test_additive_adds_new_file(tmp_path, mock_embedder):
    """Index -> add new .md -> additive -> new file indexed."""
    vault = tmp_path / "vault"
    vault.mkdir()
    f1 = vault / "note1.md"
    f1.write_text("## First\n\nFirst note content with enough words to pass the minimum chunk threshold easily.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 1

    f2 = vault / "note2.md"
    f2.write_text("## Second\n\nSecond note content with enough words to pass the minimum chunk threshold easily.\n")

    index_directory(conn, vault, mock_embedder)
    conn.commit()

    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 2
    conn.close()


def test_batch_processing_large_vault(tmp_path, mock_embedder):
    """Create 150 temp markdown files -> index -> all indexed correctly."""
    vault = tmp_path / "vault"
    vault.mkdir()
    for i in range(150):
        (vault / f"note_{i:03d}.md").write_text(
            f"## Note {i}\n\nThis is note {i} with enough content to satisfy the chunk size requirement.\n"
        )

    db_p = tmp_path / "test.db"
    init_db(db_p)
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
    content = "Real [link](real.md) here.\n\n```markdown\n[fake](fake.md)\n```\n\nAnd `[inline](inline.md)` code."
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
    (vault / "b.md").write_text("# Note B\n\n## Content\n\nThis is note B with enough words to form a valid chunk.\n")
    (vault / "c.md").write_text("# Note C\n\n## Content\n\nThis is note C with enough words to form a valid chunk.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    links = [
        row["target_path"]
        for row in conn.execute(
            "SELECT target_path FROM links l JOIN files f ON f.id = l.source_file_id WHERE f.file_path = 'vault/a.md'"
        ).fetchall()
    ]
    conn.close()
    assert "vault/b.md" in links  # standard link resolved with vault prefix
    assert "c.md" in links  # wikilink stored as-is


def test_links_removed_on_file_delete(tmp_path, mock_embedder):
    """When a file is deleted (incremental), its links are cascade-deleted."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# Note A\n\n## Links\n\nSee [B](b.md) for reference with enough words to chunk.\n")
    (vault / "b.md").write_text("# Note B\n\n## Content\n\nEnough words to form a valid chunk for indexing purposes.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM links").fetchone()["c"] > 0

    (vault / "a.md").unlink()
    index_directory(conn, vault, mock_embedder)
    conn.commit()

    # Links from a.md should be gone (CASCADE)
    assert conn.execute("SELECT COUNT(*) as c FROM links").fetchone()["c"] == 0
    conn.close()


# ---------- _list_files ----------


def test_list_files_respects_gitignore(tmp_path):
    """Files in .gitignore are excluded when in a git repo."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "visible.md").write_text("# Visible\n\nContent here.\n")
    ignored_dir = vault / "node_modules"
    ignored_dir.mkdir()
    (ignored_dir / "dep.md").write_text("# Dep\n\nShould be ignored.\n")

    import subprocess

    subprocess.run(["git", "init"], cwd=vault, capture_output=True, check=True)
    (vault / ".gitignore").write_text("node_modules/\n")
    subprocess.run(["git", "add", "."], cwd=vault, capture_output=True, check=True)

    files = _list_files(vault)
    names = [f.name for f in files]
    assert "visible.md" in names
    assert "dep.md" not in names


def test_list_files_fallback_no_git(tmp_path):
    """Without git, falls back to rglob and finds all .md files."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n")
    (vault / "b.md").write_text("# B\n")

    files = _list_files(vault)
    names = {f.name for f in files}
    assert names == {"a.md", "b.md"}


# ---------- empty chunks ----------


def test_empty_files_not_indexed(tmp_path, mock_embedder):
    """Files with no meaningful content produce no chunks."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "empty.md").write_text("")
    (vault / "whitespace.md").write_text("   \n\n  \n")
    (vault / "real.md").write_text("## Real\n\nThis has enough words to form a valid chunk for indexing.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    chunk_count = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
    conn.close()
    assert chunk_count > 0  # real.md has chunks
    # But no empty chunks exist
    conn = get_connection(db_p)
    empty = conn.execute("SELECT COUNT(*) as c FROM chunks WHERE trim(raw_content) = ''").fetchone()["c"]
    conn.close()
    assert empty == 0


# ---------- JSONL session indexing ----------


def test_jsonl_session_indexed(tmp_path, mock_embedder):
    """JSONL Claude Code session files are extracted and indexed."""
    import json

    vault = tmp_path / "vault"
    projects = vault / "projects" / "my-project"
    projects.mkdir(parents=True)

    session = [
        {"type": "user", "uuid": "1", "message": {"role": "user", "content": "How do I configure Kafka interceptors?"}},
        {
            "type": "assistant",
            "uuid": "2",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {
                        "type": "text",
                        "text": "Kafka interceptors are configured via the Gateway plugin system.",
                    },
                ],
            },
        },
        {"type": "progress", "uuid": "3", "data": {"type": "hook_progress"}},
        {"type": "user", "uuid": "4", "message": {"role": "user", "content": "Show me an example with encryption"}},
        {
            "type": "assistant",
            "uuid": "5",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is an encryption interceptor example that protects sensitive fields.",
                    },
                ],
            },
        },
    ]
    jsonl_path = projects / "abc123.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in session))

    # Also add a regular .md
    (vault / "readme.md").write_text("# Readme\n\n## Setup\n\nThis is a regular markdown file with enough words.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True, no_gitignore=True)
    conn.commit()

    files = conn.execute("SELECT file_path FROM files ORDER BY file_path").fetchall()
    paths = [r["file_path"] for r in files]
    assert any("abc123.jsonl" in p for p in paths)
    assert any("readme.md" in p for p in paths)

    # Check that JSONL content was extracted (search for user prompt text)
    chunks = conn.execute("SELECT raw_content FROM chunks").fetchall()
    all_content = " ".join(c["raw_content"] for c in chunks)
    assert "Kafka interceptors" in all_content
    assert "encryption interceptor" in all_content
    # Progress messages should NOT be in the content
    assert "hook_progress" not in all_content
    conn.close()


def test_jsonl_empty_session_skipped(tmp_path, mock_embedder):
    """JSONL files with no user/assistant messages are not indexed."""
    import json

    vault = tmp_path / "vault"
    vault.mkdir()
    session = [
        {"type": "progress", "uuid": "1", "data": {"type": "hook_progress"}},
        {"type": "progress", "uuid": "2", "data": {"type": "hook_progress"}},
    ]
    (vault / "empty.jsonl").write_text("\n".join(json.dumps(r) for r in session))

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True, no_gitignore=True)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
    conn.close()
    assert count == 0


# ---------- multi-vault ----------


def test_multi_vault_additive(tmp_path, mock_embedder):
    """Indexing two different vaults into same DB keeps both."""
    vault_a = tmp_path / "blog"
    vault_a.mkdir()
    (vault_a / "post.md").write_text("## Blog\n\nA blog post with enough words to form a valid chunk for indexing.\n")

    vault_b = tmp_path / "notes"
    vault_b.mkdir()
    (vault_b / "idea.md").write_text("## Idea\n\nAn idea note with enough words to form a valid chunk for indexing.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)

    index_directory(conn, vault_a, mock_embedder)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 1

    index_directory(conn, vault_b, mock_embedder)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 2

    paths = [r["file_path"] for r in conn.execute("SELECT file_path FROM files ORDER BY file_path").fetchall()]
    conn.close()
    assert paths == ["blog/post.md", "notes/idea.md"]


def test_multi_vault_full_only_wipes_target(tmp_path, mock_embedder):
    """full=True wipes only the targeted vault, not others."""
    vault_a = tmp_path / "blog"
    vault_a.mkdir()
    (vault_a / "post.md").write_text("## Blog\n\nA blog post with enough words to form a valid chunk for indexing.\n")

    vault_b = tmp_path / "notes"
    vault_b.mkdir()
    (vault_b / "idea.md").write_text("## Idea\n\nAn idea note with enough words to form a valid chunk for indexing.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)

    index_directory(conn, vault_a, mock_embedder)
    index_directory(conn, vault_b, mock_embedder)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 2

    # Full re-index of blog only
    index_directory(conn, vault_a, mock_embedder, full=True)
    conn.commit()

    # notes vault untouched
    assert conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"] == 2
    assert conn.execute("SELECT COUNT(*) as c FROM files WHERE file_path LIKE 'notes/%'").fetchone()["c"] == 1
    conn.close()


def test_vault_name_collision_raises(tmp_path, mock_embedder):
    """Two different paths with same directory name raises ValueError."""
    vault1 = tmp_path / "dir1" / "vault"
    vault1.mkdir(parents=True)
    (vault1 / "a.md").write_text("## A\n\nEnough words to form a valid chunk for indexing purposes here.\n")

    vault2 = tmp_path / "dir2" / "vault"
    vault2.mkdir(parents=True)
    (vault2 / "b.md").write_text("## B\n\nEnough words to form a valid chunk for indexing purposes here.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)

    index_directory(conn, vault1, mock_embedder)
    conn.commit()

    import pytest

    with pytest.raises(ValueError, match="already used"):
        index_directory(conn, vault2, mock_embedder)
    conn.close()


def test_file_path_prefixed_with_vault_name(tmp_path, mock_embedder):
    """file_path in DB is prefixed with vault directory name."""
    vault = tmp_path / "my-vault"
    vault.mkdir()
    (vault / "note.md").write_text("## Note\n\nEnough words to form a valid chunk for indexing purposes here.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder)
    conn.commit()

    fp = conn.execute("SELECT file_path FROM files").fetchone()["file_path"]
    conn.close()
    assert fp == "my-vault/note.md"


def test_vault_root_stored_in_config(tmp_path, mock_embedder):
    """Vault root is stored in vault_config after indexing."""
    vault = tmp_path / "docs"
    vault.mkdir()
    (vault / "readme.md").write_text("## Readme\n\nEnough words for a valid chunk to be indexed properly here.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p)
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder)
    conn.commit()

    row = conn.execute("SELECT value FROM vault_config WHERE key = 'vault_root:docs'").fetchone()
    conn.close()
    assert row is not None
    assert "docs" in row["value"]
