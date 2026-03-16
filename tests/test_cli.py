from typer.testing import CliRunner

from mdvault.cli import app
from tests.conftest import FIXTURES_DIR

runner = CliRunner()


def test_index_command_creates_db(tmp_path):
    """mdvault index ./fixtures/ creates vault.db with rows."""
    db_file = tmp_path / "vault.db"
    result = runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    assert db_file.exists()
    assert "Indexed" in result.output or "indexed" in result.output


def test_index_additive_rerun(tmp_path):
    """mdvault index twice (additive) works without duplication."""
    db_file = tmp_path / "vault.db"
    result = runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    # Second run: additive, should not fail or duplicate
    result = runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    assert result.exit_code == 0, result.output


def test_index_full_flag(tmp_path):
    """mdvault index --full forces re-index."""
    db_file = tmp_path / "vault.db"
    result = runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file), "--full"])
    assert result.exit_code == 0, result.output
    assert "full re-index" in result.output


def test_search_command_returns_results(tmp_path):
    """After indexing, mdvault search 'nginx' outputs matches."""
    db_file = tmp_path / "vault.db"
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    result = runner.invoke(app, ["search", "nginx", "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    assert "nginx" in result.output.lower()


def test_stats_command_output(tmp_path):
    """mdvault stats shows Files indexed, Total chunks, DB path."""
    db_file = tmp_path / "vault.db"
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    result = runner.invoke(app, ["stats", "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    assert "Files indexed" in result.output or "files" in result.output.lower()
    assert "chunks" in result.output.lower()


def test_db_flag_respected(tmp_path):
    """--db /tmp/custom.db creates DB at that path."""
    custom_db = tmp_path / "custom.db"
    result = runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(custom_db)])
    assert result.exit_code == 0, result.output
    assert custom_db.exists()


def test_search_top_k_flag(tmp_path):
    """--top-k 3 returns 3 results."""
    db_file = tmp_path / "vault.db"
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    result = runner.invoke(app, ["search", "nginx", "--db", str(db_file), "--top-k", "3"])
    assert result.exit_code == 0, result.output
    # Count result markers [1], [2], [3]
    result_count = result.output.count("[")
    assert result_count >= 3


def test_related_command(tmp_path):
    """mdvault related shows links, backlinks, similar."""
    db_file = tmp_path / "vault.db"
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\n## Content\n\nSee [B](b.md) for reference with enough words to chunk.\n")
    (vault / "b.md").write_text("# B\n\n## Content\n\nNote B with enough words to form a valid chunk for indexing.\n")
    runner.invoke(app, ["index", str(vault), "--db", str(db_file)])
    result = runner.invoke(app, ["related", "vault/a.md", "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    assert "Links" in result.output
    assert "Backlinks" in result.output
    assert "Similar" in result.output
    assert "b.md" in result.output


def test_search_vault_tool_returns_correct_schema(tmp_path, monkeypatch):
    """MCP search_vault tool returns correct schema."""
    db_file = tmp_path / "vault.db"
    # Index first via CLI
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])

    # Call the MCP tool directly
    monkeypatch.setenv("VAULT_DB", str(db_file))
    from mdvault.mcp_server import search_vault

    result = search_vault("nginx reverse proxy", top_k=3)

    assert "results" in result
    assert "query" in result
    assert "total_chunks" in result
    assert result["query"] == "nginx reverse proxy"
    assert len(result["results"]) == 3
    r = result["results"][0]
    assert "file_path" in r
    assert "chunk_idx" in r
    assert "content" in r
    assert "score" in r


# ── MCP tool registration tests ──────────────────────────────────────


def _mcp_tool_names() -> dict:
    import asyncio

    from mdvault.mcp_server import mcp_app

    tools = asyncio.run(mcp_app.list_tools())
    return {t.name: t for t in tools}


def test_mcp_store_memory_tool_exists():
    """MCP server exposes store_memory tool."""
    assert "store_memory" in _mcp_tool_names()


def test_mcp_delete_memory_tool_exists():
    """MCP server exposes delete_memory tool."""
    assert "delete_memory" in _mcp_tool_names()


def test_mcp_update_memory_tool_exists():
    """MCP server exposes update_memory tool."""
    assert "update_memory" in _mcp_tool_names()


def test_mcp_search_vault_has_source_param():
    """search_vault tool accepts source parameter."""
    tools = _mcp_tool_names()
    search_tool = tools["search_vault"]
    schema = search_tool.inputSchema
    assert "source" in schema.get("properties", {})


# ── CLI remember / forget / memories tests ────────────────────────────


def test_remember_command(tmp_path):
    """mdvault remember stores a memory."""
    db_path = tmp_path / "test.db"
    result = runner.invoke(app, ["remember", "Python is great for scripting", "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "Stored" in result.output


def test_forget_by_id(tmp_path):
    """mdvault forget --id removes a memory."""
    db_path = tmp_path / "test.db"
    result = runner.invoke(app, ["remember", "To forget soon", "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    # Extract 8-char hex id from output like "Stored a1b2c3d4 (1 chunks)"
    import re

    match = re.search(r"Stored ([0-9a-f]{12})", result.output)
    assert match is not None, f"Could not find id in: {result.output}"
    mem_id = match.group(1)

    result = runner.invoke(app, ["forget", "--id", mem_id, "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "Deleted" in result.output


def test_memories_list(tmp_path):
    """mdvault memories lists stored memories."""
    db_path = tmp_path / "test.db"
    runner.invoke(app, ["remember", "Fact one content", "--db", str(db_path), "--namespace", "test"])
    runner.invoke(app, ["remember", "Fact two content", "--db", str(db_path), "--namespace", "test"])

    result = runner.invoke(app, ["memories", "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "2" in result.output


def test_read_command(tmp_path):
    """mdvault read returns indexed file content."""
    db_file = tmp_path / "vault.db"
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    # Get a file_path from the index
    from mdvault.db import get_connection

    conn = get_connection(db_file)
    row = conn.execute("SELECT file_path FROM files LIMIT 1").fetchone()
    conn.close()
    fp = row["file_path"]

    result = runner.invoke(app, ["read", fp, "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    assert f"File: {fp}" in result.output
    assert "Chunks:" in result.output


def test_read_command_json(tmp_path):
    """mdvault read --json returns structured output."""
    import json

    db_file = tmp_path / "vault.db"
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    from mdvault.db import get_connection

    conn = get_connection(db_file)
    row = conn.execute("SELECT file_path FROM files LIMIT 1").fetchone()
    conn.close()
    fp = row["file_path"]

    result = runner.invoke(app, ["read", fp, "--db", str(db_file), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["file_path"] == fp
    assert "content" in data
    assert "chunks" in data
    assert "disk_path" in data


def test_read_command_not_found(tmp_path):
    """mdvault read with unknown file returns error."""
    db_file = tmp_path / "vault.db"
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    result = runner.invoke(app, ["read", "nonexistent/file.md", "--db", str(db_file)])
    assert result.exit_code == 1


def test_list_command(tmp_path):
    """mdvault list shows indexed files."""
    db_file = tmp_path / "vault.db"
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    result = runner.invoke(app, ["list", "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    assert "Files:" in result.output
    assert ".md" in result.output


def test_list_command_json(tmp_path):
    """mdvault list --json returns array of file objects."""
    import json

    db_file = tmp_path / "vault.db"
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    result = runner.invoke(app, ["list", "--db", str(db_file), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "file_path" in data[0]
    assert "disk_path" in data[0]
    assert "vault" in data[0]


def test_list_command_with_pattern(tmp_path):
    """mdvault list --pattern filters files."""
    db_file = tmp_path / "vault.db"
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    result = runner.invoke(app, ["list", "--db", str(db_file), "--pattern", "*nginx*"])
    assert result.exit_code == 0, result.output
    # Should have fewer files than total
    for line in result.output.strip().split("\n")[1:]:  # skip "Files: N" line
        assert "nginx" in line.lower()


def test_search_with_source_flag(tmp_path):
    """mdvault search --source memories works."""
    db_path = tmp_path / "test.db"
    runner.invoke(app, ["remember", "Databases are fundamental to computing", "--db", str(db_path)])

    result = runner.invoke(app, ["search", "databases", "--db", str(db_path), "--source", "memories"])
    assert result.exit_code == 0, result.output
