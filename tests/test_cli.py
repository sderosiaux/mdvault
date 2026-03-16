import os
import pytest
from typer.testing import CliRunner
from mdvault.cli import app
from pathlib import Path
from tests.conftest import FIXTURES_DIR

runner = CliRunner()


def test_index_command_creates_db(tmp_path):
    """mdvault index ./fixtures/ creates vault.db with rows."""
    db_file = tmp_path / "vault.db"
    result = runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    assert db_file.exists()
    assert "Indexed" in result.output or "indexed" in result.output


def test_index_incremental_flag(tmp_path):
    """mdvault index ./fixtures/ --incremental works."""
    db_file = tmp_path / "vault.db"
    # First full index
    result = runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    # Then incremental
    result = runner.invoke(
        app, ["index", str(FIXTURES_DIR), "--db", str(db_file), "--incremental"]
    )
    assert result.exit_code == 0, result.output


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
    result = runner.invoke(
        app, ["search", "nginx", "--db", str(db_file), "--top-k", "3"]
    )
    assert result.exit_code == 0, result.output
    # Count result markers [1], [2], [3]
    result_count = result.output.count("[")
    assert result_count >= 3


def test_related_command(tmp_path):
    """mdvault related shows links, backlinks, similar."""
    db_file = tmp_path / "vault.db"
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text(
        "# A\n\n## Content\n\nSee [B](b.md) for reference with enough words to chunk.\n"
    )
    (vault / "b.md").write_text(
        "# B\n\n## Content\n\nNote B with enough words to form a valid chunk for indexing.\n"
    )
    runner.invoke(app, ["index", str(vault), "--db", str(db_file)])
    result = runner.invoke(app, ["related", "a.md", "--db", str(db_file)])
    assert result.exit_code == 0, result.output
    assert "Links" in result.output
    assert "Backlinks" in result.output
    assert "Similar" in result.output
    assert "b.md" in result.output


def test_search_vault_tool_returns_correct_schema(tmp_path):
    """MCP search_vault tool returns correct schema."""
    db_file = tmp_path / "vault.db"
    # Index first via CLI
    runner.invoke(app, ["index", str(FIXTURES_DIR), "--db", str(db_file)])

    # Call the MCP tool directly
    os.environ["VAULT_DB"] = str(db_file)
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
