from unittest.mock import patch

import pytest

from mdvault.db import get_connection, init_db
from mdvault.indexer import index_directory
from mdvault.retriever import (
    bm25_search,
    expand_query_llm,
    get_total_chunks,
    hybrid_search,
    related_notes,
    rrf_fusion,
    vector_search,
)
from tests.conftest import FIXTURES_DIR


@pytest.fixture
def indexed_db(db_path, mock_embedder):
    """DB with all fixtures indexed."""
    conn = get_connection(db_path)
    index_directory(conn, FIXTURES_DIR, mock_embedder, full=True)
    conn.commit()
    conn.close()
    return db_path


# ---------- bm25_search ----------


def test_bm25_search_returns_ranked_results(indexed_db):
    """After indexing, bm25_search('nginx') returns results containing nginx.md chunks."""
    conn = get_connection(indexed_db)
    results = bm25_search(conn, "nginx")
    conn.close()
    assert len(results) > 0
    # At least one result should be from nginx.md
    file_paths = [r["file_path"] for r in results]
    assert any("nginx" in fp for fp in file_paths)


def test_bm25_search_returns_top_k(indexed_db):
    """Requesting top_k=3 returns exactly 3 results."""
    conn = get_connection(indexed_db)
    results = bm25_search(conn, "nginx", top_k=3)
    conn.close()
    assert len(results) == 3


def test_bm25_search_empty_vault_returns_empty(db_path):
    """Empty DB -> returns []."""
    conn = get_connection(db_path)
    results = bm25_search(conn, "anything")
    conn.close()
    assert results == []


# ---------- vector_search ----------


def test_vector_search_returns_ranked_results(indexed_db, mock_embedder):
    """vector_search(query_vec, top_k=5) returns 5 results ordered by distance."""
    query_vec = mock_embedder(["nginx reverse proxy"])[0]
    conn = get_connection(indexed_db)
    results = vector_search(conn, query_vec, top_k=5)
    conn.close()
    assert len(results) == 5
    # Distances should be in ascending order (closest first)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


# ---------- rrf_fusion ----------


def test_rrf_fusion_combines_both_lists():
    """A result in both lists scores higher than one in only one list."""
    bm25_results = [
        {"chunk_id": 1, "file_path": "a.md", "chunk_idx": 0, "content": "a", "raw_content": "a"},
        {"chunk_id": 2, "file_path": "b.md", "chunk_idx": 0, "content": "b", "raw_content": "b"},
    ]
    vec_results = [
        {"chunk_id": 1, "file_path": "a.md", "chunk_idx": 0, "content": "a", "raw_content": "a"},
        {"chunk_id": 3, "file_path": "c.md", "chunk_idx": 0, "content": "c", "raw_content": "c"},
    ]
    fused = rrf_fusion(bm25_results, vec_results, top_k=10)
    # chunk_id=1 is in both lists, should have highest score
    assert fused[0]["chunk_id"] == 1
    assert fused[0]["score"] > fused[1]["score"]


def test_rrf_fusion_chunk_in_only_bm25():
    """Chunk absent from vec list still appears with partial score."""
    bm25_results = [
        {"chunk_id": 10, "file_path": "x.md", "chunk_idx": 0, "content": "x", "raw_content": "x"},
    ]
    vec_results = []
    fused = rrf_fusion(bm25_results, vec_results, top_k=10)
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == 10
    assert fused[0]["score"] > 0


def test_rrf_fusion_chunk_in_only_vec():
    """Chunk absent from bm25 list still appears with partial score."""
    bm25_results = []
    vec_results = [
        {"chunk_id": 20, "file_path": "y.md", "chunk_idx": 0, "content": "y", "raw_content": "y"},
    ]
    fused = rrf_fusion(bm25_results, vec_results, top_k=10)
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == 20
    assert fused[0]["score"] > 0


# ---------- hybrid_search ----------


def test_hybrid_search_end_to_end(indexed_db, mock_embedder):
    """hybrid_search returns results with file_path, chunk_idx, content, score."""
    conn = get_connection(indexed_db)
    results = hybrid_search(conn, "nginx reverse proxy", mock_embedder, top_k=5)
    conn.close()
    assert len(results) > 0
    r = results[0]
    assert "file_path" in r
    assert "chunk_idx" in r
    assert "content" in r
    assert "raw_content" in r
    assert "score" in r


def test_hybrid_search_correct_file_paths(indexed_db, mock_embedder):
    """Results for 'nginx' queries have file_path pointing to nginx.md."""
    conn = get_connection(indexed_db)
    results = hybrid_search(conn, "nginx upstream reverse proxy", mock_embedder, top_k=5)
    conn.close()
    file_paths = [r["file_path"] for r in results]
    # At least one should reference nginx
    assert any("nginx" in fp for fp in file_paths)


def test_hybrid_search_top_k_respected(indexed_db, mock_embedder):
    """Exactly top_k results returned."""
    conn = get_connection(indexed_db)
    results = hybrid_search(conn, "nginx", mock_embedder, top_k=3)
    conn.close()
    assert len(results) == 3


def test_rrf_k_parameter():
    """k=60 is default; rrf score formula: 1/(60+rank)."""
    bm25_results = [
        {"chunk_id": 1, "file_path": "a.md", "chunk_idx": 0, "content": "a", "raw_content": "a"},
    ]
    vec_results = [
        {"chunk_id": 1, "file_path": "a.md", "chunk_idx": 0, "content": "a", "raw_content": "a"},
    ]
    fused = rrf_fusion(bm25_results, vec_results, top_k=10, k=60)
    # Rank 1 in both: 1/(60+1) + 1/(60+1) = 2/61
    expected_score = 2.0 / 61.0
    assert abs(fused[0]["score"] - expected_score) < 1e-6


# ---------- get_total_chunks ----------


def test_total_chunks_count(indexed_db):
    """get_total_chunks returns correct count."""
    conn = get_connection(indexed_db)
    total = get_total_chunks(conn)
    chunk_count = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
    conn.close()
    assert total == chunk_count
    assert total > 0


# ---------- related_notes ----------

# ---------- expand_query_llm ----------


def test_expand_query_llm_returns_none_when_ollama_unavailable():
    """When Ollama is not running, expand_query_llm returns None gracefully."""
    result = expand_query_llm("ssh tunnel", model="nonexistent")
    assert result is None


def test_hybrid_search_expand_flag(indexed_db, mock_embedder):
    """hybrid_search with expand=True still returns results (falls back if no Ollama)."""
    conn = get_connection(indexed_db)
    results = hybrid_search(conn, "nginx", mock_embedder, top_k=3, expand=True)
    conn.close()
    # Should still work (expansion fails gracefully, falls back to original query)
    assert len(results) == 3


def test_hybrid_search_with_mock_expansion(indexed_db, mock_embedder):
    """hybrid_search uses expanded query for vector search when expansion succeeds."""
    conn = get_connection(indexed_db)
    with patch("mdvault.retriever.expand_query_llm", return_value="nginx reverse proxy upstream load balancer"):
        results = hybrid_search(conn, "nginx", mock_embedder, top_k=3, expand=True)
    conn.close()
    assert len(results) == 3


# ---------- related_notes ----------


def test_related_notes_returns_structure(tmp_path, mock_embedder):
    """related_notes returns dict with links, backlinks, similar."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\n## Content\n\nSee [B](b.md) for details with enough words to chunk properly.\n")
    (vault / "b.md").write_text("# B\n\n## Content\n\nNote B references [[a]] with enough words to chunk properly.\n")
    (vault / "c.md").write_text(
        "# C\n\n## Content\n\nUnrelated note C with enough words to chunk properly for indexing.\n"
    )

    db_p = tmp_path / "test.db"
    init_db(db_p, vault_root=str(vault))
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    result = related_notes(conn, "a.md", mock_embedder)
    conn.close()

    assert result["file_path"] == "a.md"
    assert "b.md" in result["links"]  # a links to b
    assert "b.md" in result["backlinks"]  # b wikilinks to a
    assert isinstance(result["similar"], list)


def test_related_notes_backlinks_by_filename(tmp_path, mock_embedder):
    """Backlinks match by filename for wikilinks stored without path."""
    vault = tmp_path / "vault"
    sub = vault / "sub"
    sub.mkdir(parents=True)
    (sub / "deep.md").write_text("# Deep\n\n## Content\n\nEnough words to chunk. See [[top]] for parent reference.\n")
    (vault / "top.md").write_text("# Top\n\n## Content\n\nTop level note with enough words to form a valid chunk.\n")

    db_p = tmp_path / "test.db"
    init_db(db_p, vault_root=str(vault))
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    result = related_notes(conn, "top.md", mock_embedder)
    conn.close()

    # [[top]] stored as "top.md", backlink query matches by filename
    assert "sub/deep.md" in result["backlinks"]


def test_related_notes_similar_excludes_self(tmp_path, mock_embedder):
    """Similar files list never includes the queried file itself."""
    vault = tmp_path / "vault"
    vault.mkdir()
    for name in ["x.md", "y.md", "z.md"]:
        (vault / name).write_text(
            f"# {name}\n\n## Content\n\nThis is {name} with enough words for a valid chunk to index.\n"
        )

    db_p = tmp_path / "test.db"
    init_db(db_p, vault_root=str(vault))
    conn = get_connection(db_p)
    index_directory(conn, vault, mock_embedder, full=True)
    conn.commit()

    result = related_notes(conn, "x.md", mock_embedder)
    conn.close()

    assert "x.md" not in result["similar"]
    assert len(result["similar"]) <= 5
