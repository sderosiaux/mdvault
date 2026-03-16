import pytest
import numpy as np
from mdvault.db import get_connection, init_db
from mdvault.indexer import index_directory
from mdvault.retriever import (
    bm25_search,
    vector_search,
    rrf_fusion,
    hybrid_search,
    get_total_chunks,
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
