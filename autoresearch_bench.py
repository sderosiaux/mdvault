"""Autoresearch benchmark: MDN Content + GitHub Docs (30 queries, 2 corpora)."""

import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_VERBOSITY"] = "error"

from model2vec import StaticModel

from mdvault.db import get_connection, init_db
from mdvault.indexer import index_directory
from mdvault.retriever import hybrid_search

MDN = Path("/tmp/mdn-content/files/en-us")  # noqa: S108
GITHUB = Path("/tmp/github-docs/content")  # noqa: S108
DB_PATH = Path("/tmp/autoresearch_multi_bench.db")  # noqa: S108
GOLD_PATH = Path("autoresearch_gold.json")

model = StaticModel.from_pretrained("minishlab/potion-base-8M")


def embedder(texts):
    return model.encode(texts)


def ensure_index():
    if DB_PATH.exists():
        return
    init_db(DB_PATH)
    conn = get_connection(str(DB_PATH))
    index_directory(conn, MDN, embedder, full=True)
    index_directory(conn, GITHUB, embedder)
    conn.close()


def run_bench():
    ensure_index()
    gold = json.loads(GOLD_PATH.read_text())
    conn = get_connection(str(DB_PATH))
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    recalls, rrs, latencies = [], [], []
    for entry in gold:
        query = entry["query"]
        expected = set(entry["expected_files"])
        t0 = time.perf_counter()
        results = hybrid_search(conn, query, embedder, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)
        found_paths = {r["file_path"] for r in results}
        recall = 1.0 if expected & found_paths else 0.0
        recalls.append(recall)
        rr = 0.0
        for i, r in enumerate(results):
            if r["file_path"] in expected:
                rr = 1.0 / (i + 1)
                break
        rrs.append(rr)
        mark = "v" if recall > 0 else "x"
        print(
            f"  {mark} recall={recall:.2f} rr={rr:.2f} | {query}",
            file=sys.stderr,
        )

    conn.close()
    recall_at_5 = sum(recalls) / len(recalls)
    mrr = sum(rrs) / len(rrs)
    quality = 50 * recall_at_5 + 50 * mrr
    avg_lat = sum(latencies) / len(latencies)
    print(f"METRIC quality_score={quality:.2f}")
    print(f"METRIC recall_at_5={recall_at_5:.4f}")
    print(f"METRIC mrr={mrr:.4f}")
    print(f"METRIC avg_latency_ms={avg_lat:.1f}")
    print(f"METRIC total_chunks={total_chunks}")
    print(f"METRIC num_queries={len(gold)}")


if __name__ == "__main__":
    run_bench()
