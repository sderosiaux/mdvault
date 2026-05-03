import os
from pathlib import Path

import numpy as np
import pytest

from mdvault.db import init_db

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _prewarm_embedder():
    """Download the model2vec model once before any test runs.

    Subsequent ``runner.invoke`` calls hit a warm cache, so no download
    progress bars leak into ``result.output`` and break ``json.loads``.
    Tolerant to offline / no-cache: tests that genuinely need the real
    embedder (CLI tests via runner.invoke) will fail naturally with a
    clearer error if the model is unavailable.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from mdvault.cli import _get_embedder

        _get_embedder()
    except Exception:  # noqa: S110 - tests using the real embedder fail naturally if download is impossible
        pass


@pytest.fixture
def db_path(tmp_path):
    """Fresh in-file DB per test (sqlite-vec needs file, not :memory: for some ops)."""
    path = tmp_path / "test_vault.db"
    init_db(path)
    return path


@pytest.fixture
def mock_embedder():
    """Deterministic fake embedder -- hash-based, 256-dim, no model download."""

    def embed(texts: list[str]) -> np.ndarray:
        result = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.random(256, dtype=np.float32)
            vec /= np.linalg.norm(vec)
            result.append(vec)
        return np.array(result, dtype=np.float32)

    return embed
