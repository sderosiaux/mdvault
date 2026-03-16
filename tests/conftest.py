from pathlib import Path

import numpy as np
import pytest

from mdvault.db import init_db

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
