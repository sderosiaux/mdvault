# Testing Strategies

## Test Pyramid

From bottom to top:
1. **Unit tests**: Fast, isolated, test single functions/classes
2. **Integration tests**: Test component interactions (DB, APIs)
3. **End-to-end tests**: Test full user flows, slowest

Aim for 70% unit, 20% integration, 10% E2E.

## Pytest Fixtures

```python
import pytest
from pathlib import Path

@pytest.fixture
def tmp_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("debug: true\nport: 8080\n")
    return config_file

@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

def test_load_config(tmp_config):
    config = load_config(tmp_config)
    assert config["debug"] is True
    assert config["port"] == 8080
```

## Property-Based Testing with Hypothesis

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_sort_is_idempotent(xs):
    assert sorted(sorted(xs)) == sorted(xs)

@given(st.text(min_size=1))
def test_roundtrip_encode_decode(s):
    assert decode(encode(s)) == s
```

## Mocking External Services

```python
from unittest.mock import AsyncMock, patch

@patch("myapp.client.httpx.AsyncClient.get")
async def test_fetch_user(mock_get):
    mock_get.return_value = AsyncMock(
        status_code=200,
        json=lambda: {"id": 1, "name": "Alice"}
    )
    user = await fetch_user(1)
    assert user["name"] == "Alice"
```
