# Python Development Patterns

## Virtual Environments

Always isolate project dependencies:

```bash
# Using uv (fastest)
uv venv
uv pip install -r requirements.txt

# Using venv (stdlib)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Type Hints and Protocols

Modern Python uses type hints extensively for better IDE support and static analysis:

```python
from typing import Protocol, TypeVar
from collections.abc import Sequence

class Repository(Protocol):
    def get(self, id: int) -> dict | None: ...
    def list(self, limit: int = 10) -> Sequence[dict]: ...
    def save(self, entity: dict) -> int: ...

T = TypeVar("T")

def first_or_none(items: Sequence[T]) -> T | None:
    return items[0] if items else None
```

## Context Managers

Use context managers for resource cleanup:

```python
from contextlib import contextmanager
import sqlite3

@contextmanager
def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

## Async Patterns

Asyncio patterns for concurrent I/O:

```python
import asyncio
import httpx

async def fetch_all(urls: list[str]) -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]
```

## Dataclasses and Pydantic

Structured data with validation:

```python
from pydantic import BaseModel, Field

class Config(BaseModel):
    host: str = "localhost"
    port: int = Field(default=8080, ge=1024, le=65535)
    debug: bool = False
    workers: int = Field(default=4, ge=1)
```
