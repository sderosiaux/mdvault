# mdvault — Spec

## Vision

CLI + MCP server that indexes a folder of Markdown files into a local SQLite database and exposes hybrid search (BM25 + vector + RRF) directly usable from Claude Code.

Zero infrastructure. Everything lives in a single `.db` file.

---

## Features

### CLI

```bash
# Full index (first run or complete reindex — wipes and rebuilds)
mdvault index ./notes/ [--db ./vault.db]

# Incremental index (only modified/new/deleted files)
mdvault index ./notes/ --incremental [--db ./vault.db]

# Search from the CLI
mdvault search "nginx reverse proxy config" [--db ./vault.db] [--top-k 5]

# Start the MCP server (for Claude Code)
mdvault serve [--db ./vault.db]

# Index stats
mdvault stats [--db ./vault.db]
```

DB resolution order for all commands: `--db` flag → `VAULT_DB` env var → `platformdirs.user_data_dir("mdvault")/vault.db`.

### MCP Server

Exposes a `search_vault` tool to Claude Code:

```json
{
  "name": "search_vault",
  "description": "Search the local markdown vault using hybrid BM25 + semantic search",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string" },
      "top_k": { "type": "integer", "default": 5 }
    },
    "required": ["query"]
  }
}
```

Example queries Claude Code can answer:
- *"search my notes for how I configured SSH"*
- *"find docs related to timeout errors"*

---

## Tech Stack

| Component | Library | Role |
|---|---|---|
| Embeddings | `model2vec` (potion-base-8M) | 256-dim vectors, CPU-only, fast |
| Vector search | `sqlite-vec` | SQLite extension for exact vector search |
| Full-text search | SQLite FTS5 | Inverted index + native BM25 scoring |
| Fusion | RRF (manual) | Merges BM25 and vector result lists |
| MCP | `mcp` (official Python SDK) | Claude Code integration |
| CLI | `typer` | Command-line interface |
| Default paths | `platformdirs` | XDG-compliant data directory resolution |

---

## Project Layout

```
mdvault/
├── mdvault/
│   ├── __init__.py
│   ├── cli.py          # typer commands: index, search, serve, stats
│   ├── indexer.py      # file reading, chunking, embeddings, DB writes
│   ├── retriever.py    # BM25 search, vector search, RRF fusion
│   ├── mcp_server.py   # MCP server exposing search_vault
│   └── db.py           # schema setup, migrations, connection helpers
├── tests/
│   ├── test_indexer.py
│   ├── test_retriever.py
│   └── fixtures/       # sample markdown files
├── README.md
└── pyproject.toml
```

---

## Database

Single file `vault.db`. Location resolved as: `--db` flag → `VAULT_DB` env var → `$XDG_DATA_HOME/mdvault/vault.db`.

Every connection must run `PRAGMA foreign_keys = ON` to enforce `ON DELETE CASCADE`.

### Table `vault_config`

```sql
CREATE TABLE vault_config (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- Populated at first index:
-- INSERT OR REPLACE INTO vault_config VALUES ('vault_root', '/abs/path/to/notes');
```

### Table `files`

```sql
CREATE TABLE files (
  id         INTEGER PRIMARY KEY,
  file_path  TEXT NOT NULL UNIQUE,  -- relative to vault_root
  file_hash  TEXT NOT NULL,         -- SHA256 of file content
  indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Table `chunks`

```sql
CREATE TABLE chunks (
  id        INTEGER PRIMARY KEY,
  file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_idx INTEGER NOT NULL,  -- 0-based position within the file
  content   TEXT NOT NULL
);

CREATE INDEX idx_chunks_file_id ON chunks(file_id);
```

### FTS5 table

External content index over `chunks` (no content duplication). Manually synchronized in code — no triggers.

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  content,
  content='chunks',
  content_rowid='id'
);
```

- Insert: `INSERT INTO chunks_fts(rowid, content) VALUES (:chunk_id, :content)`
- Delete: `DELETE FROM chunks_fts WHERE rowid = :chunk_id` — **must happen before** `DELETE FROM chunks`

### Vector table (sqlite-vec)

```sql
CREATE VIRTUAL TABLE chunks_vec USING vec0(
  embedding FLOAT[256]
);
```

**Invariant**: `chunks_vec.rowid == chunks.id`. Always insert with an explicit rowid:

```sql
INSERT INTO chunks_vec(rowid, embedding) VALUES (:chunk_id, :embedding);
```

Delete: `DELETE FROM chunks_vec WHERE rowid = :chunk_id` — **must happen before** `DELETE FROM chunks`.

---

## Indexing Pipeline

### Full index (`mdvault index ./notes/`)

Drops and rebuilds the entire index. Existing `files`, `chunks`, `chunks_fts`, `chunks_vec` rows are purged. `vault_config` is updated with the vault root.

```
For each .md file (batches of 100 files):
    │
    ▼
Chunk the file (see Chunking Strategy)
    │
    ▼
Embed chunks via Model2Vec (batches of 32 chunks)
    │
    ▼
Atomic transaction per file:
    INSERT INTO files
    INSERT INTO chunks         (one row per chunk)
    INSERT INTO chunks_fts     (one row per chunk, explicit rowid)
    INSERT INTO chunks_vec     (one row per chunk, explicit rowid)
```

### Incremental index (`mdvault index ./notes/ --incremental`)

```
Scan all .md files on disk → compute SHA256 per file
    │
    ├── hash matches files.file_hash → skip
    ├── hash differs               → update (see below)
    └── not in DB                  → index as new file

After scan:
    SET(disk paths) vs SET(files.file_path in DB)
    → paths in DB but not on disk → delete
```

**Update a modified file** (single transaction):

```sql
-- 1. Collect chunk ids
SELECT id FROM chunks WHERE file_id = (SELECT id FROM files WHERE file_path = ?);

-- 2. Remove from FTS and vector tables first
DELETE FROM chunks_fts WHERE rowid IN (:ids);
DELETE FROM chunks_vec WHERE rowid IN (:ids);

-- 3. Delete file row — CASCADE removes chunks rows
DELETE FROM files WHERE file_path = ?;

-- 4. Re-index as a new file (same pipeline as full index)
```

**Delete a removed file** (same sequence, skip step 4).

**File batching**: process 100 files per transaction to bound memory usage on large vaults (10k+ files).

---

## Chunking Strategy

"Word" = token = `text.split()` entry. No tokenizer dependency.

| Rule | Detail |
|---|---|
| Primary split | `##` and `###` headings. The heading line is included at the start of its chunk. |
| Oversized block | Block > 400 words → split by paragraph (double newline) |
| Oversized paragraph | Paragraph > 400 words → hard split at 400 words |
| No headings | Treat entire file as one block, then apply paragraph/hard split |
| Overlap | Last 50 words of chunk N prepended to chunk N+1 |
| Minimum chunk | < 20 words → merge into previous chunk |

---

## Retrieval Pipeline (RRF)

```
Query string
    │
    ├──► Embed query via Model2Vec
    │         │
    │         ▼
    │    chunks_vec exact search → top-50 results (rank 1 = closest vector)
    │
    └──► FTS5 BM25 search → top-50 results (rank 1 = highest BM25 score)
                │
                ▼
        Reciprocal Rank Fusion
        score(chunk) = 1/(60 + rank_bm25) + 1/(60 + rank_vec)
        chunks present in only one list: missing rank = ∞ → contribution = 0
                │
                ▼
        Sort by RRF score desc → return top_k chunks
```

Ranks are 1-indexed. A chunk absent from one list contributes only its present-list term.

---

## MCP Response Format

```json
{
  "results": [
    {
      "file_path": "infra/nginx.md",
      "chunk_idx": 2,
      "content": "## Reverse proxy config\n\nTo forward traffic to...",
      "score": 0.041
    }
  ],
  "query": "nginx reverse proxy",
  "total_chunks": 49821
}
```

`file_path` is relative to `vault_root`. `total_chunks` is the count of all indexed chunks (sqlite-vec performs exact search, so all chunks are scanned).

---

## CLI Output: `mdvault search`

```
Query: nginx reverse proxy config
Chunks searched: 49821

[1] infra/nginx.md (chunk 2) — score 0.041
──────────────────────────────────────────
## Reverse proxy config

To forward traffic to an upstream service...
──────────────────────────────────────────

[2] infra/traefik.md (chunk 0) — score 0.031
...
```

---

## Claude Code Configuration (`.claude/mcp.json`)

Transport: **stdio**. Claude Code launches `mdvault serve` as a subprocess; MCP JSON-RPC is exchanged over stdin/stdout.

```json
{
  "mcpServers": {
    "mdvault": {
      "command": "uvx",
      "args": ["mdvault", "serve"],
      "env": {
        "VAULT_DB": "/path/to/vault.db"
      }
    }
  }
}
```

---

## Implicit Behaviors

### `mdvault stats` output

```
Vault root    : /Users/me/notes
DB path       : /Users/me/.mdvault/vault.db
Files indexed : 1,234
Total chunks  : 48,291
DB size       : 312 MB
```

### Scale notes

- sqlite-vec performs **exact** search (no ANN index) — safe up to ~500k chunks on a modern CPU
- FTS5 BM25 handles millions of documents without tuning
- For vaults > 500k chunks: consider partitioning or HNSW (out of scope v1)

---

## Out of Scope (v1)

- Non-Markdown formats (PDF, DOCX)
- Web UI / dashboard
- Multilingual embeddings (potion-base-8M is primarily English — note in README)
- Neural reranker (cross-encoder) — RRF is sufficient for v1
- Multi-vault (one folder per instance)
