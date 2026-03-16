# mdvault

[![CI](https://github.com/sderosiaux/mdvault/actions/workflows/ci.yml/badge.svg)](https://github.com/sderosiaux/mdvault/actions/workflows/ci.yml)

CLI + MCP server that indexes a folder of Markdown files into a local SQLite database and exposes **hybrid search** (BM25 + semantic + RRF) directly usable from Claude Code.

Zero infrastructure. Everything lives in a single `.db` file.

## Features

- **Hybrid search** — combines FTS5 BM25 and 256-dim vector search via Reciprocal Rank Fusion
- **Contextual chunking** — each chunk is prefixed with `[path > title > heading]` for better retrieval ([Anthropic approach](https://www.anthropic.com/news/contextual-retrieval))
- **Link graph** — parses Markdown links and `[[wikilinks]]`, surfaces backlinks and semantically similar files
- **Query expansion** — optional local LLM (Ollama) expands queries for richer vector search
- **Fast embeddings** — [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) via model2vec, CPU-only, ~30MB
- **Incremental indexing** — SHA256-based change detection, only reprocesses modified files
- **MCP server** — Claude Code can search your vault and explore related notes
- **Single file** — one `.db` holds everything (FTS5 index + vectors + link graph + metadata)

## Install

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uvx mdvault --help
```

Or install permanently:

```bash
uv tool install mdvault
```

## CLI Usage

```bash
# Index your notes (downloads ~30MB model on first run)
mdvault index ~/notes/

# Incremental update (only changed/new/deleted files)
mdvault index ~/notes/ --incremental

# Search
mdvault search "nginx reverse proxy config"
mdvault search "ssh tunnel" --top-k 10

# Search with query expansion (requires Ollama running locally)
mdvault search "ssh tunnel" --expand

# Related notes: links, backlinks, and semantically similar files
mdvault related path/to/note.md

# Stats
mdvault stats

# Custom DB location
mdvault index ~/notes/ --db ~/vault.db
mdvault search "query" --db ~/vault.db
```

## Claude Code Integration (MCP)

Add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "mdvault": {
      "command": "uvx",
      "args": ["mdvault", "serve"],
      "env": {
        "VAULT_DB": "/absolute/path/to/vault.db"
      }
    }
  }
}
```

If `VAULT_DB` is omitted, defaults to `~/.local/share/mdvault/vault.db` (Linux) or `~/Library/Application Support/mdvault/vault.db` (macOS).

**MCP tools exposed:**

| Tool | Description |
|---|---|
| `search_vault` | Hybrid BM25 + semantic search with optional query expansion |
| `related_notes` | Links, backlinks, and semantically similar files for a given note |

Once connected, ask Claude Code:
- *"search my notes for how I configured SSH tunnels"*
- *"find everything I wrote about postgres replication"*
- *"what notes are related to my kubernetes setup?"*

## How it works

```
Query
  ├── FTS5 BM25 search  → top-50 ranked results
  └── Vector search     → top-50 nearest neighbors
          │
          ▼
    Reciprocal Rank Fusion  (k=60)
    score = 1/(60+rank_bm25) + 1/(60+rank_vec)
          │
          ▼
    Top-N results with file_path + content excerpt
```

**Chunking** splits files on `##`/`###` headings (max 400 words, 50-word overlap, small sections merged). Each chunk is prefixed with its document context (`[path > title > heading]`) before embedding and FTS indexing — this improves retrieval by grounding chunks in their source document.

**Query expansion** (opt-in via `--expand`) calls a local Ollama model to generate a short paragraph that a relevant document might contain, then concatenates it with the original query for vector search. BM25 always uses the original query for precise lexical matching.

## Tech Stack

| Component | Library |
|---|---|
| Embeddings | [model2vec](https://github.com/MinishLab/model2vec) (potion-base-8M) |
| Vector search | [sqlite-vec](https://github.com/asg017/sqlite-vec) |
| Full-text search | SQLite FTS5 |
| MCP server | [mcp](https://github.com/modelcontextprotocol/python-sdk) (official Python SDK) |
| CLI | [typer](https://typer.tiangolo.com) |
| Linter | [ruff](https://github.com/astral-sh/ruff) |
| Query expansion | [Ollama](https://ollama.ai) (optional) |

## Limitations

- English-only (potion-base-8M is primarily trained on English)
- Markdown only (no PDF, DOCX)
- Single vault per instance
- Exact vector search — scales to ~500k chunks on commodity hardware

## Development

```bash
git clone https://github.com/sderosiaux/mdvault
cd mdvault
uv sync --dev
uv run pytest -q
```

Pre-commit hooks (ruff lint + format) are installed automatically:

```bash
uv run pre-commit install
```

## License

MIT
