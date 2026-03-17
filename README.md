# mdvault

[![CI](https://github.com/sderosiaux/mdvault/actions/workflows/ci.yml/badge.svg)](https://github.com/sderosiaux/mdvault/actions/workflows/ci.yml)

CLI + MCP server that indexes a folder of Markdown files into a local SQLite database and exposes **hybrid search** (BM25 + semantic + RRF) directly usable from Claude Code.

Zero infrastructure. Everything lives in a single `.db` file.

## Features

- **Hybrid search** — combines FTS5 BM25 and 256-dim vector search via Reciprocal Rank Fusion
- **Contextual chunking** — each chunk is prefixed with `[path > title > heading]` for better retrieval ([Anthropic approach](https://www.anthropic.com/news/contextual-retrieval))
- **Link graph** — parses Markdown links and `[[wikilinks]]`, surfaces backlinks and semantically similar files
- **Query expansion** — optional local LLM (Ollama) expands queries for richer vector search
- **Memory intelligence** — temporal decay, confidence scoring, and automatic promotion of recurring queries into long-term memory
- **Fast embeddings** — [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) via model2vec, CPU-only, ~30MB
- **Incremental indexing** — SHA256-based change detection, only reprocesses modified files
- **MCP server** — Claude Code can search your vault and explore related notes
- **Single file** — one `.db` holds everything (FTS5 index + vectors + link graph + metadata)

## Quick Start

> Paste `install https://github.com/sderosiaux/mdvault` in Claude Code — it will follow these steps.

**1. Install**
```bash
uv tool install mdvault
```

**2. Index your notes**
```bash
mdvault index ~/notes/
```

**3. Add MCP server** to `~/.claude/mcp.json`:
```json
{
  "mcpServers": {
    "mdvault": { "command": "uvx", "args": ["mdvault", "serve"] }
  }
}
```

**4. Restart Claude Code**, then ask: *"search my notes for kubernetes setup"*

### Other install methods

```bash
uvx mdvault --help              # run without installing
pipx install mdvault             # without uv
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
mdvault search "ssh tunnel" --expand --expand-model qwen3:0.6b  # default model

# Related notes: links, backlinks, and semantically similar files
# (file path is relative to the vault root)
mdvault related path/to/note.md

# Stats (includes memory & query analytics)
mdvault stats

# Store a memory (searchable alongside your files)
mdvault remember "Kafka timeout is controlled by max.poll.interval.ms"

# List stored memories (with confidence, hits, decay)
mdvault memories

# Show knowledge gaps (recurring queries with poor results)
mdvault gaps

# Delete a memory
mdvault forget <memory-id>

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
| `store_memory` | Store a memory (auto-chunked, searchable alongside files) |
| `delete_memory` | Delete memories by id or namespace |

Once connected, ask Claude Code:
- *"search my notes for how I configured SSH tunnels"*
- *"find everything I wrote about postgres replication"*
- *"what notes are related to my kubernetes setup?"*

## How it works

```
Query
  ├── FTS5 BM25 search  → top-75 (NEAR bigrams + focused AND clause)
  └── Vector search     → top-75 nearest neighbors
          │
          ▼
    Reciprocal Rank Fusion  (k=15, BM25 weight 4×)
          │
          ▼
    Multi-signal re-ranking
      ├── Cosine similarity (continuous, from vec distance)
      ├── Query term coverage (squared, with bonuses at ≥80% and 100%)
      ├── First-chunk coverage (intro paragraph = topic signal)
      ├── Heading match (H2/H3 heading vs query terms)
      ├── Title match (H1 heading vs query terms)
      ├── Path match (filename + parent dirs vs query terms)
      └── Overview boost (about/intro pages with high coverage)
          │
          ▼
    Content-hash dedup → top-N results
```

**Chunking** splits files on `##`/`###` headings (max 400 words, 50-word overlap, small sections merged). Each chunk is prefixed with its document context (`[path > title > heading]`) before embedding and FTS indexing — this improves retrieval by grounding chunks in their source document.

**Query expansion** (opt-in via `--expand`) calls a local [Ollama](https://ollama.ai) model (default: `qwen3:0.6b`) to generate a short paragraph that a relevant document might contain, then concatenates it with the original query for vector search. BM25 always uses the original query for precise lexical matching. Install Ollama and pull the model with `ollama pull qwen3:0.6b`.

## Memory Intelligence

Memories are searchable alongside your files but ranked by three implicit, metadata-driven signals — no configuration needed.

**Temporal decay** — memories lose relevance over time. Linear decay over 180 days (floor 0.1), reset on each hit. A note you search for regularly stays fresh; one you never touch fades.

**Confidence scoring** — base confidence depends on source (`user`=0.7, `agent`=0.5, `promoted`=0.3) plus a logarithmic hit boost (capped at +0.3). Memories that prove useful gain weight.

**Automatic promotion** — every search is logged. Queries are clustered by embedding similarity (cosine > 0.85). When a cluster reaches 5+ occurrences:
- If results were good (avg score ≥ 0.3), the best result is crystallized as a permanent memory
- If results were poor (avg score < 0.15), a knowledge gap is recorded

```
Search query
  ├── query_log + query_vec (logged)
  ├── hit tracking on memory results
  └── every 20 queries:
        ├── cluster_recent_queries (embedding similarity)
        └── maybe_promote (crystallize or flag gap)
```

Use `mdvault gaps` to surface recurring queries that your vault can't answer well.

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

Install pre-commit hooks (ruff lint + format):

```bash
uv run pre-commit install
```

## License

MIT
