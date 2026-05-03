# mdvault

[![CI](https://github.com/sderosiaux/mdvault/actions/workflows/ci.yml/badge.svg)](https://github.com/sderosiaux/mdvault/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mdvault)](https://pypi.org/project/mdvault/)

Your Markdown notes, searchable in Claude Code.

Index any folder of `.md` files — Obsidian vault, `~/.claude/` history, project docs — and search them in natural language, directly from Claude Code or the terminal. Zero infrastructure. One `.db` file.

## What it does

Search combines FTS5 BM25 and 256-dim vectors, fused with RRF and re-ranked on 7 signals (term coverage, heading match, path match, etc).

Each chunk carries its document context as a prefix (`[path > title > heading]`) before indexing, following [Anthropic's contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) approach.

Other things worth knowing:
- Parses Markdown links and `[[wikilinks]]`, finds backlinks and similar files
- Optional query expansion via local Ollama LLM
- Memories decay over time, gain confidence when hit, and get auto-promoted from recurring queries
- [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) embeddings, CPU-only, ~30MB download
- Incremental indexing (SHA256 change detection, only reprocesses what changed)
- MCP server so Claude Code can search your vault directly
- Everything in one `.db` file (FTS5 index + vectors + link graph + metadata)

## Quick Start

> Paste `install https://github.com/sderosiaux/mdvault` in Claude Code — it will follow these steps.

**1. Install**
```bash
uv tool install mdvault
```

**2. Index your notes**
```bash
mdvault index ~/.claude/

# Keep rotated session logs in the index even after Claude Code deletes them
mdvault index ~/.claude/ --keep-deleted projects
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

### Keep the index fresh

Indexing is incremental (SHA256 change detection). Set up a cron to keep it updated:

```bash
# Every 30 minutes
(crontab -l; echo '*/30 * * * * uvx mdvault index ~/.claude/ 2>/dev/null') | crontab -
```

### Other install methods

```bash
uvx mdvault --help              # run without installing
pipx install mdvault             # without uv
```

## Example

```
$ mdvault search "memory system LLM"

[1] 0.983  .claude/projects/.../ae863d59.jsonl:70
### Dedicated Memory Platforms
- **Mem0**: Universal memory layer. $24M raised (YC-backed).
  41K GitHub stars, 13M+ PyPI downloads...

[2] 0.870  .claude/projects/.../agent-a581b10.jsonl:2
## The mapping: CPU, RAM, disk, I/O
Andrej Karpathy posted in October 2023 that LLMs should be
understood "not as a chatbot, but the kernel process of a new OS."
```

## CLI Usage

```bash
# Index your notes (downloads ~30MB model on first run)
mdvault index ~/.claude/

# Incremental update (only changed/new/deleted files)
mdvault index ~/.claude/ --incremental

# Retain entries when matching files are removed from disk (repeatable).
# Pattern is a path prefix or fnmatch glob, relative to the vault root.
# --full also honors keep-deleted: it rebuilds from disk while preserving
# matching entries. Drop the flag to wipe everything.
mdvault index ~/.claude/ --keep-deleted projects --keep-deleted '*.jsonl'

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
mdvault index ~/.claude/ --db ~/vault.db
mdvault search "query" --db ~/vault.db
```

## Claude Code integration (MCP)

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
| `search_vault` | Hybrid BM25 + semantic search. Filter by `vault`, `source`, `namespace` |
| `related_notes` | Links, backlinks, and semantically similar files for a given note |
| `store_memory` | Store a memory (auto-chunked, searchable alongside files) |
| `delete_memory` | Delete memories by id or namespace |

Then ask Claude things like "search my notes for how I configured SSH tunnels" or "what notes are related to my kubernetes setup?".

## Search pipeline

```
Query
  ├── FTS5 BM25 search  → top-75 (NEAR bigrams + focused AND clause)
  └── Vector search     → top-75 nearest neighbors
          │
          ▼
    Reciprocal Rank Fusion  (k=15, BM25 weight 4×)
          │
          ▼
    Re-ranking (7 signals)
      ├── Cosine similarity (continuous, from vec distance)
      ├── Query term coverage (squared, bonuses at ≥80% and 100%)
      ├── First-chunk coverage (intro paragraph = topic signal)
      ├── Heading match (H2/H3 vs query terms)
      ├── Title match (H1 vs query terms)
      ├── Path match (filename + parent dirs vs query terms)
      └── Overview boost (about/intro pages with high coverage)
          │
          ▼
    Content-hash dedup → top-N results
```

Files are split on `##`/`###` headings (max 400 words, 50-word overlap, small sections merged). Each chunk gets a context prefix (`[path > title > heading]`) before embedding and FTS indexing.

Query expansion (`--expand`) calls a local [Ollama](https://ollama.ai) model (default: `qwen3:0.6b`) to generate a paragraph a relevant document might contain, then appends it to the original query for vector search. BM25 always uses the raw query. Pull the model with `ollama pull qwen3:0.6b`.

## Memories

Memories are searchable alongside files. Three things happen behind the scenes:

**Decay.** Memories fade over 180 days (floor 0.1). Every search hit resets the clock. Search for something regularly and it stays relevant. Stop, and it drops in ranking.

**Confidence.** Base score depends on source (`user`=0.7, `agent`=0.5, `promoted`=0.3) plus a log hit boost (capped at +0.3). Memories that keep getting matched climb higher.

**Auto-promotion.** Every search is logged and clustered by embedding similarity (cosine > 0.85). When a cluster hits 5+ occurrences:
- Good results (avg score >= 0.3): the best result becomes a permanent memory
- Bad results (avg score < 0.15): a knowledge gap is recorded

```
Search query
  ├── query_log + query_vec (logged)
  ├── hit tracking on memory results
  └── every 20 queries:
        ├── cluster_recent_queries (embedding similarity)
        └── maybe_promote (crystallize or flag gap)
```

Run `mdvault gaps` to see what you keep searching for but can't find.

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

- English-optimized: potion-base-8M is trained mostly on English. Semantic search degrades on other languages (BM25 keyword search still works)
- Markdown only (no PDF, DOCX)
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
