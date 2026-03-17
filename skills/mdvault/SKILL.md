---
name: mdvault
description: Search your local knowledge vault — past sessions, research, skills, notes, memories. Use when you need to recall prior work, find solutions, look up context, or search documentation. Triggers on "do I have notes on", "what did I do about", "find my", "search vault", "check my history", or any request requiring memory of past work.
---

# mdvault — Local Knowledge Vault

Hybrid search (BM25 + vector + multi-signal re-ranking) across indexed Markdown files, session transcripts, and stored memories.

## When to Use

- Recalling prior work or past conversations
- Finding notes, documentation, or research
- Checking if something was already solved
- Looking up configuration, patterns, or decisions
- Searching across multiple vaults (notes, .claude history, docs)

## Search

```bash
mdvault search "<query>" --top-k 10
```

Results are ranked by a multi-signal pipeline: BM25 lexical match, vector similarity, query term coverage, title/heading/path alignment, and first-chunk topic detection.

**Key flags:**

| Flag | Use when |
|---|---|
| `--top-k 10` | Broad queries — cast a wider net |
| `--source memories` | Only search stored memories |
| `--source files` | Only search indexed files |
| `--vault .claude` | Scope to a specific vault |
| `--paths-only` | Just need file locations, not content |
| `--no-truncate` | Need full chunk content |
| `--expand` | Vague queries — uses local LLM to enrich vector search |
| `--json` | Machine-readable output |

## Read Full Content

```bash
mdvault read "<file_path>"
```

Use after search to get the full content of a result. File paths come from search output.

## Find Related Notes

```bash
mdvault related "<file_path>"
```

Returns outgoing links, backlinks, and semantically similar files. Use to explore context around a result.

## Browse Files

```bash
mdvault list --pattern "*kafka*"
mdvault list --vault .claude --json
```

List indexed files. Filter by vault or glob pattern.

## Memories

```bash
# Store
mdvault remember "important fact" --namespace user/prefs

# List (with confidence, hits, decay status)
mdvault memories --json

# Delete
mdvault forget --id <id>
mdvault forget --namespace drafts
```

Memories are searchable alongside files. They gain confidence when hit by searches and decay over 180 days if unused.

## Knowledge Gaps

```bash
mdvault gaps
```

Shows recurring queries that consistently return poor results. Use to identify what's missing from the vault.

## Typical Workflow

1. **Search** — `mdvault search "kafka interceptors" --top-k 10`
2. **Read** — `mdvault read "<file_path>"` on the best result
3. **Explore** — `mdvault related "<file_path>"` for connected notes
4. **Remember** — `mdvault remember "key finding"` to store for later

## Rules

- Always search before saying "I don't have context on that"
- Use `--top-k 10` or higher for broad or ambiguous queries
- Use `--paths-only` when you just need to locate files, then `read` for content
- Use `--source memories` to check what was previously stored
- Use `--vault` to scope when you know which vault is relevant
- Combine search + read + related for deep exploration
