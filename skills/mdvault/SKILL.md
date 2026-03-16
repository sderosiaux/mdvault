---
name: mdvault
description: Search your local knowledge vault — past sessions, research, skills, notes. Use when you need to recall prior work, find solutions, look up context from past conversations, or search documentation. Triggers on "do I have notes on", "what did I do about", "find my", "search vault", "check my history", or any request requiring memory of past work.
---

# mdvault — Local Knowledge Vault

Search across indexed Markdown files and Claude Code session transcripts.

## Quick Search

```bash
mdvault search "<query>" --top-k 10
```

Returns file paths with content previews, ranked by relevance (hybrid BM25 + vector).

## Read a Result

```bash
mdvault read "<file_path>"
```

Returns the full content of an indexed file. Use after search to get details.

## Filtered Search

```bash
# Only memories (stored via remember)
mdvault search "<query>" --source memories

# Only a specific vault
mdvault search "<query>" --vault .claude

# Just file paths (deduped)
mdvault search "<query>" --paths-only

# Full content in results (no truncation)
mdvault search "<query>" --no-truncate
```

## JSON for Structured Use

All commands support `--json` for machine-readable output:

```bash
mdvault search "<query>" --top-k 5 --json
mdvault read "<file_path>" --json
mdvault list --json
mdvault stats --json
```

## Workflow

1. **Search** — find relevant files: `mdvault search "kafka interceptors" --top-k 10`
2. **Read** — get full content: `mdvault read ".claude/history/research/...md"`
3. **List** — browse indexed files: `mdvault list --pattern "*kafka*"`
4. **Related** — find connected notes: `mdvault related "<file_path>"`

## Store Memories

```bash
mdvault remember "important fact" --namespace user/prefs
mdvault memories --json
mdvault forget --id <id>
```

## Rules

- Always search before saying "I don't have context on that"
- Use `--top-k 10` or higher for broad queries
- Use `--paths-only` when you just need to locate files
- Use `read` to get full content after finding relevant results
- Combine with `--vault` to scope search to a specific vault
