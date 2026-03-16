# Autoresearch: Multi-Corpus Search Quality

## Objective
Optimize hybrid search quality (Recall@5 + MRR) on two large real-world corpora:
- MDN Web Docs (14K md files, all named index.md, deep directory hierarchy)
- GitHub Docs (3.5K md files, descriptive filenames, moderate depth)

30-query golden set: 15 MDN + 15 GitHub, mix of paraphrase/technical/use-case queries.

## Metrics
- **Primary**: quality_score (points, higher is better) = 50*Recall@5 + 50*MRR
- **Secondary**: recall_at_5, mrr, avg_latency_ms

## How to Run
`./autoresearch.sh` outputs `METRIC name=number` lines.
Log results with `/Users/sderosiaux/.claude/plugins/cache/sderosiaux-claude-plugins/autoresearch/0.6.0/scripts/log-experiment.sh`.

## Files in Scope
- `mdvault/retriever.py` — hybrid search pipeline: BM25, vector, RRF fusion, path re-ranker, suffix dedup
- `mdvault/indexer.py` — chunking, FTS5 indexing, vector embedding, contextual prefix

## Off Limits
- `mdvault/db.py` — schema is fixed
- `autoresearch_gold.json` — golden set (unless provably wrong)

## Constraints
None.

## Corpus Characteristics
- **MDN**: 14K files, ALL named `index.md`. Path is the ONLY filename signal. Deep paths like `web/javascript/reference/global_objects/array/reduce/index.md`. Very structured content (YAML frontmatter, consistent MDN template).
- **GitHub Docs**: 3.5K files with descriptive names. Moderate depth. GitHub-specific domain.
- **Combined**: ~17.5K files, ~150K+ chunks expected.
- No Releases/ duplication issue (unlike PAI corpus).

## Profiling Notes
(To be filled after baseline)

## What's Been Tried
(To be filled as experiments accumulate)
