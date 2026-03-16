#!/usr/bin/env bash
set -euo pipefail

# Pre-check: both corpora must exist
[[ -d /tmp/mdn-content/files/en-us ]] || { echo "FATAL: mdn-content not found" >&2; exit 1; }
[[ -d /tmp/github-docs/content ]] || { echo "FATAL: github-docs not found" >&2; exit 1; }

# Run benchmark
.venv/bin/python autoresearch_bench.py 2>/dev/null
