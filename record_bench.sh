#!/usr/bin/env bash
# Recording 4: tool arm vs core arm (site slot benchmark.mp4).
#
# One trial per arm on the small benchmark task, LOCAL backend (no Modal). The tool arm runs
# first and finishes in about 2 minutes (6 requests). The core arm, which has only a shell and
# file tools, takes 10-15 minutes and ~30 requests; speed that part up in the edit. The last
# thing printed is the summary table (pass@k, stage reach, requests, tokens, time per arm).
#
# Only the tool arm (short recording):  ARMS=tool ./record_bench.sh
# Needs FIREWORKS_API_KEY and LENSCRAFT_MODEL in .env (already there).
cd "$(dirname "$0")"
ARMS="${ARMS:-tool core}"
OUT="reports/bench_demo_$(date +%H%M%S)"
# shellcheck disable=SC2086
uv run lenscraft bench --trials 1 --arms $ARMS --out "$OUT"
echo
echo "written: $OUT/results.jsonl and $OUT/summary.json"
