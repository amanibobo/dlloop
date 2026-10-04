#!/usr/bin/env bash
# Recording 5: generating the report (site slot report.mp4).
#
# Builds the markdown report with figures from what is already on the Modal volume: the 5k
# dataset, the base / active-learning / uniform-control classifiers, the AAE, the saved loop
# history and the saved benchmark summary. Reads records from Modal, so allow about a minute.
# It prints the output path and figure list, then opens the report (Finder's default .md app;
# in VS Code use "Markdown: Open Preview" on reports/demo_report/report.md instead).
cd "$(dirname "$0")"
OUT="reports/demo_report"
uv run lenscraft report --backend modal \
    --runs tr5k_none tr5k_sub tr5k_vor te1k_none te1k_sub te1k_vor uni_none uni_sub uni_vor al_r1 \
    --models clf_base_b clf_al_b clf_uni_b aae_5k_v2 --eval-runs te1k_none te1k_sub te1k_vor \
    --loop-json reports/compare/loop_history.json --bench-summary reports/bench_kimi3/summary.json \
    --title "Dark Matter by Feedback: dataset, models, active-learning round, and benchmark" \
    --out "$OUT"
echo
ls "$OUT"
open "$OUT/report.md"
