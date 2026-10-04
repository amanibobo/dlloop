#!/usr/bin/env bash
# Recording 3: one active-learning round on Modal (site slot loop-modal.mp4).
#
# What you will see: the held-out evaluation of the base classifier, the weakest cell it finds,
# the proposed LensCard waiting for your approval (type y), then simulate -> retrain -> evaluate
# with before/after numbers. Everything runs on the deployed Modal app; nothing trains locally.
#
# Timing on Modal: evaluate ~20 s, uncertainty a few seconds, 500 images ~30 s, 3 epochs on an
# L4 ~1 min, evaluate again ~20 s. About 3 minutes total; cut the waits in the edit, or keep the
# Modal dashboard (modal.com/apps -> lenscraft) open beside the terminal while it runs.
#
# Re-takes: TAKE=2 ./record_loop.sh gives the new batch a fresh run id (demo2_r1). The new model
# is always clf_base_b_r1 and is overwritten, which is fine; it is not used anywhere else.
cd "$(dirname "$0")"
TAKE="${TAKE:-1}"
uv run lenscraft loop --backend modal --model-id clf_base_b \
    --train-runs tr5k_none tr5k_sub tr5k_vor --test-runs te1k_none te1k_sub te1k_vor \
    --rounds 1 --n-images 500 --epochs 3 --prefix "demo${TAKE}"
