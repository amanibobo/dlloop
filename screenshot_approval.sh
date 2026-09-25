#!/usr/bin/env bash
# Run this in a terminal, then screenshot the window when the approval prompt appears.
# It asks the agent for a batch on the LOCAL backend (no Modal cost). Type `n` and a reason to
# exit without rendering anything, or `y` to let it run the 20 images (a few seconds).
cd "$(dirname "$0")"
uv run lenscraft agent "Simulate 20 axion vortex images at mass fraction 0.03, seed 7, run id vortex_demo" --backend local --data-dir /tmp/lenscraft-demo
