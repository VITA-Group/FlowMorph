#!/bin/bash
# Baseline: single-variable direct latent optimization.
set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-./data/pairs/example_source.png}"
TARGET_IMAGE="${TARGET_IMAGE:-./data/pairs/example_target.png}"

python -m baselines.direct_latent \
    "$SOURCE_IMAGE" \
    "$TARGET_IMAGE" \
    --prompt "high quality image" \
    --output-dir "./outputs/baseline_direct_latent" \
    --start-timestep-idx 35 \
    --inference-timestep-indices "35,55,75,95" \
    --optim-steps 100 \
    --latent-lr 0.01 \
    --num-frames 20
