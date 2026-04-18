#!/bin/bash
# Baseline: GaussianInit optimization (ablates the encoded-latent init of FlowMorph).
set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-./data/pairs/example_source.png}"
TARGET_IMAGE="${TARGET_IMAGE:-./data/pairs/example_target.png}"

python -m baselines.gaussian_init \
    "$SOURCE_IMAGE" \
    "$TARGET_IMAGE" \
    --prompt "high quality image" \
    --output-dir "./outputs/baseline_gaussian_init" \
    --start-timestep-idx 35 \
    --inference-timestep-indices "35,55,75,95" \
    --optim-steps 100 \
    --noise-lr 0.01 \
    --pred-lr 0.04 \
    --init-scale 0.1 \
    --num-frames 20
