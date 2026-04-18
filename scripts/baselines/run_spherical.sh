#!/bin/bash
# Baseline: pure spherical interpolation of encoded latents.
set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-./data/pairs/example_source.png}"
TARGET_IMAGE="${TARGET_IMAGE:-./data/pairs/example_target.png}"

python -m baselines.spherical_interp \
    "$SOURCE_IMAGE" \
    "$TARGET_IMAGE" \
    --prompt "high quality image" \
    --output-dir "./outputs/baseline_spherical" \
    --start-timestep-idx 35 \
    --inference-timestep-indices "35,55,75,95" \
    --num-frames 20
