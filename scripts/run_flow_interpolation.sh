#!/bin/bash
# FlowMorph: Flow-Interpolation (decoupled Delta + u mixing).
#
# Reproduces the main Table 1 / Fig. 4 results. We fit (Delta, u) for both
# endpoints with separate optimizer instances (paper's recipe), then mix
# them with the decoupled linear/SLERP policy.

set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-./data/pairs/example_source.png}"
TARGET_IMAGE="${TARGET_IMAGE:-./data/pairs/example_target.png}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/flow_interp}"
NUM_FRAMES="${NUM_FRAMES:-20}"

python -m flowmorph.flow_interpolation \
    "$SOURCE_IMAGE" \
    "$TARGET_IMAGE" \
    --source-prompt "a cute fluffy cat, high quality photo" \
    --target-prompt "a playful robot dog, high quality photo" \
    --output-dir "$OUTPUT_DIR" \
    --num-frames "$NUM_FRAMES" \
    --start-timestep-idx 35 \
    --inference-timestep-indices "35,55,75,95" \
    --optim-steps 100 \
    --noise-lr 0.01 \
    --pred-lr 0.04 \
    --mixing decoupled
