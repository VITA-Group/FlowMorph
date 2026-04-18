#!/bin/bash
# FlowMorph: Flow-Optimizer (direct target fitting).
#
# Reproduces the geometry-preservation runs used in Fig. 3 of the paper.
# The optimizer starts from the source latent and minimizes one-step
# reconstruction loss against the target latent at a single noise level.

set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-./data/pairs/example_source.png}"
TARGET_IMAGE="${TARGET_IMAGE:-./data/pairs/example_target.png}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/flow_optimizer}"

python -m flowmorph.flow_optimizer \
    "$SOURCE_IMAGE" \
    "$TARGET_IMAGE" \
    --source-prompt "a photorealistic portrait, high quality" \
    --target-prompt "a photorealistic portrait, high quality" \
    --output-dir "$OUTPUT_DIR" \
    --start-timestep-idx 35 \
    --inference-timestep-indices "35,55,75,95" \
    --optim-steps 100 \
    --noise-lr 0.01 \
    --pred-lr 0.04 \
    --sampling-count 10 \
    --num-frames 20
