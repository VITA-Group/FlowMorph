#!/bin/bash
# Ablation: separate vs. shared optimizer and prompts for source / target.
#
# Section 5: "Prompt conditioning and optimizer separation". Running with
# --no-separate-optimizers reuses the source-prompt FluxOptimizer for the
# target optimization; this is cheaper but typically less smooth.

set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-./data/pairs/example_source.png}"
TARGET_IMAGE="${TARGET_IMAGE:-./data/pairs/example_target.png}"

echo "=== Separate optimizer + prompt per endpoint (default) ==="
python -m flowmorph.flow_interpolation \
    "$SOURCE_IMAGE" \
    "$TARGET_IMAGE" \
    --source-prompt "a cute fluffy cat, high quality photo" \
    --target-prompt "a playful robot dog, high quality photo" \
    --output-dir "./outputs/ablation_separated" \
    --num-frames 20 \
    --start-timestep-idx 35 \
    --optim-steps 100 \
    --mixing decoupled

echo "=== Shared optimizer, single prompt ==="
python -m flowmorph.flow_interpolation \
    "$SOURCE_IMAGE" \
    "$TARGET_IMAGE" \
    --source-prompt "high quality image" \
    --target-prompt "high quality image" \
    --output-dir "./outputs/ablation_shared" \
    --num-frames 20 \
    --start-timestep-idx 35 \
    --optim-steps 100 \
    --mixing decoupled \
    --no-separate-optimizers
