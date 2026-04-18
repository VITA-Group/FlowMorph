#!/bin/bash
# Ablation on the mixing strategy for (Delta, u).
#
# Section 5: "Mixing strategy". Confirms that replacing SLERP on u with
# linear mixing degrades semantic consistency and replacing linear mixing
# on Delta with SLERP introduces shape jitter.

set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-./data/pairs/example_source.png}"
TARGET_IMAGE="${TARGET_IMAGE:-./data/pairs/example_target.png}"

for MIX in decoupled linear slerp linear_states; do
    echo "=== Flow-Interpolation with mixing=${MIX} ==="
    python -m flowmorph.flow_interpolation \
        "$SOURCE_IMAGE" \
        "$TARGET_IMAGE" \
        --source-prompt "high quality image" \
        --target-prompt "high quality image" \
        --output-dir "./outputs/ablation_mixing_${MIX}" \
        --num-frames 20 \
        --start-timestep-idx 35 \
        --optim-steps 100 \
        --mixing "$MIX"
done
