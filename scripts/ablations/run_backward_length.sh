#!/bin/bash
# Ablation on the backward length delta_sigma = sigma_last - sigma_{t_i}.
#
# Section 5: "Backward length". Larger delta_sigma (smaller t_i) injects a
# stronger semantic signal into u and unlocks richer morphs beyond trivial
# shape blending. Here we sweep t_i in {20, 35, 55, 75}.

set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-./data/pairs/example_source.png}"
TARGET_IMAGE="${TARGET_IMAGE:-./data/pairs/example_target.png}"

for T_I in 20 35 55 75; do
    echo "=== Flow-Interpolation with start_timestep_idx=${T_I} ==="
    python -m flowmorph.flow_interpolation \
        "$SOURCE_IMAGE" \
        "$TARGET_IMAGE" \
        --source-prompt "high quality image" \
        --target-prompt "high quality image" \
        --output-dir "./outputs/ablation_backward_length_ti${T_I}" \
        --num-frames 20 \
        --start-timestep-idx "$T_I" \
        --optim-steps 100 \
        --mixing decoupled
done
