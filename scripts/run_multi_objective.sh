#!/bin/bash
# FlowMorph: Flow-Optimizer with multi-objective composition.
#
# Demonstrates Fig. 6 (supplement): the sum-of-losses objective converges
# to a point inside the overlap of the per-target basins, giving a blended
# attribute output.

set -euo pipefail

python - <<'PY'
from flowmorph.flow_optimizer import multi_objective_flow_optimizer

img, out_dir = multi_objective_flow_optimizer(
    source_path="./data/pairs/multi_source.png",
    target_paths=[
        "./data/pairs/multi_target_1.png",
        "./data/pairs/multi_target_2.png",
    ],
    target_weights=[0.5, 0.5],
    source_prompt="a high quality portrait",
    target_prompts=[
        "same subject with glasses and a hat",
        "same subject in a forest scene",
    ],
    output_dir="./outputs/multi_objective",
    start_timestep_idx=35,
    optim_steps=100,
)
print(f"Saved to: {out_dir}")
PY
