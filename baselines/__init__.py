"""Re-implemented baselines used in the FlowMorph paper.

All baselines share the FLUX backbone, scheduler, and VAE decoder provided
by `flowmorph.FluxOptimizer`, so the comparison in Table 1 is apples-to-
apples. For external baselines (IMPUS, DiffMorpher, FreeMorph) we provide a
runner that dispatches to the authors' upstream repositories; see
`baselines/README.md`.
"""
