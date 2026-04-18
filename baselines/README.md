# Baselines

All baselines here share the FLUX backbone, scheduler, and VAE decoder
provided by `flowmorph.FluxOptimizer`, so the comparison in Table 1 of the
FlowMorph paper is apples-to-apples.

| Script                  | Row in paper              | Intuition |
|-------------------------|---------------------------|-----------|
| `spherical_interp.py`   | "Spherical Interpolation" | Pure SLERP of the encoded latents; no optimization. |
| `sdedit_interp.py`      | "SDEditInterp"            | SDEdit-style noise injection + SLERP in the noisy latent space. |
| `gaussian_init.py`      | "GaussianInit" (Fig. 5)   | Same two-variable optimization as FlowMorph but with Gaussian initialization instead of the encoded latent. |
| `direct_latent.py`      | Single-variable ablation  | Optimizes the decoded state directly (no `(Delta, u)` split). |

## External baselines

The following baselines are run against the authors' upstream code bases.
Our convention is to clone them into `third_party/` and replace their
sampler/scheduler configuration with the FLUX.1-Depth-dev defaults so the
input/output pipeline matches.

- **IMPUS** — https://github.com/GoL2022/IMPUS
- **DiffMorpher** — https://github.com/Kevin-thu/DiffMorpher
- **FreeMorph** — https://github.com/fast-codi/FreeMorph
- **RF-Inversion** — https://github.com/LituRout/RF-Inversion

See `scripts/baselines/` for runnable shell commands that reproduce the
numbers in Table 1 and the face-pair geometry-preservation experiment.
