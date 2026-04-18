# FlowMorph method in one page

FlowMorph exposes a locally-optimizable neighborhood in the latent space of
a frozen rectified flow model by re-parameterizing that neighborhood with
two small tensors:

- `Delta` -- a geometry offset that controls shape,
- `u`     -- a one-step vector that carries semantics.

At a single noise level `t_i` with scheduler sigma `sigma_i` (and terminal
sigma `sigma_last`), the local state is

    s(Delta, u) = (z_{t_i} + Delta) - (sigma_last - sigma_i) * u

and the one-step Tweedie reconstruction is

    z_hat_{t_i} = s(Delta, u) + (sigma_last - sigma_i) * v_theta(s, t_i),

where `v_theta` is the frozen flow transformer.

## Flow-Optimizer

Fit `(Delta, u)` to drive the reconstruction toward a target latent:

    min_{Delta, u} || s + (sigma_last - sigma_i) * v_theta(s, t_i) - z_tgt ||_2^2

The loss sums naturally across multiple targets for multi-objective edits.

## Flow-Interpolation

Fit `(Delta*, u*)` for source and target independently, then interpolate:

    Delta_alpha = (1 - alpha) * Delta_src + alpha * Delta_tgt           (linear)
    u_hat_alpha = slerp(u_src / ||u_src||, u_tgt / ||u_tgt||; alpha)     (direction)
    u_alpha     = ((1 - alpha)||u_src|| + alpha||u_tgt||) * u_hat_alpha   (magnitude)
    s_alpha     = (z^alpha_{t_i} + Delta_alpha) - delta_sigma * u_alpha

Push `s_alpha` deterministically through the scheduler to `sigma_last` and
decode to RGB.

## Why decoupled mixing?

- Geometry is approximately Euclidean in the offset space (`Delta`), so
  linear mixing keeps the shape smooth with no jitter.
- Semantic directions are more spherical (`u` concentrates on a shell);
  SLERP on the direction preserves the norm trajectory while the explicit
  norm lerp controls the magnitude.

## Hyperparameters

| Symbol       | Value | Notes |
|--------------|-------|-------|
| `t_i`        | 35    | Starting index; longer backward step (smaller t_i) increases semantic capacity of u |
| optim steps  | 100   | AdamW |
| `lr_u`       | 0.01  | (= `noise_lr` in code) |
| `lr_{z+Delta}`| 0.04 | (= `pred_lr` in code) |
| inference chain | [35, 55, 75, 95] | short deterministic chain to `sigma_last` |
