"""Flow-Interpolation: decoupled geometry / semantics mixing.

Given optimized (Delta*, u*) for source and target (obtained by one-step
reconstruction at a single noise level t_i, Eq. 8 of the paper), this module
mixes them with:

    Delta_alpha = (1 - alpha) * Delta_src + alpha * Delta_tgt           (linear)
    u_alpha     = ((1 - alpha)||u_src|| + alpha||u_tgt||)                 (norm lerp
                  * slerp(u_src/||u_src||, u_tgt/||u_tgt||; alpha)         + direction SLERP)

Paper Eq. (12)-(14). This linear-for-geometry, spherical-for-semantics policy
avoids shape jitter while keeping the direction change smooth, which is the
decoupled mixing policy used for the reported quantitative results.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from flowmorph.flux_optim import FluxOptimizer
from flowmorph.utils import make_image_grid, slerp_direction


class OneStepReconstructor(nn.Module):
    """Reconstruct one image into its (Delta, u) solution.

    Parameters mirror the paper notation:
        `pred_optim`  = z^(y)_{t_i} + Delta
        `noise_optim` = u

    Minimizes Eq. (8): ||z_hat - z||^2 + lambdas. We keep the regularizers
    implicit (0) by default; set `lambda_delta` / `lambda_u` to enable.
    """

    def __init__(
        self,
        flux: FluxOptimizer,
        target_latents: torch.Tensor,
        timestep_idx: int,
        guidance=None,
        lambda_delta: float = 0.0,
        lambda_u: float = 0.0,
    ):
        super().__init__()
        self.flux = flux
        self.target_latents = target_latents.detach()
        self.lambda_delta = lambda_delta
        self.lambda_u = lambda_u

        self.timestep = flux.timesteps[timestep_idx].to(target_latents.dtype)
        self.timestep = self.timestep.repeat(target_latents.shape[0])
        self.sigma = flux.pipe.scheduler.sigmas[timestep_idx]
        self.sigma_last = flux.pipe.scheduler.sigmas[-1]
        self.guidance = guidance

        # Initialize: Delta = 0 (so pred_optim = z) and u = 0.
        self.noise_optim = nn.Parameter(torch.zeros_like(target_latents))
        self.pred_optim = nn.Parameter(torch.clone(target_latents))

    def forward(self):
        states_pred = self.pred_optim - (self.sigma_last - self.sigma) * self.noise_optim
        with torch.enable_grad():
            noise_pred = self.flux.predict_noise(states_pred, self.timestep, self.guidance)
            latents_pred = states_pred + (self.sigma_last - self.sigma) * noise_pred

        recon = torch.norm(latents_pred - self.target_latents)
        reg = (
            self.lambda_delta * torch.norm(self.pred_optim - self.target_latents) ** 2
            + self.lambda_u * torch.norm(self.noise_optim) ** 2
        )
        return recon + reg, latents_pred, states_pred

    def optimize(self, num_steps: int = 100, lr_noise: float = 0.01, lr_pred: float = 0.04, verbose: bool = True):
        opt = torch.optim.AdamW(
            [
                {"params": self.noise_optim, "lr": lr_noise},
                {"params": self.pred_optim, "lr": lr_pred},
            ]
        )
        losses = []
        for i in range(num_steps):
            loss, _, _ = self.forward()
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
            if verbose and (i % 10 == 0 or i == num_steps - 1):
                print(f"  step {i:3d}/{num_steps}: loss={loss.item():.4f}")
        return losses


def flow_inference(flux: FluxOptimizer, initial_latents, timestep_indices, guidance=None):
    latents = initial_latents
    for i in range(len(timestep_indices) - 1):
        t_idx = timestep_indices[i]
        next_t_idx = timestep_indices[i + 1]
        timestep = flux.timesteps[t_idx].to(latents.dtype).repeat(latents.shape[0])
        sigma = flux.pipe.scheduler.sigmas[t_idx]
        next_sigma = flux.pipe.scheduler.sigmas[next_t_idx]
        with torch.no_grad():
            noise_pred = flux.predict_noise(latents, timestep, guidance)
            latents = latents + (next_sigma - sigma) * noise_pred

    last_t_idx = timestep_indices[-1]
    timestep = flux.timesteps[last_t_idx].to(latents.dtype).repeat(latents.shape[0])
    sigma = flux.pipe.scheduler.sigmas[last_t_idx]
    final_sigma = flux.pipe.scheduler.sigmas[-1]
    with torch.no_grad():
        noise_pred = flux.predict_noise(latents, timestep, guidance)
        latents = latents + (final_sigma - sigma) * noise_pred
    return latents


def flow_interpolation(
    source_path: str,
    target_path: str,
    source_prompt: str = "high quality image",
    target_prompt: str = "high quality image",
    output_dir: str = "./outputs/flow_interp",
    num_frames: int = 20,
    init_inference_steps: int = 2,
    start_timestep_idx: int = 35,
    inference_timestep_indices: list | None = None,
    optim_steps: int = 100,
    noise_lr: float = 0.01,
    pred_lr: float = 0.04,
    mixing: str = "decoupled",
    guidance: float | None = None,
    height: int = 512,
    width: int = 512,
    device: str | None = None,
    separate_optimizers: bool = True,
):
    """Flow-Interpolation pipeline.

    We keep the backbone frozen. For each endpoint we fit (Delta, u) via
    `OneStepReconstructor`, then mix along `alpha`. `mixing` controls the
    blending policy used for the ablation in Section 5:

      - "decoupled":  linear Delta + SLERP u           (paper default)
      - "linear":     linear Delta + linear u          (ablates SLERP on u)
      - "slerp":      SLERP Delta + SLERP u            (ablates linear on Delta)
      - "linear_states": linear mix of the decoded states s_alpha (baseline)

    `separate_optimizers=True` uses a distinct FluxOptimizer instance for
    source and target (matching the paper's "optimizer separation" recipe).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(output_dir) / datetime.datetime.now().strftime("flow_interp_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    pil_src = Image.open(source_path).convert("RGB").resize((width, height))
    pil_tgt = Image.open(target_path).convert("RGB").resize((width, height))
    pil_src.save(out / "source.png")
    pil_tgt.save(out / "target.png")

    inference_timestep_indices = inference_timestep_indices or [start_timestep_idx, 55, 75, 95]
    inference_timestep_indices = sorted(set(inference_timestep_indices))
    if inference_timestep_indices[0] != start_timestep_idx:
        inference_timestep_indices = [start_timestep_idx] + [t for t in inference_timestep_indices if t > start_timestep_idx]

    def _fit(prompt: str, pil_img: Image.Image) -> tuple[FluxOptimizer, torch.Tensor, torch.Tensor, torch.Tensor]:
        flux = FluxOptimizer(device=device, height=height, width=width)
        flux.init_prompt(prompt, init_inference_steps)
        with torch.no_grad():
            z = flux.img_to_latents(pil_img)
        solver = OneStepReconstructor(flux, z, start_timestep_idx, guidance)
        solver.optimize(num_steps=optim_steps, lr_noise=noise_lr, lr_pred=pred_lr)
        delta = (solver.pred_optim.detach() - z)
        u = solver.noise_optim.detach()
        return flux, z, delta, u

    print(f"[src] optimizing with prompt: {source_prompt!r}")
    flux_src, z_src, delta_src, u_src = _fit(source_prompt, pil_src)

    if separate_optimizers:
        print(f"[tgt] optimizing with prompt: {target_prompt!r}")
        flux_tgt, z_tgt, delta_tgt, u_tgt = _fit(target_prompt, pil_tgt)
    else:
        flux_src.init_prompt(target_prompt, init_inference_steps)
        with torch.no_grad():
            z_tgt = flux_src.img_to_latents(pil_tgt)
        solver = OneStepReconstructor(flux_src, z_tgt, start_timestep_idx, guidance)
        solver.optimize(num_steps=optim_steps, lr_noise=noise_lr, lr_pred=pred_lr)
        delta_tgt = (solver.pred_optim.detach() - z_tgt)
        u_tgt = solver.noise_optim.detach()
        flux_tgt = flux_src

    sigma_idx = flux_src.pipe.scheduler.sigmas[start_timestep_idx]
    sigma_last = flux_src.pipe.scheduler.sigmas[-1]
    dsigma = sigma_last - sigma_idx

    alphas = np.linspace(0.0, 1.0, num_frames)
    frames = []
    for i, alpha in enumerate(tqdm(alphas, desc=f"Flow-Interpolation ({mixing})")):
        alpha = float(alpha)

        # Mix Delta and u according to the selected policy.
        if mixing == "decoupled":
            delta_a = (1 - alpha) * delta_src + alpha * delta_tgt
            u_a = slerp_direction(u_src, u_tgt, alpha)
        elif mixing == "linear":
            delta_a = (1 - alpha) * delta_src + alpha * delta_tgt
            u_a = (1 - alpha) * u_src + alpha * u_tgt
        elif mixing == "slerp":
            delta_a = slerp_direction(delta_src, delta_tgt, alpha)
            u_a = slerp_direction(u_src, u_tgt, alpha)
        elif mixing == "linear_states":
            # Baseline: mix the full `states = (z + Delta) - dsigma * u` linearly.
            s_src = (z_src + delta_src) - dsigma * u_src
            s_tgt = (z_tgt + delta_tgt) - dsigma * u_tgt
            states = (1 - alpha) * s_src + alpha * s_tgt
            latents = flow_inference(flux_src, states, inference_timestep_indices, guidance)
            img = flux_src.latents_to_img(latents)
            frames.append(img)
            img.save(out / f"frame_{i:03d}_alpha_{alpha:.2f}.png")
            continue
        else:
            raise ValueError(f"Unknown mixing mode: {mixing}")

        z_alpha = (1 - alpha) * z_src + alpha * z_tgt
        states = (z_alpha + delta_a) - dsigma * u_a
        latents = flow_inference(flux_src, states, inference_timestep_indices, guidance)
        img = flux_src.latents_to_img(latents)
        frames.append(img)
        img.save(out / f"frame_{i:03d}_alpha_{alpha:.2f}.png")

    grid = make_image_grid(frames, row_width=min(len(frames), 10))
    if grid is not None:
        grid.save(out / "interpolation_grid.png")

    with open(out / "config.json", "w") as fp:
        json.dump(
            {
                "source": str(source_path),
                "target": str(target_path),
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "start_timestep_idx": start_timestep_idx,
                "inference_timestep_indices": inference_timestep_indices,
                "optim_steps": optim_steps,
                "noise_lr": noise_lr,
                "pred_lr": pred_lr,
                "mixing": mixing,
                "num_frames": num_frames,
                "separate_optimizers": separate_optimizers,
            },
            fp,
            indent=2,
        )
    return frames, out


def main():
    parser = argparse.ArgumentParser(description="FlowMorph Flow-Interpolation")
    parser.add_argument("source", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("--source-prompt", type=str, default="high quality image")
    parser.add_argument("--target-prompt", type=str, default="high quality image")
    parser.add_argument("--output-dir", type=str, default="./outputs/flow_interp")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--start-timestep-idx", type=int, default=35)
    parser.add_argument("--inference-timestep-indices", type=str, default=None)
    parser.add_argument("--optim-steps", type=int, default=100)
    parser.add_argument("--noise-lr", type=float, default=0.01)
    parser.add_argument("--pred-lr", type=float, default=0.04)
    parser.add_argument("--mixing", type=str, default="decoupled",
                        choices=["decoupled", "linear", "slerp", "linear_states"])
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-separate-optimizers", dest="separate_optimizers",
                        action="store_false")
    parser.set_defaults(separate_optimizers=True)
    args = parser.parse_args()

    if not os.path.exists(args.source):
        sys.exit(f"Source not found: {args.source}")
    if not os.path.exists(args.target):
        sys.exit(f"Target not found: {args.target}")

    inference_timestep_indices = None
    if args.inference_timestep_indices:
        inference_timestep_indices = [int(s) for s in args.inference_timestep_indices.split(",")]

    flow_interpolation(
        source_path=args.source,
        target_path=args.target,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        start_timestep_idx=args.start_timestep_idx,
        inference_timestep_indices=inference_timestep_indices,
        optim_steps=args.optim_steps,
        noise_lr=args.noise_lr,
        pred_lr=args.pred_lr,
        mixing=args.mixing,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
        device=args.device,
        separate_optimizers=args.separate_optimizers,
    )


if __name__ == "__main__":
    main()
