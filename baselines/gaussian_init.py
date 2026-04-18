"""GaussianInit ablation.

Like Flow-Interpolation but initializes `(noise_optim, pred_optim)` with
Gaussian noise instead of using the encoded source/target latents as the
starting point. This is the "GaussianInit" column in Fig. 5 of the paper;
it illustrates the importance of the encoded-latent starting point.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from flowmorph.flux_optim import FluxOptimizer
from flowmorph.utils import make_image_grid


class GaussianInitSolver(nn.Module):
    def __init__(self, flux: FluxOptimizer, target_latents, timestep_idx: int, guidance=None,
                 init_scale: float = 0.1):
        super().__init__()
        self.flux = flux
        self.target_latents = target_latents.detach()
        self.timestep = flux.timesteps[timestep_idx].to(target_latents.dtype).repeat(target_latents.shape[0])
        self.sigma = flux.pipe.scheduler.sigmas[timestep_idx]
        self.sigma_last = flux.pipe.scheduler.sigmas[-1]
        self.guidance = guidance
        self.noise_optim = nn.Parameter(torch.randn_like(target_latents) * init_scale)
        self.pred_optim = nn.Parameter(torch.randn_like(target_latents) * init_scale)

    def optimize(self, num_steps=100, lr_noise=0.01, lr_pred=0.04, verbose=True):
        opt = torch.optim.AdamW(
            [
                {"params": self.noise_optim, "lr": lr_noise},
                {"params": self.pred_optim, "lr": lr_pred},
            ]
        )
        for i in range(num_steps):
            states = self.pred_optim - (self.sigma_last - self.sigma) * self.noise_optim
            noise_pred = self.flux.predict_noise(states, self.timestep, self.guidance)
            latents_pred = states + (self.sigma_last - self.sigma) * noise_pred
            loss = torch.norm(latents_pred - self.target_latents)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if verbose and (i % 10 == 0 or i == num_steps - 1):
                print(f"  step {i:3d}/{num_steps}: loss={loss.item():.4f}")
        return (self.pred_optim.detach() - (self.sigma_last - self.sigma) * self.noise_optim.detach())


def flow_inference(flux, latents, timestep_indices, guidance=None):
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


def main():
    parser = argparse.ArgumentParser(description="Baseline: GaussianInit optimization")
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--prompt", type=str, default="high quality image")
    parser.add_argument("--output-dir", type=str, default="./outputs/baseline_gaussian_init")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--start-timestep-idx", type=int, default=35)
    parser.add_argument("--inference-timestep-indices", type=str, default="35,55,75,95")
    parser.add_argument("--optim-steps", type=int, default=100)
    parser.add_argument("--noise-lr", type=float, default=0.01)
    parser.add_argument("--pred-lr", type=float, default=0.04)
    parser.add_argument("--init-scale", type=float, default=0.1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir) / datetime.datetime.now().strftime("gaussian_init_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    pil_src = Image.open(args.source).convert("RGB").resize((args.width, args.height))
    pil_tgt = Image.open(args.target).convert("RGB").resize((args.width, args.height))
    pil_src.save(out / "source.png")
    pil_tgt.save(out / "target.png")

    flux = FluxOptimizer(device=device, height=args.height, width=args.width)
    flux.init_prompt(args.prompt, 2)

    with torch.no_grad():
        z_src = flux.img_to_latents(pil_src)
        z_tgt = flux.img_to_latents(pil_tgt)

    print("[src] GaussianInit-based optimization")
    src_states = GaussianInitSolver(flux, z_src, args.start_timestep_idx, args.guidance,
                                    args.init_scale).optimize(
        num_steps=args.optim_steps, lr_noise=args.noise_lr, lr_pred=args.pred_lr)

    print("[tgt] GaussianInit-based optimization")
    tgt_states = GaussianInitSolver(flux, z_tgt, args.start_timestep_idx, args.guidance,
                                    args.init_scale).optimize(
        num_steps=args.optim_steps, lr_noise=args.noise_lr, lr_pred=args.pred_lr)

    ts = [int(s) for s in args.inference_timestep_indices.split(",")]
    alphas = np.linspace(0.0, 1.0, args.num_frames)

    frames = []
    for i, alpha in enumerate(tqdm(alphas, desc="GaussianInit interp")):
        latents = (1 - float(alpha)) * src_states + float(alpha) * tgt_states
        latents = flow_inference(flux, latents, ts, args.guidance)
        img = flux.latents_to_img(latents)
        frames.append(img)
        img.save(out / f"frame_{i:03d}_alpha_{alpha:.2f}.png")

    grid = make_image_grid(frames, row_width=min(len(frames), 10))
    if grid is not None:
        grid.save(out / "interpolation_grid.png")

    with open(out / "config.json", "w") as fp:
        json.dump(vars(args), fp, indent=2)


if __name__ == "__main__":
    main()
