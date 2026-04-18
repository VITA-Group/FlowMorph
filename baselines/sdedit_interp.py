"""SDEditInterp baseline.

Linearly combines source and target latents with Gaussian noise following
the rectified flow scheduler (SDEdit-style), then decodes through the
frozen reverse chain without any local optimization. This is the
"SDEditInterp" row in Table 1, which isolates "noise injection only" from
the full FlowMorph recipe.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from flowmorph.flux_optim import FluxOptimizer
from flowmorph.utils import make_image_grid, slerp


def _sdedit_noise(flux: FluxOptimizer, clean_latents: torch.Tensor, timestep_idx: int,
                  noise_strength: float = 1.0) -> torch.Tensor:
    """Add noise following the scheduler's forward process at timestep_idx."""
    sigma = flux.pipe.scheduler.sigmas[timestep_idx]
    noise = torch.randn_like(clean_latents)
    alpha = 1.0 / torch.sqrt(1.0 + sigma ** 2)
    return alpha * clean_latents + sigma * noise_strength * noise


def flow_inference(flux: FluxOptimizer, latents, timestep_indices, guidance=None):
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
    parser = argparse.ArgumentParser(description="Baseline: SDEditInterp (noise + SLERP)")
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--prompt", type=str, default="high quality image")
    parser.add_argument("--output-dir", type=str, default="./outputs/baseline_sdedit")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--start-timestep-idx", type=int, default=35)
    parser.add_argument("--inference-timestep-indices", type=str, default="35,55,75,95")
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir) / datetime.datetime.now().strftime("sdedit_%Y%m%d_%H%M%S")
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

    # Follow SDEdit: inject scheduler-calibrated noise, then SLERP.
    s_src = _sdedit_noise(flux, z_src, args.start_timestep_idx, args.noise_strength)
    s_tgt = _sdedit_noise(flux, z_tgt, args.start_timestep_idx, args.noise_strength)

    ts = [int(s) for s in args.inference_timestep_indices.split(",")]
    alphas = np.linspace(0.0, 1.0, args.num_frames)

    frames = []
    for i, alpha in enumerate(tqdm(alphas, desc="SDEditInterp")):
        latents = slerp(s_src, s_tgt, float(alpha))
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
