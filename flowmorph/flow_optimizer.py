"""Flow-Optimizer: direct target fitting.

Starting from the source latent at a single noise level t_i, we optimize the
two-variable parameterization (Delta, u) so that the one-step reconstruction
approximates the target latent (Eq. 8 of the paper). This supports
multi-objective compositions by summing losses across several targets.
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
from PIL import Image
from tqdm import tqdm

from flowmorph.flux_optim import FluxOptimizer
from flowmorph.utils import make_image_grid


def _default_timestep_chain(start_idx: int):
    """Pick a short deterministic reverse chain for decoding."""
    if start_idx <= 35:
        return [35, 55, 75, 95]
    if start_idx <= 55:
        return [55, 75, 95]
    return [start_idx]


def _flow_inference(flux_optimizer: FluxOptimizer, initial_latents, timestep_indices, guidance=None):
    """Advance `initial_latents` deterministically along the scheduler."""
    latents = initial_latents
    for i in range(len(timestep_indices) - 1):
        t_idx = timestep_indices[i]
        next_t_idx = timestep_indices[i + 1]
        timestep = flux_optimizer.timesteps[t_idx].to(latents.dtype).repeat(latents.shape[0])
        sigma = flux_optimizer.pipe.scheduler.sigmas[t_idx]
        next_sigma = flux_optimizer.pipe.scheduler.sigmas[next_t_idx]
        with torch.no_grad():
            noise_pred = flux_optimizer.predict_noise(latents, timestep, guidance)
            latents = latents + (next_sigma - sigma) * noise_pred

    last_t_idx = timestep_indices[-1]
    timestep = flux_optimizer.timesteps[last_t_idx].to(latents.dtype).repeat(latents.shape[0])
    sigma = flux_optimizer.pipe.scheduler.sigmas[last_t_idx]
    final_sigma = flux_optimizer.pipe.scheduler.sigmas[-1]
    with torch.no_grad():
        noise_pred = flux_optimizer.predict_noise(latents, timestep, guidance)
        latents = latents + (final_sigma - sigma) * noise_pred
    return latents


def flow_optimizer(
    source_path: str,
    target_path: str,
    source_prompt: str = "high quality image",
    target_prompt: str = "high quality image",
    output_dir: str = "./outputs/flow_optimizer",
    num_frames: int = 20,
    init_inference_steps: int = 2,
    start_timestep_idx: int = 35,
    inference_timestep_indices: list | None = None,
    optim_steps: int = 100,
    noise_lr: float = 0.01,
    pred_lr: float = 0.04,
    sampling_count: int = 10,
    guidance: float | None = None,
    height: int = 512,
    width: int = 512,
    device: str | None = None,
):
    """Fit the source latent toward the target latent and return the
    optimization trajectory (decoded at every iteration).

    Per the paper, we freeze the FLUX backbone and only optimize
    `noise_optim` (= u) and `pred_optim` (= z + Delta). Multi-objective
    compositions are obtained by summing losses over several targets; see
    `multi_objective_flow_optimizer` in this module.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(output_dir) / datetime.datetime.now().strftime("flow_optimizer_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    pil_src = Image.open(source_path).convert("RGB").resize((width, height))
    pil_tgt = Image.open(target_path).convert("RGB").resize((width, height))
    pil_src.save(out / "source.png")
    pil_tgt.save(out / "target.png")

    flux = FluxOptimizer(device=device, height=height, width=width)
    flux.init_prompt(source_prompt, init_inference_steps)

    flux.init_model(
        pil_src,
        pil_tgt,
        sampling_count=sampling_count,
        timestep_idx=start_timestep_idx,
        guidance=guidance,
        noise_lr=noise_lr,
        pred_lr=pred_lr,
    )

    traj = []
    for i in tqdm(range(optim_steps), desc="Flow-Optimizer"):
        loss = flux.step()
        if i % max(1, optim_steps // max(num_frames, 1)) == 0 or i == optim_steps - 1:
            traj.append(flux.latents_to_img(flux.pred_optim.detach()))

    # Push the final state through the reverse chain for a clean decode.
    inference_timestep_indices = inference_timestep_indices or _default_timestep_chain(start_timestep_idx)
    states = flux.pred_optim.detach() - (flux.pipe.scheduler.sigmas[-1] - flux.sigma) * flux.noise_optim.detach()
    final_latents = _flow_inference(flux, states, inference_timestep_indices, guidance)
    final_img = flux.latents_to_img(final_latents)
    final_img.save(out / "final.png")
    traj.append(final_img)

    grid = make_image_grid(traj, row_width=min(len(traj), 10))
    if grid is not None:
        grid.save(out / "trajectory.png")

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
                "guidance": guidance,
                "height": height,
                "width": width,
            },
            fp,
            indent=2,
        )

    return traj, out


def multi_objective_flow_optimizer(
    source_path: str,
    target_paths: list,
    target_weights: list | None = None,
    source_prompt: str = "high quality image",
    target_prompts: list | None = None,
    output_dir: str = "./outputs/multi_objective",
    start_timestep_idx: int = 35,
    optim_steps: int = 100,
    noise_lr: float = 0.01,
    pred_lr: float = 0.04,
    sampling_count: int = 10,
    guidance: float | None = None,
    height: int = 512,
    width: int = 512,
    device: str | None = None,
):
    """Sum-of-losses variant of Flow-Optimizer (paper Section 5 / Fig. multiobject).

    `target_paths` defines the basin we want to reach; `target_weights`
    lets us blend their losses, producing mixtures of attributes from each
    target within the local optimizable domain.
    """
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(output_dir) / datetime.datetime.now().strftime("multi_obj_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    target_weights = target_weights or [1.0 / len(target_paths)] * len(target_paths)
    target_prompts = target_prompts or [source_prompt] * len(target_paths)
    assert len(target_paths) == len(target_weights) == len(target_prompts)

    pil_src = Image.open(source_path).convert("RGB").resize((width, height))
    pil_src.save(out / "source.png")
    pil_targets = [Image.open(p).convert("RGB").resize((width, height)) for p in target_paths]
    for i, img in enumerate(pil_targets):
        img.save(out / f"target_{i}.png")

    flux = FluxOptimizer(device=device, height=height, width=width)
    flux.init_prompt(source_prompt, 2)

    # Initialize (Delta, u) from the source latent against the first target.
    flux.init_model(
        pil_src,
        pil_targets[0],
        sampling_count=sampling_count,
        timestep_idx=start_timestep_idx,
        guidance=guidance,
        noise_lr=noise_lr,
        pred_lr=pred_lr,
    )

    with torch.no_grad():
        tgt_latents_list = [flux.img_to_latents(img).detach() for img in pil_targets]

    sigma_last = flux.pipe.scheduler.sigmas[-1]
    for i in tqdm(range(optim_steps), desc="Multi-objective Flow-Optimizer"):
        states_pred = flux.pred_optim - (sigma_last - flux.sigma) * flux.noise_optim
        latents_pred, _ = flux.predict_latents(states_pred, flux.timestep, flux.sigma, guidance)
        loss = sum(
            w * torch.norm(latents_pred - z) for w, z in zip(target_weights, tgt_latents_list)
        )
        loss.backward()
        flux.optimizer.step()
        flux.optimizer.zero_grad()

    final_img = flux.latents_to_img(flux.pred_optim.detach())
    final_img.save(out / "final.png")
    with open(out / "config.json", "w") as fp:
        json.dump(
            {
                "source": str(source_path),
                "target_paths": [str(p) for p in target_paths],
                "target_weights": target_weights,
                "target_prompts": target_prompts,
                "start_timestep_idx": start_timestep_idx,
                "optim_steps": optim_steps,
                "noise_lr": noise_lr,
                "pred_lr": pred_lr,
            },
            fp,
            indent=2,
        )
    return final_img, out


def main():
    parser = argparse.ArgumentParser(description="FlowMorph Flow-Optimizer (direct target fitting)")
    parser.add_argument("source", type=str, help="Path to source image")
    parser.add_argument("target", type=str, help="Path to target image")
    parser.add_argument("--source-prompt", type=str, default="high quality image")
    parser.add_argument("--target-prompt", type=str, default="high quality image")
    parser.add_argument("--output-dir", type=str, default="./outputs/flow_optimizer")
    parser.add_argument("--start-timestep-idx", type=int, default=35)
    parser.add_argument("--inference-timestep-indices", type=str, default=None,
                        help="Comma-separated indices, e.g. '35,55,75,95'")
    parser.add_argument("--optim-steps", type=int, default=100)
    parser.add_argument("--noise-lr", type=float, default=0.01)
    parser.add_argument("--pred-lr", type=float, default=0.04)
    parser.add_argument("--sampling-count", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=20,
                        help="How many optimization-trajectory snapshots to save")
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.source):
        sys.exit(f"Source not found: {args.source}")
    if not os.path.exists(args.target):
        sys.exit(f"Target not found: {args.target}")

    inference_timestep_indices = None
    if args.inference_timestep_indices:
        inference_timestep_indices = [int(s) for s in args.inference_timestep_indices.split(",")]

    flow_optimizer(
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
        sampling_count=args.sampling_count,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
        device=args.device,
    )


if __name__ == "__main__":
    main()
