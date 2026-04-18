"""FlowMorph base optimizer.

Implements the two-variable parameterization used by the paper:

    s(Delta, u) = (z_{t_i} + Delta) - (sigma_last - sigma_i) * u

and the one-step reconstruction

    z_hat_{t_i} = s(Delta, u) + (sigma_last - sigma_i) * v_theta(s, t_i)

`Delta` is a geometry offset and `u` is the one-step semantic vector. In this
implementation `pred_optim` plays the role of `z_{t_i} + Delta` and
`noise_optim` plays the role of `u`, so that `states` can be written as
`pred_optim - (sigma_last - sigma_i) * noise_optim` directly.

The backbone `v_theta` is the frozen FLUX transformer and is never trained.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from .pipeline_flux import FluxPipeline


def image_pt2pil(img: torch.Tensor) -> Image.Image:
    arr = img.permute(1, 2, 0).detach().cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


class FluxOptimizer(nn.Module):
    """Wraps a frozen FLUX model and exposes the (Delta, u) optimization.

    The defaults follow the paper: FLUX.1-schnell backbone, 100-step scheduler,
    512x512 resolution. Swap `model_id` for FLUX.1-Depth-dev if reproducing
    the depth-conditioned benchmarks in the paper.
    """

    def __init__(
        self,
        inference_steps: int = 100,
        height: int = 512,
        width: int = 512,
        device: str = "cuda",
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.inference_steps = inference_steps
        self.height, self.width = height, width
        self.device = device

        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
        self.vae_processor = VaeImageProcessor()
        self.num_channels_latents = self.pipe.transformer.config.in_channels // 4

    @torch.no_grad()
    def init_prompt(self, text_prompt: str, init_inference_steps: int = 2, seed: int = 42):
        """Encode a prompt and prepare timesteps/latent IDs.

        We run a dummy generation call to populate `prompt_embeds`,
        `pooled_prompt_embeds` and `text_ids` on the pipeline (see
        `pipeline_flux.FluxPipeline`), then drop the text encoders to free
        GPU memory -- the optimization only needs the transformer and VAE.
        """
        self.pipe(
            text_prompt,
            guidance_scale=3.5,
            height=self.height,
            width=self.width,
            output_type="pil",
            num_inference_steps=init_inference_steps,
            generator=torch.Generator(self.device).manual_seed(seed),
        ).images[0]

        _, self.latent_image_ids = self.pipe.prepare_latents(
            1,
            self.num_channels_latents,
            self.height,
            self.width,
            self.pipe.prompt_embeds.dtype,
            self.pipe.device,
            torch.Generator(self.device).manual_seed(seed),
            None,
        )

        image_seq_len = 16
        mu = calculate_shift(
            image_seq_len,
            self.pipe.scheduler.config.base_image_seq_len,
            self.pipe.scheduler.config.max_image_seq_len,
            self.pipe.scheduler.config.base_shift,
            self.pipe.scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            scheduler=self.pipe.scheduler,
            num_inference_steps=self.inference_steps,
            device=self.device,
            timesteps=None,
            sigmas=np.linspace(1.0, 1 / self.inference_steps, self.inference_steps),
            mu=mu,
        )
        self.timesteps = timesteps
        self.pipe._num_timesteps = len(timesteps)

        del self.pipe.text_encoder
        del self.pipe.text_encoder_2
        torch.cuda.empty_cache()

        for p in self.pipe.transformer.parameters():
            p.requires_grad = False

    def img_to_latents(self, img: Image.Image) -> torch.Tensor:
        img = self.vae_processor.preprocess(img, height=self.height, width=self.width).to(torch.bfloat16).to(self.device)
        latents = self.pipe.vae.encode(img)[0].sample()
        latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        latents = self.pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
        return latents

    def latents_to_img(self, latents: torch.Tensor) -> Image.Image:
        latents = self.pipe._unpack_latents(latents, self.height, self.width, self.pipe.vae_scale_factor)
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0].detach()
        return self.pipe.image_processor.postprocess(image, output_type="pil")[0]

    def predict_noise(self, hidden_states, timestep, guidance=None):
        return self.pipe.transformer(
            hidden_states=hidden_states,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=self.pipe.pooled_prompt_embeds,
            encoder_hidden_states=self.pipe.prompt_embeds,
            txt_ids=self.pipe.text_ids,
            img_ids=self.latent_image_ids,
            joint_attention_kwargs=self.pipe.joint_attention_kwargs,
            return_dict=False,
        )[0]

    def predict_latents(self, hidden_states, timestep, sigma, guidance=None):
        noise_pred = self.predict_noise(hidden_states, timestep, guidance)
        latents_predict = hidden_states + (self.pipe.scheduler.sigmas[-1] - sigma) * noise_pred
        return latents_predict, noise_pred

    # ---------------------------------------------------------------------
    # Flow-Optimizer: direct target fitting at a single timestep index.
    # See Section 3 "Flow-Optimizer (direct target fitting)".
    # ---------------------------------------------------------------------
    def init_model(
        self,
        src_img,
        tgt_img,
        sampling_count: int = 10,
        timestep_idx: int = 55,
        guidance=None,
        noise_lr: float = 0.01,
        pred_lr: float = 0.04,
    ):
        """Initialize `noise_optim` (= u) and `pred_optim` (= z+Delta) from source.

        A short iterative solver averages several forward passes to produce a
        stable starting point around the source latent; this is only used for
        initialization, not during optimization.
        """

        def iterative_solver(init_states, target_states, timestep, sigma, sampling_count=sampling_count):
            states_list = []
            states = init_states
            for _ in range(sampling_count):
                states_list.append(states)
                noise_pred = self.predict_noise(states, timestep, guidance)
                states = target_states - (self.pipe.scheduler.sigmas[-1] - sigma) * noise_pred
            return torch.stack(states_list).mean(dim=0)

        with torch.no_grad():
            src_latents = self.img_to_latents(src_img)
            tgt_latents = self.img_to_latents(tgt_img)
            timestep = self.timesteps[timestep_idx].to(src_latents.dtype)
            self.timestep = timestep.repeat(src_latents.shape[0])
            self.sigma = self.pipe.scheduler.sigmas[timestep_idx]

            src_latents_mean = iterative_solver(src_latents, src_latents, self.timestep, self.sigma, sampling_count)
            src_predict_mean, src_noise_pred_mean = self.predict_latents(src_latents_mean, self.timestep, self.sigma, guidance)

            self.target_latents = tgt_latents.detach()

        self.noise_optim = nn.Parameter(torch.clone(src_noise_pred_mean))
        self.pred_optim = nn.Parameter(torch.clone(src_predict_mean))
        self.guidance = guidance

        for p in self.pipe.transformer.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.noise_optim, "lr": noise_lr},
                {"params": self.pred_optim, "lr": pred_lr},
            ]
        )

    def step(self):
        states_pred = self.pred_optim - ((self.pipe.scheduler.sigmas[-1] - self.sigma) * self.noise_optim)
        latents_pred, _ = self.predict_latents(states_pred, self.timestep, self.sigma, self.guidance)

        loss = torch.norm(latents_pred - self.target_latents)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def flux_optim(
        self,
        src_img,
        tgt_img,
        optim_iters: int = 100,
        sampling_count: int = 10,
        timestep_idx: int = 55,
        guidance=None,
        noise_lr: float = 0.01,
        pred_lr: float = 0.04,
    ):
        """Full Flow-Optimizer loop; returns a list of decoded PIL frames."""
        self.init_model(src_img, tgt_img, sampling_count, timestep_idx, guidance, noise_lr, pred_lr)
        imgs = []
        for i in range(optim_iters):
            self.step()
            imgs.append(self.latents_to_img(self.pred_optim.detach()))
        return imgs
