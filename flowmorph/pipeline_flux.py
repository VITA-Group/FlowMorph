"""Thin wrapper around diffusers' FluxPipeline.

We only extend `encode_prompt` so that prompt embeddings, pooled embeddings
and text IDs are cached on the pipeline instance. This is the single hook
FlowMorph needs to re-use prompt embeddings across the optimization and the
deterministic reverse chain without re-running the text encoders.
"""

import diffusers.pipelines.flux.pipeline_flux as pipeline_flux


class FluxPipeline(pipeline_flux.FluxPipeline):
    def encode_prompt(self, *args, **kwargs):
        out = super().encode_prompt(*args, **kwargs)
        (
            self.prompt_embeds,
            self.pooled_prompt_embeds,
            self.text_ids,
        ) = out
        return out
