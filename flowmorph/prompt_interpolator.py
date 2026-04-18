"""Prompt interpolation helpers used by Flow-Interpolation.

The decoupled geometry/semantics mixing of FlowMorph does not require prompt
interpolation to work, but optionally plugging in interpolated prompts at
each alpha gives the backbone a smoother text-conditioned trajectory. Both
functions are deterministic and do not depend on external APIs.
"""

from __future__ import annotations

from typing import List


def generate_interpolated_prompts(source_prompt: str, target_prompt: str, num_steps: int) -> List[str]:
    """Produce descriptive in-between prompts with 5 piecewise regions."""
    prompts = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1) if num_steps > 1 else 0
        if alpha <= 0:
            prompts.append(source_prompt)
        elif alpha >= 1:
            prompts.append(target_prompt)
        elif alpha < 0.2:
            prompts.append(f"{source_prompt}, with subtle hints of transformation")
        elif alpha < 0.4:
            prompts.append(f"{source_prompt} beginning to transform, showing early signs of {target_prompt}")
        elif alpha < 0.6:
            prompts.append(f"halfway between {source_prompt} and {target_prompt}, blended features")
        elif alpha < 0.8:
            prompts.append(f"{target_prompt} emerging, with fading elements of {source_prompt}")
        else:
            prompts.append(f"{target_prompt}, nearly complete transformation")
    return prompts


def generate_creative_prompts(source_prompt: str, target_prompt: str, num_steps: int) -> List[str]:
    """A terser version that embeds the percentage of the transition into the
    prompt and picks a transition verb."""
    prompts: List[str] = []
    src_key = source_prompt.split(",")[0] if "," in source_prompt else source_prompt
    tgt_key = target_prompt.split(",")[0] if "," in target_prompt else target_prompt
    verbs = [
        "morphing into", "transforming to", "evolving into",
        "shifting towards", "transitioning to", "becoming",
    ]

    for i in range(num_steps):
        alpha = i / (num_steps - 1) if num_steps > 1 else 0
        if alpha <= 0:
            prompts.append(source_prompt)
            continue
        if alpha >= 1:
            prompts.append(target_prompt)
            continue
        verb = verbs[min(int(alpha * len(verbs)), len(verbs) - 1)]
        pct = int(alpha * 100)
        if alpha < 0.5:
            prompts.append(f"{src_key} {verb} {tgt_key} ({pct}% transformed), high quality")
        else:
            prompts.append(f"{tgt_key} emerged from {src_key} ({pct}% complete), high quality")
    return prompts
