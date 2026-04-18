"""FlowMorph: a training-free framework for geometry-preserving, semantics-aware
image morphing on top of frozen rectified flow models.

See the paper "FlowMorph: Revealing an Optimizable Flow Latent Space for
Controlled Image Morphing" (WACV 2026).
"""

from .flux_optim import FluxOptimizer
from .utils import make_image_grid, slerp, slerp_direction

__all__ = ["FluxOptimizer", "make_image_grid", "slerp", "slerp_direction"]
__version__ = "1.0.0"
