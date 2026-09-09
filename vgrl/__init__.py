"""View-grounded RL for driving VLMs.

Adds a perception term to GRPO that contrasts blanking the camera view carrying
the answer's evidence against blanking an irrelevant view, so the policy is
pushed to actually read the right view rather than to guess from language priors.
"""

from .config import ViewGroundingConfig
from .losses import (
    ViewGroundingOutput,
    masked_sequence_mean,
    per_token_kl_k1,
    per_token_kl_k3,
    pick_mask_views,
    view_grounding_loss,
)
from .masking import ViewMasker
from .views import (
    CAM_TO_INDEX,
    CAMERAS,
    NUM_VIEWS,
    complement_views,
    evidence_views,
    parse_views,
)

__all__ = [
    "CAMERAS",
    "CAM_TO_INDEX",
    "NUM_VIEWS",
    "ViewGroundingConfig",
    "ViewGroundingOutput",
    "ViewMasker",
    "complement_views",
    "evidence_views",
    "masked_sequence_mean",
    "parse_views",
    "per_token_kl_k1",
    "per_token_kl_k3",
    "pick_mask_views",
    "view_grounding_loss",
]


def get_trainer_cls():
    """Import the trainer lazily — it pulls in trl/transformers."""
    from .grpo_view_trainer import ViewGroundedGRPOTrainer

    return ViewGroundedGRPOTrainer
