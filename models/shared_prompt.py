from typing import Iterable

import torch
import torch.nn as nn

from models.flyprompt import Prompt
from models.frozen_vit import FrozenViTClassifier


class SharedPrompt(FrozenViTClassifier):
    """FlyPrompt prompt tokens with one expert shared by the whole stream.

    There is no REAR router: every sample uses expert 0. With ``use_ema`` the
    EMA bank is one global bank that keeps updating across internal steps.
    """

    def build_adapter(self,
                      len_prompt: int = 20,
                      pos_prompt: Iterable[int] = (0, 1, 2, 3, 4),
                      **kwargs) -> nn.Module:
        return Prompt(
            num_experts=1,
            len_prompt=len_prompt,
            embed_dim=self.embed_dim,
            pos_prompt=pos_prompt,
        )

    def extract_features(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        expert_ids = torch.zeros(
            inputs.size(0), device=inputs.device, dtype=torch.long
        )
        return self.adapter(self.backbone, inputs, expert_ids)
