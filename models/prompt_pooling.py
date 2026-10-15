"""Shared spatial pooling and FP32 low-rank output weights for prompt methods."""
import math

import torch
import torch.nn as nn


class SpatialGELUPooling(nn.Module):
    """Keep CLS, average the patch grid to 3x3, then GELU and W_o.

    This side branch compresses features only; original ViT tokens survive.
    No learned query, Q/K projection or attention is instantiated.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Match hflayers' output-projection initialization.
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, tokens):
        side = math.isqrt(tokens.shape[1] - 1)
        if side * side != tokens.shape[1] - 1 or side < 3:
            raise ValueError("Spatial pooling requires CLS followed by a square patch grid")
        patches = tokens[:, 1:].transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
        pooled = torch.nn.functional.adaptive_avg_pool2d(patches, (3, 3)).flatten(2).transpose(1, 2)
        compressed = torch.cat((tokens[:, :1], pooled), dim=1)
        return self.out_proj(torch.nn.functional.gelu(compressed))


class LoRAOutputWeight(nn.Module):
    """Expose W + (alpha / rank) BA through hflayers' out_proj.weight access."""

    def __init__(self, out_features: int, in_features: int, rank: int,
                 alpha: float, layer_index: int):
        super().__init__()
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        # A separate deterministic stream keeps image augmentation RNG aligned
        # with the full-W_o control and across merge schedules.
        seed = (torch.initial_seed() + 1009 * (layer_index + 1)) % (2 ** 63 - 1)
        self.register_buffer("reset_seed", torch.tensor(seed, dtype=torch.long))
        self.register_buffer("reset_count", torch.zeros((), dtype=torch.long))
        self.reset()

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        # Keep the accumulated weight and low-rank update FP32 under AMP.
        if any(tensor.dtype != torch.float32 for tensor in
               (base_weight, self.lora_A, self.lora_B)):
            raise ValueError("HFPool output LoRA weights must stay FP32")
        with torch.autocast(device_type=base_weight.device.type, enabled=False):
            return base_weight + self.scale * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def reset(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.reset_seed) + int(self.reset_count))
        fresh_a = torch.empty(self.lora_A.shape, dtype=torch.float32)
        nn.init.kaiming_uniform_(fresh_a, a=5 ** 0.5, generator=generator)
        self.lora_A.copy_(fresh_a)
        self.lora_B.zero_()
        self.reset_count.add_(1)

    @torch.no_grad()
    def merge_into(self, base_weight: torch.Tensor) -> tuple:
        if any(tensor.dtype != torch.float32 for tensor in
               (base_weight, self.lora_A, self.lora_B)):
            raise ValueError("HFPool output LoRA weights must stay FP32")
        with torch.autocast(device_type=base_weight.device.type, enabled=False):
            delta = self.scale * (self.lora_B @ self.lora_A)
            delta_norm = torch.linalg.vector_norm(delta).item()
            base_norm = torch.linalg.vector_norm(base_weight).item()
            base_weight.add_(delta)
        self.reset()
        return delta_norm, base_norm
