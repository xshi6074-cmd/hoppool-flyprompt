from typing import Iterable

import torch
import torch.nn as nn

from models.frozen_vit import FrozenViTClassifier


class AttentionResidualGating(nn.Module):
    """Gate each frozen ViT attention head at selected blocks."""

    def __init__(self, gate_blocks: Iterable[int], num_heads: int):
        super().__init__()
        gate_blocks = tuple(int(block) for block in gate_blocks)
        if not gate_blocks:
            raise ValueError("gate_blocks must not be empty")
        if tuple(sorted(set(gate_blocks))) != gate_blocks:
            raise ValueError(
                "gate_blocks must be sorted and unique, got "
                f"{gate_blocks}"
            )
        if gate_blocks[0] < 0:
            raise ValueError(
                f"gate_blocks must be non-negative, got {gate_blocks}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        self.gate_blocks = gate_blocks
        self.num_heads = int(num_heads)
        self.gate_logits = nn.Parameter(
            torch.zeros(len(gate_blocks), self.num_heads)
        )
        self._gate_slot_by_block = {
            block: slot for slot, block in enumerate(gate_blocks)
        }

    @staticmethod
    def _forward_gated_attention(
            attention: nn.Module,
            x: torch.Tensor,
            head_gates: torch.Tensor) -> torch.Tensor:
        required = (
            "num_heads", "scale", "qkv", "attn_drop", "proj", "proj_drop"
        )
        missing = [name for name in required if not hasattr(attention, name)]
        if missing:
            raise TypeError(
                "AttentionResidualGating requires the repository ViT attention; "
                f"missing attributes {missing}"
            )

        batch_size, token_count, embed_dim = x.shape
        num_heads = int(attention.num_heads)
        if head_gates.shape != (num_heads,):
            raise ValueError(
                "head gate shape does not match attention heads: "
                f"{tuple(head_gates.shape)} != {(num_heads,)}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension {embed_dim} is not divisible by {num_heads}"
            )
        head_dim = embed_dim // num_heads
        qkv = attention.qkv(x).reshape(
            batch_size, token_count, 3, num_heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        weights = (query @ key.transpose(-2, -1)) * attention.scale
        weights = attention.attn_drop(weights.softmax(dim=-1))
        head_output = weights @ value
        head_output = head_output * head_gates.view(1, num_heads, 1, 1)
        output = head_output.transpose(1, 2).reshape(
            batch_size, token_count, embed_dim
        )
        output = attention.proj(output)
        return attention.proj_drop(output)

    @staticmethod
    def _forward_gated_block(
            block: nn.Module,
            x: torch.Tensor,
            head_gates: torch.Tensor) -> torch.Tensor:
        required = (
            "norm1", "attn", "ls1", "drop_path1",
            "norm2", "mlp", "ls2", "drop_path2",
        )
        missing = [name for name in required if not hasattr(block, name)]
        if missing:
            raise TypeError(
                "AttentionResidualGating requires a pre-norm ViT block; "
                f"missing attributes {missing}"
            )

        attention_output = AttentionResidualGating._forward_gated_attention(
            attention=block.attn,
            x=block.norm1(x),
            head_gates=head_gates,
        )
        x = x + block.drop_path1(block.ls1(attention_output))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def forward(
            self,
            backbone: nn.Module,
            inputs: torch.Tensor) -> torch.Tensor:
        depth = len(backbone.blocks)
        if self.gate_blocks[-1] >= depth:
            raise ValueError(
                "gate block cannot exceed backbone depth: "
                f"{self.gate_blocks[-1]} >= {depth}"
            )

        x = backbone.patch_embed(inputs)
        if backbone.cls_token is not None:
            cls_token = backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = backbone.pos_drop(x + backbone.pos_embed)

        gates = 1.0 + torch.tanh(self.gate_logits)
        for block_idx, block in enumerate(backbone.blocks):
            gate_slot = self._gate_slot_by_block.get(block_idx)
            if gate_slot is None:
                x = block(x)
            else:
                x = self._forward_gated_block(
                    block=block,
                    x=x,
                    head_gates=gates[gate_slot],
                )

        x = backbone.norm(x)
        return x[:, 0]


class Gate(FrozenViTClassifier):
    """Frozen ViT + FC with a learned per-head gain at ``gate_blocks``.

    The gain ``1 + tanh(alpha)`` starts at identity and scales each head's
    ``softmax(QK)V`` output before the attention output projection. No prompt
    tokens are written.
    """

    def build_adapter(self,
                      gate_blocks: Iterable[int] = (0, 1, 2, 3, 4),
                      **kwargs) -> nn.Module:
        return AttentionResidualGating(
            gate_blocks=gate_blocks,
            num_heads=self.backbone.blocks[0].attn.num_heads,
        )
