import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from hflayers import HopfieldPooling

from models.frozen_vit import FrozenViTClassifier


logger = logging.getLogger()

TRAINABLE_PARTS = ("query", "q", "k", "v", "o")
INPUT_MODES = ("local", "full_vit")


class HopfieldPoolingPrompts(nn.Module):
    """Hopfield pooling layers that write prompts into consecutive ViT blocks.

    Each layer pools ``prompt_length`` tokens, which are inserted after CLS for
    one block and removed afterwards. ``trainable`` selects the parts that
    receive gradients: ``query`` (learned pooling weights), ``q``/``k``/``v``
    (slices of the packed input projection) and ``o`` (output projection).
    V is a static identity map unless ``v`` is trainable, in which case a
    projection initialised to identity replaces it.

    ``input_mode='local'`` pools from the tokens entering each prompted block.
    ``input_mode='full_vit'`` pools every layer from one detached, normalised
    clean ViT token sequence.
    """

    def __init__(self,
                 embed_dim: int,
                 num_pooling_heads: int = 1,
                 num_pooling_blocks: int = 3,
                 prompt_length: int = 10,
                 deep_prompts: bool = False,
                 start_layer: Optional[int] = None,
                 input_mode: str = "local",
                 trainable: Iterable[str] = ("q",)):
        super().__init__()

        trainable = tuple(dict.fromkeys(trainable))
        unknown = sorted(set(trainable) - set(TRAINABLE_PARTS))
        if not trainable or unknown:
            raise ValueError(
                f"trainable must be a non-empty subset of {TRAINABLE_PARTS}, "
                f"got {trainable}"
            )
        if embed_dim <= 0 or num_pooling_heads <= 0 or num_pooling_blocks <= 0:
            raise ValueError(
                "embed_dim, num_pooling_heads and num_pooling_blocks must be "
                f"positive, got {embed_dim}, {num_pooling_heads}, "
                f"{num_pooling_blocks}"
            )
        if prompt_length <= 0:
            raise ValueError(f"prompt_length must be positive, got {prompt_length}")
        if start_layer is not None and start_layer < 0:
            raise ValueError(
                f"start_layer must be non-negative when set, got {start_layer}"
            )
        if input_mode not in INPUT_MODES:
            raise ValueError(
                f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}"
            )

        self.embed_dim = int(embed_dim)
        self.num_pooling_heads = int(num_pooling_heads)
        self.num_pooling_blocks = int(num_pooling_blocks)
        self.prompt_length = int(prompt_length)
        self.deep_prompts = bool(deep_prompts)
        self.start_layer = start_layer
        self.input_mode = input_mode
        self.trainable = frozenset(trainable)

        self.hopfield_pooling_layers = nn.ModuleList([
            HopfieldPooling(
                input_size=self.embed_dim,
                hidden_size=self.embed_dim,
                update_steps_max=0,
                quantity=self.prompt_length,
                num_heads=self.num_pooling_heads,
                stored_pattern_as_static=False,
                pattern_projection_as_static="v" not in self.trainable,
                state_pattern_as_static=False,
            )
            for _ in range(self.num_pooling_blocks)
        ])
        if "v" in self.trainable:
            self._init_value_projection_to_identity()
        self._configure_trainable_parameters()

        logger.info(
            "HFPool | trainable=%s | input_mode=%s | start_layer=%s | "
            "deep_prompts=%s | blocks=%d | heads=%d | prompt_length=%d",
            sorted(self.trainable), self.input_mode, self.start_layer,
            self.deep_prompts, self.num_pooling_blocks,
            self.num_pooling_heads, self.prompt_length,
        )

    @staticmethod
    def _projection_slices(association_core) -> dict:
        """Return row slices in hflayers' packed Q/K/V projection order."""
        association_dim = association_core.num_heads * association_core.head_dim
        pattern_dim = association_core.num_heads * association_core.pattern_dim
        offset = 0
        slices = {}
        for name, is_static, width in (
                ("q", association_core.query_as_static, association_dim),
                ("k", association_core.key_as_static, association_dim),
                ("v", association_core.value_as_static, pattern_dim)):
            if not is_static:
                slices[name] = slice(offset, offset + width)
                offset += width
        return slices

    @torch.no_grad()
    def _init_value_projection_to_identity(self) -> None:
        for layer in self.hopfield_pooling_layers:
            association_core = layer.hopfield.association_core
            value_slice = self._projection_slices(association_core)["v"]
            weight = association_core.in_proj_weight[value_slice]
            if weight.shape[0] != weight.shape[1]:
                raise ValueError(
                    "identity V initialisation needs a square projection, got "
                    f"{tuple(weight.shape)}"
                )
            weight.copy_(torch.eye(weight.shape[0]))
            if association_core.in_proj_bias is not None:
                association_core.in_proj_bias[value_slice].zero_()

    def _configure_trainable_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

        projections = tuple(name for name in ("q", "k", "v") if name in self.trainable)
        for layer in self.hopfield_pooling_layers:
            association_core = layer.hopfield.association_core
            if "query" in self.trainable:
                layer.pooling_weights.requires_grad = True
            if "o" in self.trainable:
                if association_core.out_proj is None:
                    raise ValueError("training 'o' requires a Hopfield out_proj")
                for param in association_core.out_proj.parameters():
                    param.requires_grad = True
            if projections:
                self._enable_projection_slices(association_core, projections)

    def _enable_projection_slices(self, association_core, names: tuple) -> None:
        if association_core.in_proj_weight is None:
            raise ValueError("HFPool expects hflayers' packed in_proj_weight")
        projection_slices = self._projection_slices(association_core)
        active_slices = tuple(projection_slices[name] for name in names)
        self._enable_masked_parameter(association_core.in_proj_weight, active_slices)
        if association_core.in_proj_bias is not None:
            self._enable_masked_parameter(association_core.in_proj_bias, active_slices)

    @staticmethod
    def _enable_masked_parameter(
            parameter: nn.Parameter, active_slices: tuple) -> None:
        parameter.requires_grad = True
        # Q/K/V share one packed Parameter. Optimizer weight decay would modify
        # masked rows after this hook, so optimizers place this parameter in a
        # zero-weight-decay group.
        parameter._hopfield_slice_masked = True

        def mask_gradient(gradient: torch.Tensor) -> torch.Tensor:
            masked = torch.zeros_like(gradient)
            for active_slice in active_slices:
                masked[active_slice] = gradient[active_slice]
            return masked

        parameter.register_hook(mask_gradient)

    def _resolve_prompt_range(self, depth: int) -> Tuple[int, int]:
        if self.start_layer is None:
            first = depth - self.num_pooling_blocks if self.deep_prompts else 0
        else:
            first = self.start_layer
        last = first + self.num_pooling_blocks
        if first < 0 or last > depth:
            raise ValueError(
                "Hopfield prompt blocks must stay inside the backbone: "
                f"start={first}, count={self.num_pooling_blocks}, depth={depth}"
            )
        return first, last

    @staticmethod
    def _embed_inputs(backbone: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        x = backbone.patch_embed(inputs)
        cls_token = backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        return backbone.pos_drop(x + backbone.pos_embed)

    @torch.no_grad()
    def prepare_full_vit_context(
            self,
            backbone: nn.Module,
            inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one clean ViT pass for ``input_mode='full_vit'``.

        Returns the clean tokens entering the first prompted block and the
        normalised final token sequence every pooling layer reads from.
        """
        first, _ = self._resolve_prompt_range(len(backbone.blocks))
        x = self._embed_inputs(backbone, inputs)
        insertion_tokens = x
        for block_idx, block in enumerate(backbone.blocks):
            x = block(x)
            if block_idx + 1 == first:
                insertion_tokens = x
        return insertion_tokens.detach(), backbone.norm(x).detach()

    def forward(
            self,
            backbone: nn.Module,
            inputs: torch.Tensor,
            hopfield_context: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if backbone.cls_token is None:
            raise ValueError("HFPool requires a CLS-token backbone")
        if backbone.embed_dim != self.embed_dim:
            raise ValueError(
                "Hopfield/backbone embedding dimensions do not match: "
                f"{self.embed_dim} != {backbone.embed_dim}"
            )

        depth = len(backbone.blocks)
        first, last = self._resolve_prompt_range(depth)
        if self.input_mode == "full_vit":
            if hopfield_context is None:
                hopfield_context = self.prepare_full_vit_context(backbone, inputs)
            x, pooling_source = hopfield_context
            if x.shape[0] != inputs.shape[0]:
                raise ValueError(
                    "hopfield_context batch size does not match inputs: "
                    f"{x.shape[0]} != {inputs.shape[0]}"
                )
        else:
            if hopfield_context is not None:
                raise ValueError("hopfield_context is only valid with input_mode='full_vit'")
            x = self._embed_inputs(backbone, inputs)
            for block_idx in range(first):
                x = backbone.blocks[block_idx](x)
            pooling_source = None

        batch_size = x.shape[0]
        for layer_idx, block_idx in enumerate(range(first, last)):
            source = x if pooling_source is None else pooling_source
            prompt = self.hopfield_pooling_layers[layer_idx](source).reshape(
                batch_size, self.prompt_length, self.embed_dim
            )
            x = torch.cat((x[:, :1], prompt, x[:, 1:]), dim=1)
            x = backbone.blocks[block_idx](x)
            x = torch.cat((x[:, :1], x[:, self.prompt_length + 1:]), dim=1)

        for block_idx in range(last, depth):
            x = backbone.blocks[block_idx](x)
        return backbone.norm(x)[:, 0]


class HFPool(FrozenViTClassifier):
    """Frozen ViT + FC with shared Hopfield-pooling prompts (no routing)."""

    def build_adapter(self,
                      num_pooling_heads: int = 1,
                      num_pooling_blocks: int = 3,
                      prompt_length: int = 10,
                      deep_prompts: int = 0,
                      hopfield_start_layer: Optional[int] = None,
                      hopfield_input_mode: str = "local",
                      hopfield_trainable: Iterable[str] = ("q",),
                      **kwargs) -> nn.Module:
        return HopfieldPoolingPrompts(
            embed_dim=self.embed_dim,
            num_pooling_heads=num_pooling_heads,
            num_pooling_blocks=num_pooling_blocks,
            prompt_length=prompt_length,
            deep_prompts=deep_prompts,
            start_layer=hopfield_start_layer,
            input_mode=hopfield_input_mode,
            trainable=hopfield_trainable,
        )

    @property
    def uses_full_vit_input(self) -> bool:
        return self.adapter.input_mode == "full_vit"

    @torch.no_grad()
    def prepare_hopfield_context(
            self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.adapter.prepare_full_vit_context(self.backbone, inputs)
