import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from hflayers import HopfieldPooling

from models.frozen_vit import FrozenViTClassifier
from models.prompt_pooling import LoRAOutputWeight, SpatialGELUPooling


logger = logging.getLogger()

TRAINABLE_PARTS = ("query", "q", "k", "v", "o")
INPUT_MODES = ("local", "full_vit")
OUTPUT_MODES = ("full", "lora", "lora_merge_1024", "lora_merge_batch")






class HopfieldPoolingPrompts(nn.Module):
    """Hopfield pooling layers that write prompts into ViT blocks.

    Each layer pools ``prompt_length`` tokens, which are inserted after CLS for
    one block and removed afterwards. ``trainable`` selects the parts that
    receive gradients: ``query`` (learned pooling weights), ``q``/``k``/``v``
    (slices of the packed input projection) and ``o`` (output projection).
    V is a static identity map unless ``v`` is trainable, in which case a
    projection initialised to identity replaces it.

    ``input_mode='local'`` pools from the tokens entering each prompted block.
    ``input_mode='full_vit'`` pools every layer from one detached, normalised
    clean ViT token sequence.
    Explicit ``prompt_blocks`` uses one module: generate at ``start_layer``
    once per forward, then reuse that tensor at every selected insertion block.
    """

    def __init__(self,
                 embed_dim: int,
                 num_pooling_heads: int = 1,
                 num_pooling_blocks: int = 3,
                 prompt_length: int = 10,
                 deep_prompts: bool = False,
                 start_layer: Optional[int] = None,
                 prompt_blocks: Optional[Iterable[int]] = None,
                 replace_cls: bool = False,
                 input_mode: str = "local",
                 pooling_mode: str = "attention",
                 trainable: Iterable[str] = ("q",),
                 o_mode: str = "full",
                 lora_rank: int = 8,
                 lora_alpha: float = 8.0):
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
        if o_mode not in OUTPUT_MODES:
            raise ValueError(f"o_mode must be one of {OUTPUT_MODES}, got {o_mode!r}")
        if o_mode != "full" and set(trainable) != {"o"}:
            raise ValueError("HFPool LoRA modes require --hopfield_trainable o only")
        if lora_rank <= 0 or lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")

        self.embed_dim = int(embed_dim)
        self.num_pooling_heads = int(num_pooling_heads)
        self.num_pooling_blocks = int(num_pooling_blocks)
        self.prompt_length = int(prompt_length)
        self.deep_prompts = bool(deep_prompts)
        self.start_layer = start_layer
        self.prompt_blocks = None if prompt_blocks is None else tuple(int(i) for i in prompt_blocks)
        self.replace_cls = bool(replace_cls)
        if self.replace_cls and (self.num_pooling_blocks != 1 or self.prompt_length != 1
                                 or self.prompt_blocks is not None):
            raise ValueError("CLS replacement requires one HFPool block, one prompt token, and no explicit prompt blocks")
        if self.prompt_blocks is not None:
            if num_pooling_blocks != 1:
                raise ValueError("Explicit hopfield_prompt_blocks requires num_pooling_blocks=1")
            if not self.prompt_blocks or tuple(sorted(set(self.prompt_blocks))) != self.prompt_blocks:
                raise ValueError("hopfield_prompt_blocks must be non-empty and strictly increasing")
        self.input_mode = input_mode
        self.trainable = frozenset(trainable)
        self.o_mode = o_mode
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.pooling_mode = pooling_mode
        if pooling_mode not in {"attention", "spatial_gelu"}:
            raise ValueError("Unknown pooling mode")
        if pooling_mode == "spatial_gelu" and (self.trainable != {"o"} or prompt_length != 10
                                                or num_pooling_heads != 1 or replace_cls):
            raise ValueError("Spatial GELU requires 10 tokens, one head, trainable o, and ordinary prompts")

        self.hopfield_pooling_layers = nn.ModuleList([
            SpatialGELUPooling(self.embed_dim) if pooling_mode == "spatial_gelu" else HopfieldPooling(
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
        if self.o_mode != "full":
            self._configure_lora_output_weights()

        logger.info(
            "HFPool | trainable=%s | input_mode=%s | start_layer=%s | "
            "deep_prompts=%s | blocks=%d | heads=%d | prompt_length=%d | o_mode=%s | insertion_blocks=%s | replace_cls=%s | pooling_mode=%s",
            sorted(self.trainable), self.input_mode, self.start_layer,
            self.deep_prompts, self.num_pooling_blocks,
            self.num_pooling_heads, self.prompt_length, self.o_mode, self.prompt_blocks,
            self.replace_cls, self.pooling_mode,
        )

    def output_projections(self):
        return [layer.out_proj if self.pooling_mode == "spatial_gelu" else
                layer.hopfield.association_core.out_proj for layer in self.hopfield_pooling_layers]

    def _configure_lora_output_weights(self) -> None:
        for layer_index, out_proj in enumerate(self.output_projections()):
            if out_proj is None or out_proj.weight.ndim != 2:
                raise ValueError("HFPool LoRA requires a two-dimensional out_proj.weight")
            out_features, in_features = out_proj.weight.shape
            parametrization = LoRAOutputWeight(
                out_features, in_features, self.lora_rank,
                self.lora_alpha, layer_index,
            )
            parametrize.register_parametrization(out_proj, "weight", parametrization)
            out_proj.parametrizations.weight.original.requires_grad_(False)
            if out_proj.bias is not None:
                out_proj.bias.requires_grad_(False)

    def lora_output_weights(self):
        if self.o_mode == "full":
            return []
        return [
            projection.parametrizations.weight[0]
            for projection in self.output_projections()
        ]

    @torch.no_grad()
    def merge_lora_output_weights(self):
        results = []
        for projection, lora in zip(self.output_projections(), self.lora_output_weights()):
            base = projection.parametrizations.weight.original
            results.append(lora.merge_into(base))
        return results

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

        if self.pooling_mode == "spatial_gelu":
            for projection in self.output_projections():
                for parameter in projection.parameters():
                    parameter.requires_grad = True
            return

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

    def resolve_prompt_blocks(self, depth: int) -> Tuple[int, ...]:
        first, last = self._resolve_prompt_range(depth)
        if self.prompt_blocks is None:
            return tuple(range(first, last))
        if self.prompt_blocks[0] < first or self.prompt_blocks[-1] >= depth:
            raise ValueError(
                "hopfield_prompt_blocks must be inside the backbone and at or after "
                f"the generation block: blocks={self.prompt_blocks}, source={first}, depth={depth}"
            )
        return self.prompt_blocks

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

        Returns the clean tokens entering the generation block and the
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
            return_prompts: bool = False,
    ) -> torch.Tensor:
        if backbone.cls_token is None:
            raise ValueError("HFPool requires a CLS-token backbone")
        if backbone.embed_dim != self.embed_dim:
            raise ValueError(
                "Hopfield/backbone embedding dimensions do not match: "
                f"{self.embed_dim} != {backbone.embed_dim}"
            )

        depth = len(backbone.blocks)
        first, _ = self._resolve_prompt_range(depth)
        prompt_blocks = self.resolve_prompt_blocks(depth)
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
        captured_prompts = []
        if self.replace_cls:
            source = x if pooling_source is None else pooling_source
            replacement = self.hopfield_pooling_layers[0](source).reshape(
                batch_size, 1, self.embed_dim
            )
            if return_prompts:
                captured_prompts.append(replacement.detach())
            x = torch.cat((replacement, x[:, 1:]), dim=1)
            for block_idx in range(first, depth):
                x = backbone.blocks[block_idx](x)
            features = backbone.norm(x)[:, 0]
            if return_prompts:
                return features, torch.stack(captured_prompts, dim=1)
            return features
        shared_prompt = None
        if self.prompt_blocks is not None:
            source = x if pooling_source is None else pooling_source
            shared_prompt = self.hopfield_pooling_layers[0](source).reshape(
                batch_size, self.prompt_length, self.embed_dim
            )
        slot_by_block = {block: slot for slot, block in enumerate(prompt_blocks)}
        for block_idx in range(first, depth):
            if block_idx in slot_by_block:
                if shared_prompt is None:
                    source = x if pooling_source is None else pooling_source
                    prompt = self.hopfield_pooling_layers[slot_by_block[block_idx]](source).reshape(
                        batch_size, self.prompt_length, self.embed_dim
                    )
                else:
                    prompt = shared_prompt
                if return_prompts:
                    captured_prompts.append(prompt.detach())
                x = torch.cat((x[:, :1], prompt, x[:, 1:]), dim=1)
            x = backbone.blocks[block_idx](x)
            if block_idx in slot_by_block:
                x = torch.cat((x[:, :1], x[:, self.prompt_length + 1:]), dim=1)
        features = backbone.norm(x)[:, 0]
        if return_prompts:
            return features, torch.stack(captured_prompts, dim=1)
        return features


class HFPool(FrozenViTClassifier):
    """Frozen ViT + FC with shared Hopfield-pooling prompts (no routing)."""

    def build_adapter(self,
                      num_pooling_heads: int = 1,
                      num_pooling_blocks: int = 3,
                      prompt_length: int = 10,
                      deep_prompts: int = 0,
                      hopfield_start_layer: Optional[int] = None,
                      hopfield_prompt_blocks: Optional[Iterable[int]] = None,
                      hopfield_replace_cls: bool = False,
                      hopfield_input_mode: str = "local",
                      hopfield_trainable: Iterable[str] = ("q",),
                      hopfield_o_mode: str = "full",
                      hopfield_lora_rank: int = 8,
                      hopfield_lora_alpha: float = 8.0,
                      **kwargs) -> nn.Module:
        return HopfieldPoolingPrompts(
            embed_dim=self.embed_dim,
            num_pooling_heads=num_pooling_heads,
            num_pooling_blocks=num_pooling_blocks,
            prompt_length=prompt_length,
            deep_prompts=deep_prompts,
            start_layer=hopfield_start_layer,
            prompt_blocks=hopfield_prompt_blocks,
            replace_cls=hopfield_replace_cls,
            input_mode=hopfield_input_mode,
            pooling_mode=kwargs.get("hopfield_pooling_mode", "attention"),
            trainable=hopfield_trainable,
            o_mode=hopfield_o_mode,
            lora_rank=hopfield_lora_rank,
            lora_alpha=hopfield_lora_alpha,
        )

    @property
    def uses_full_vit_input(self) -> bool:
        return self.adapter.input_mode == "full_vit"

    @torch.no_grad()
    def prepare_hopfield_context(
            self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.adapter.prepare_full_vit_context(self.backbone, inputs)

    @torch.no_grad()
    def capture_prompts(self, inputs: torch.Tensor):
        blocks = self.adapter.resolve_prompt_blocks(len(self.backbone.blocks))
        _, prompts = self.adapter(self.backbone, inputs, return_prompts=True)
        return blocks, prompts
