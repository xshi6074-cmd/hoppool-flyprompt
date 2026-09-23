import copy
import math
from typing import Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import create_backbone
from utils.distill_gradnorm import resolve_distill_weight_mode


class ClassFeatureStatistics(nn.Module):
    """Online class statistics with teacher-aligned replay snapshots.

    The exact per-dimension variance is always retained with parallel Welford
    updates. ``low_rank`` additionally keeps a truncated class-wise M2 sketch;
    sampling combines that correlated component with residual diagonal noise.
    """

    def __init__(
            self,
            num_classes: int,
            feature_dim: int,
            history_size: int,
            statistic_type: str = "diagonal",
            covariance_rank: int = 16,
            min_replay_samples: int = 2,
            variance_prior_strength: float = 16.0,
            variance_floor: float = 1e-5,
            variance_ceiling: Optional[float] = 10.0,
            sketch_shrinkage: str = "none"):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if history_size <= 0:
            raise ValueError(f"history_size must be positive, got {history_size}")
        if statistic_type not in {"diagonal", "low_rank"}:
            raise ValueError(
                "statistic_type must be diagonal or low_rank, got "
                f"{statistic_type}"
            )
        if covariance_rank <= 0 or covariance_rank > feature_dim:
            raise ValueError(
                "covariance_rank must be in [1, feature_dim], got "
                f"{covariance_rank}"
            )
        if min_replay_samples < 2:
            raise ValueError(
                "min_replay_samples must be at least 2 when variance is used, "
                f"got {min_replay_samples}"
            )
        if variance_prior_strength < 0:
            raise ValueError(
                "variance_prior_strength must be non-negative, got "
                f"{variance_prior_strength}"
            )
        if variance_floor <= 0:
            raise ValueError(
                f"variance_floor must be positive, got {variance_floor}"
            )
        if variance_ceiling is not None and variance_ceiling <= variance_floor:
            raise ValueError(
                "variance_ceiling must exceed variance_floor, got "
                f"{variance_ceiling} <= {variance_floor}"
            )

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.history_size = int(history_size)
        self.statistic_type = statistic_type
        self.covariance_rank = int(covariance_rank)
        if sketch_shrinkage not in {"none", "fd"}:
            raise ValueError(
                f"sketch_shrinkage must be none or fd, got {sketch_shrinkage}"
            )
        self.sketch_shrinkage = sketch_shrinkage
        self.min_replay_samples = int(min_replay_samples)
        self.variance_prior_strength = float(variance_prior_strength)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = variance_ceiling
        sketch_rank = self.covariance_rank if statistic_type == "low_rank" else 0

        self.register_buffer(
            "live_count", torch.zeros(self.num_classes, dtype=torch.long)
        )
        self.register_buffer(
            "live_mean", torch.zeros(self.num_classes, self.feature_dim)
        )
        self.register_buffer(
            "live_m2", torch.zeros(self.num_classes, self.feature_dim)
        )
        self.register_buffer(
            "live_sketch",
            torch.zeros(self.num_classes, sketch_rank, self.feature_dim),
        )

        history_shape = (self.history_size, self.num_classes)
        self.register_buffer(
            "replay_count", torch.zeros(history_shape, dtype=torch.long)
        )
        self.register_buffer(
            "replay_mean", torch.zeros(*history_shape, self.feature_dim)
        )
        self.register_buffer(
            "replay_m2", torch.zeros(*history_shape, self.feature_dim)
        )
        self.register_buffer(
            "replay_sketch",
            torch.zeros(*history_shape, sketch_rank, self.feature_dim),
        )
        self.register_buffer(
            "replay_eligible", torch.zeros(history_shape, dtype=torch.bool)
        )
        self.register_buffer(
            "replay_ready", torch.zeros(self.history_size, dtype=torch.bool)
        )
        # Eligibility is read from a snapshot, never from live counts, so a
        # blurry stream trickling old-class samples into every step cannot
        # change which classes are replayable.

    @torch.no_grad()
    def _merge_low_rank_sketch(
            self,
            class_id: int,
            class_features: torch.Tensor,
            batch_mean: torch.Tensor,
            old_count: int,
            batch_count: int,
            total_count: int,
            delta: torch.Tensor) -> None:
        centered_batch = class_features - batch_mean
        factors = [self.live_sketch[class_id], centered_batch]
        if old_count > 0:
            correction_scale = (
                float(old_count * batch_count) / float(total_count)
            ) ** 0.5
            factors.append(delta.unsqueeze(0) * correction_scale)
        merged_factor = torch.cat(factors, dim=0)

        # If B^T B is the accumulated M2 matrix, truncating S*Vh is the best
        # rank-r approximation available from the current online sketch.
        _u, singular_values, vh = torch.linalg.svd(
            merged_factor, full_matrices=False
        )
        retained = min(self.covariance_rank, singular_values.numel())
        # Plain truncation is per-step optimal (Eckart-Young) but biased over a
        # stream: directions that never reach the top r are discarded forever,
        # however much energy they accumulate. Frequent Directions removes that
        # bias by subtracting the first discarded eigenvalue -- at the cost of
        # deflating everything when the spectrum is flat, which is exactly the
        # regime a rank-16 sketch of a 768-dim feature is likely to be in. So
        # this is a measured choice, not a default.
        if (self.sketch_shrinkage == "fd"
                and singular_values.numel() > retained):
            shrinkage = singular_values[retained].square()
        else:
            shrinkage = singular_values.new_zeros(())
        retained_values = (
            singular_values[:retained].square() - shrinkage
        ).clamp_min(0.0).sqrt()
        self.live_sketch[class_id].zero_()
        self.live_sketch[class_id, :retained].copy_(
            retained_values.unsqueeze(1) * vh[:retained]
        )

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Merge a batch with exact parallel-Welford class statistics."""
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError(
                "features must have shape [B, feature_dim], got "
                f"{tuple(features.shape)}"
            )
        if labels.ndim != 1 or labels.size(0) != features.size(0):
            raise ValueError(
                "labels must have shape [B] and match features, got "
                f"{tuple(labels.shape)} for {tuple(features.shape)}"
            )
        if labels.numel() == 0:
            return

        features = features.detach().to(
            device=self.live_mean.device, dtype=self.live_mean.dtype
        )
        labels = labels.detach().to(
            device=self.live_count.device, dtype=torch.long
        )
        if labels.min().item() < 0 or labels.max().item() >= self.num_classes:
            raise ValueError("labels contain a class index outside the statistics bank")
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("clean CLS features contain NaN or Inf")

        for class_id_tensor in torch.unique(labels, sorted=True):
            class_id = int(class_id_tensor.item())
            class_features = features[labels == class_id]
            batch_count = int(class_features.size(0))
            batch_mean = class_features.mean(dim=0)
            centered = class_features - batch_mean
            batch_m2 = (centered * centered).sum(dim=0)

            old_count = int(self.live_count[class_id].item())
            total_count = old_count + batch_count
            delta = batch_mean - self.live_mean[class_id]

            if self.statistic_type == "low_rank":
                self._merge_low_rank_sketch(
                    class_id=class_id,
                    class_features=class_features,
                    batch_mean=batch_mean,
                    old_count=old_count,
                    batch_count=batch_count,
                    total_count=total_count,
                    delta=delta,
                )

            if old_count == 0:
                self.live_count[class_id] = batch_count
                self.live_mean[class_id].copy_(batch_mean)
                self.live_m2[class_id].copy_(batch_m2)
                continue

            merged_mean = (
                self.live_mean[class_id]
                + delta * (float(batch_count) / float(total_count))
            )
            merged_m2 = (
                self.live_m2[class_id]
                + batch_m2
                + delta.square()
                * (float(old_count * batch_count) / float(total_count))
            )
            self.live_count[class_id] = total_count
            self.live_mean[class_id].copy_(merged_mean)
            self.live_m2[class_id].copy_(merged_m2)

    @torch.no_grad()
    def snapshot(self) -> bool:
        """Push a teacher-aligned copy of the live statistics into history."""
        for slot in range(self.history_size - 1, 0, -1):
            self.replay_count[slot].copy_(self.replay_count[slot - 1])
            self.replay_mean[slot].copy_(self.replay_mean[slot - 1])
            self.replay_m2[slot].copy_(self.replay_m2[slot - 1])
            self.replay_sketch[slot].copy_(self.replay_sketch[slot - 1])
            self.replay_eligible[slot].copy_(self.replay_eligible[slot - 1])
            self.replay_ready[slot].copy_(self.replay_ready[slot - 1])

        self.replay_count[0].copy_(self.live_count)
        self.replay_mean[0].copy_(self.live_mean)
        self.replay_m2[0].copy_(self.live_m2)
        self.replay_sketch[0].copy_(self.live_sketch)
        self.replay_eligible[0].copy_(
            self.live_count >= self.min_replay_samples
        )
        ready = bool(self.replay_eligible[0].any().item())
        self.replay_ready[0].fill_(ready)
        return ready

    def eligible_count(self, eligibility_slot: int) -> int:
        return int(self.replay_eligible[eligibility_slot].sum().item())

    def replay_available(self, eligibility_slot: int) -> bool:
        return bool(self.replay_ready[eligibility_slot].item())

    def _pooled_within_class_variance(
            self, snapshot_slot: int) -> torch.Tensor:
        eligible = self.replay_eligible[snapshot_slot]
        counts = self.replay_count[snapshot_slot, eligible].to(
            self.replay_mean.dtype
        )
        m2 = self.replay_m2[snapshot_slot, eligible]
        degrees_of_freedom = counts.sum() - float(counts.numel())
        return m2.sum(dim=0) / degrees_of_freedom.clamp_min(1.0)

    def _variance_components(
            self,
            snapshot_slot: int,
            class_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        means = self.replay_mean[snapshot_slot, class_ids]
        counts = self.replay_count[snapshot_slot, class_ids].to(means.dtype)
        class_variance = self.replay_m2[snapshot_slot, class_ids] / (
            counts.unsqueeze(1) - 1.0
        ).clamp_min(1.0)
        pooled_variance = self._pooled_within_class_variance(
            snapshot_slot
        ).unsqueeze(0)
        shrinkage = counts / (counts + self.variance_prior_strength)
        total_variance = (
            shrinkage.unsqueeze(1) * class_variance
            + (1.0 - shrinkage).unsqueeze(1) * pooled_variance
        ).clamp_min(self.variance_floor)
        if self.variance_ceiling is not None:
            total_variance = total_variance.clamp_max(self.variance_ceiling)
        return total_variance, shrinkage

    def distribution(
            self,
            snapshot_slot: int,
            class_ids: torch.Tensor):
        """Return (means, low_rank_factor_or_None, residual_variance).

        The construction guarantees
        ``diag(factor^T factor) + residual == total_variance``, so the diagonal
        and low-rank branches describe identical per-dimension marginals and
        differ only in the off-diagonal structure. Both the sampler and the
        closed-form objective read the distribution from here.
        """
        means = self.replay_mean[snapshot_slot, class_ids]
        total_variance, shrinkage = self._variance_components(
            snapshot_slot, class_ids
        )
        if self.statistic_type == "diagonal":
            return means, None, total_variance

        counts = self.replay_count[snapshot_slot, class_ids].to(means.dtype)
        covariance_factor = self.replay_sketch[
            snapshot_slot, class_ids
        ] / (counts - 1.0).clamp_min(1.0).sqrt().view(-1, 1, 1)
        covariance_factor = (
            covariance_factor * shrinkage.sqrt().view(-1, 1, 1)
        )

        # Preserve the dominant correlated component without exceeding the
        # exact Welford diagonal after shrinkage and clipping.
        low_rank_diagonal = covariance_factor.square().sum(dim=1)
        factor_scale = torch.minimum(
            torch.ones_like(low_rank_diagonal),
            (total_variance / low_rank_diagonal.clamp_min(1e-12)).sqrt(),
        )
        covariance_factor = covariance_factor * factor_scale.unsqueeze(1)
        low_rank_diagonal = covariance_factor.square().sum(dim=1)
        residual_variance = (
            total_variance - low_rank_diagonal
        ).clamp_min(0.0)

        return means, covariance_factor, residual_variance

    def _sample_local(
            self,
            snapshot_slot: int,
            class_ids: torch.Tensor) -> torch.Tensor:
        means, covariance_factor, residual_variance = self.distribution(
            snapshot_slot, class_ids
        )
        diagonal_noise = residual_variance.sqrt() * torch.randn_like(means)
        if covariance_factor is None:
            return means + diagonal_noise
        rank_noise = means.new_empty(
            (means.size(0), self.covariance_rank)
        ).normal_()
        correlated_noise = torch.einsum(
            "br,brd->bd", rank_noise, covariance_factor
        )
        return means + correlated_noise + diagonal_noise

    @torch.no_grad()
    def select_classes(
            self,
            eligibility_slot: int,
            class_budget: int) -> torch.Tensor:
        """Uniformly pick replay classes, identically on every DDP rank.

        ``eligibility_slot`` may be older than the slot the distribution is
        read from: a teacher frozen at step t should not be asked to preserve
        classes that were introduced during step t itself.
        """
        if eligibility_slot < 0 or eligibility_slot >= self.history_size:
            raise ValueError(f"invalid eligibility_slot {eligibility_slot}")
        if class_budget <= 0:
            raise ValueError(f"class_budget must be positive, got {class_budget}")

        empty = self.replay_count.new_empty((0,), dtype=torch.long)
        if not self.replay_available(eligibility_slot):
            return empty

        eligible_ids = self.replay_eligible[eligibility_slot].nonzero(
            as_tuple=False
        ).flatten()
        if eligible_ids.numel() == 0:
            return empty

        selected_count = min(int(class_budget), int(eligible_ids.numel()))
        distributed = dist.is_available() and dist.is_initialized()
        if not distributed or dist.get_rank() == 0:
            order = torch.randperm(
                eligible_ids.numel(), device=eligible_ids.device
            )
            class_ids = eligible_ids[order[:selected_count]].contiguous()
        else:
            class_ids = eligible_ids.new_empty((selected_count,))
        # Keep the effective replay budget invariant to DDP world size.
        if distributed:
            dist.broadcast(class_ids, src=0)
        return class_ids

    @torch.no_grad()
    def sample_anchors(
            self,
            snapshot_slot: int,
            class_budget: int,
            anchors_per_class: int = 1,
            eligibility_slot: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select replay classes and draw ``anchors_per_class`` anchors each.

        ``anchors_per_class`` controls how well the draw resolves the stored
        second-order structure. At 1 the loss only ever sees a rank-1 estimate
        of the anchor second moment, which no covariance model can influence.
        """
        if anchors_per_class <= 0:
            raise ValueError(
                f"anchors_per_class must be positive, got {anchors_per_class}"
            )
        class_ids = self.select_classes(
            snapshot_slot if eligibility_slot is None else eligibility_slot,
            class_budget,
        )
        if class_ids.numel() == 0:
            return (
                self.replay_mean.new_empty((0, self.feature_dim)),
                class_ids,
            )

        class_ids = class_ids.repeat_interleave(int(anchors_per_class))
        distributed = dist.is_available() and dist.is_initialized()
        if not distributed or dist.get_rank() == 0:
            anchors = self._sample_local(snapshot_slot, class_ids).contiguous()
        else:
            anchors = self.replay_mean.new_empty(
                (class_ids.numel(), self.feature_dim)
            )
        # The draw is random, so only rank 0 may produce it.
        if distributed:
            dist.broadcast(anchors, src=0)
        return anchors, class_ids


class MLPPromptGenerator(nn.Module):
    """Generate layer-specific prompts from one clean final CLS vector."""

    def __init__(
            self,
            embed_dim: int,
            prompt_length: int,
            prompt_count: int,
            mlp_layers: int = 1,
            hidden_dim: Optional[int] = None):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if prompt_length <= 0 or prompt_count <= 0:
            raise ValueError("prompt_length and prompt_count must be positive")
        if mlp_layers not in {1, 2, 3}:
            raise ValueError(f"mlp_layers must be 1, 2 or 3, got {mlp_layers}")
        hidden_dim = embed_dim if hidden_dim is None else int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.embed_dim = int(embed_dim)
        self.prompt_length = int(prompt_length)
        self.prompt_count = int(prompt_count)
        self.mlp_layers = int(mlp_layers)
        self.hidden_dim = hidden_dim
        output_dim = self.prompt_count * self.prompt_length * self.embed_dim

        layers = []
        input_dim = self.embed_dim
        for _ in range(self.mlp_layers - 1):
            layers.extend([nn.Linear(input_dim, self.hidden_dim), nn.GELU()])
            input_dim = self.hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))

        self.condition_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(*layers)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, clean_cls: torch.Tensor) -> torch.Tensor:
        if clean_cls.ndim != 2 or clean_cls.size(1) != self.embed_dim:
            raise ValueError(
                "clean_cls must have shape [B, embed_dim], got "
                f"{tuple(clean_cls.shape)}"
            )
        prompt = self.mlp(self.condition_norm(clean_cls))
        return prompt.reshape(
            clean_cls.size(0),
            self.prompt_count,
            self.prompt_length,
            self.embed_dim,
        )


class MLPGenerator(nn.Module):
    """Frozen ViT with clean-CLS-conditioned MLP prompt generation."""

    def __init__(
            self,
            task_num: int = 10,
            num_classes: int = 100,
            backbone_name: Optional[str] = None,
            len_prompt: int = 10,
            prompt_blocks: Iterable[int] = (0,),
            generator_layers: int = 1,
            generator_hidden_dim: Optional[int] = None,
            replay_class_budget: int = 8,
            min_replay_samples: int = 2,
            statistic_type: str = "diagonal",
            covariance_rank: int = 16,
            variance_prior_strength: float = 16.0,
            variance_floor: float = 1e-5,
            variance_ceiling: Optional[float] = 10.0,
            teacher_lags: Iterable[int] = (1,),
            distill_metric: str = "mse",
            sketch_shrinkage: str = "none",
            distill_weight: float = 1.0,
            distill_weight_mode: str = "fixed",
            learnable_distill_weight: bool = False,
            anchors_per_class: int = 1,
            replay_eligibility: str = "previous",
            distill_mode: str = "sample",
            statistics_space: str = "raw",
            **kwargs):
        super().__init__()

        if backbone_name is None:
            raise ValueError("backbone_name must be specified")
        if task_num <= 1:
            raise ValueError(f"task_num must be greater than 1, got {task_num}")
        if replay_class_budget <= 0:
            raise ValueError(
                "replay_class_budget must be positive, got "
                f"{replay_class_budget}"
            )
        if distill_metric not in {"mse", "cosine"}:
            raise ValueError(
                f"distill_metric must be mse or cosine, got {distill_metric}"
            )
        if distill_weight < 0:
            raise ValueError(
                f"distill_weight must be non-negative, got {distill_weight}"
            )
        distill_weight_mode = resolve_distill_weight_mode(
            distill_weight_mode, learnable_distill_weight
        )
        if anchors_per_class <= 0:
            raise ValueError(
                f"anchors_per_class must be positive, got {anchors_per_class}"
            )
        if replay_eligibility not in {"teacher", "previous"}:
            raise ValueError(
                "replay_eligibility must be teacher or previous, got "
                f"{replay_eligibility}"
            )
        if distill_mode not in {"sample", "closed_form"}:
            raise ValueError(
                f"distill_mode must be sample or closed_form, got {distill_mode}"
            )
        if statistics_space not in {"raw", "normalized"}:
            raise ValueError(
                "statistics_space must be raw or normalized, got "
                f"{statistics_space}"
            )
        if distill_mode == "closed_form":
            if int(generator_layers) != 1:
                raise ValueError(
                    "closed_form distillation needs an affine generator: it is "
                    "only defined for --generator_layers 1, got "
                    f"{generator_layers}"
                )
            if distill_metric != "mse":
                raise ValueError(
                    "closed_form distillation evaluates a quadratic form, so it "
                    "only supports --distill_metric mse; cosine has no closed "
                    "form"
                )
            if statistics_space != "normalized":
                # The closed form treats the statistics variable as the
                # generator's post-normalisation input, so the statistics have
                # to live in that space for the identity to hold.
                statistics_space = "normalized"
        if distill_weight_mode == "uncertainty" and distill_weight <= 0:
            raise ValueError(
                "uncertainty weighting needs a positive --distill_weight as its "
                "initial value; use fixed mode with --distill_weight 0 for the "
                "no-distillation control"
            )

        prompt_blocks = tuple(int(block) for block in prompt_blocks)
        if not prompt_blocks or tuple(sorted(set(prompt_blocks))) != prompt_blocks:
            raise ValueError(
                "prompt_blocks must be a non-empty, strictly increasing list"
            )
        teacher_lags = tuple(sorted(set(int(lag) for lag in teacher_lags)))
        if not teacher_lags or teacher_lags[0] <= 0:
            raise ValueError("teacher_lags must contain positive integers")
        if teacher_lags[-1] >= task_num:
            raise ValueError(
                "the largest teacher lag must be smaller than step_num/task_num "
                f"so it can become active: {teacher_lags[-1]} >= {task_num}"
            )
        if replay_class_budget < len(teacher_lags):
            raise ValueError(
                "replay_class_budget must allow at least one class per teacher"
            )

        self.task_num = int(task_num)
        self.num_classes = int(num_classes)
        self.prompt_length = int(len_prompt)
        self.prompt_blocks = prompt_blocks
        self.generator_layers = int(generator_layers)
        self.replay_class_budget = int(replay_class_budget)
        self.teacher_lags = teacher_lags
        self.history_size = teacher_lags[-1]
        # The teacher bank keeps max_lag snapshots. The statistics bank keeps
        # one more, so the lag-k teacher can read its eligible classes from the
        # snapshot taken one internal step before it was frozen.
        self.statistics_history = self.history_size + 1
        self.distill_metric = distill_metric
        self.distill_weight = float(distill_weight)
        self.distill_weight_mode = str(distill_weight_mode)
        self.learnable_distill_weight = self.distill_weight_mode == "uncertainty"
        self.anchors_per_class = int(anchors_per_class)
        self.replay_eligibility = str(replay_eligibility)
        self.distill_mode = str(distill_mode)
        self.statistics_space = str(statistics_space)
        # fixed/0 is the no-teacher control: the teacher forward and anchor
        # sampling are skipped entirely rather than multiplied by zero.
        self.distillation_enabled = (
            self.distill_weight_mode != "fixed" or self.distill_weight > 0
        )
        self.task_count = 0
        self.snapshot_count = 0

        self.backbone = create_backbone(
            backbone_name,
            num_classes=self.num_classes,
            pretrained=kwargs.get("pretrained", True),
            backbone_path=kwargs.get("backbone_path"),
        )
        self.embed_dim = int(self.backbone.num_features)
        if self.backbone.cls_token is None:
            raise ValueError("MLPGenerator requires a CLS-token backbone")
        depth = len(self.backbone.blocks)
        if self.prompt_blocks[0] < 0 or self.prompt_blocks[-1] >= depth:
            raise ValueError(
                f"prompt_blocks {self.prompt_blocks} exceed backbone depth {depth}"
            )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        if self.backbone.fc.bias is not None:
            self.backbone.fc.bias.requires_grad = True

        self.prompt_generator = MLPPromptGenerator(
            embed_dim=self.embed_dim,
            prompt_length=self.prompt_length,
            prompt_count=len(self.prompt_blocks),
            mlp_layers=self.generator_layers,
            hidden_dim=generator_hidden_dim,
        )
        self.teacher_generators = nn.ModuleList([
            copy.deepcopy(self.prompt_generator)
            for _ in range(self.history_size)
        ])
        for teacher in self.teacher_generators:
            for parameter in teacher.parameters():
                parameter.requires_grad = False
            teacher.eval()
        self.register_buffer(
            "teacher_valid", torch.zeros(self.history_size, dtype=torch.bool)
        )

        # Homoscedastic-uncertainty weighting. Minimising exp(-s)*D + s over s
        # gives exp(-s) = 1/D, so the weighted term settles at a constant
        # contribution whatever raw scale the metric happens to have. That is
        # what removes the mse-vs-cosine scale confound.
        if self.distill_weight_mode == "uncertainty":
            self.log_distill_scale = nn.Parameter(
                torch.tensor(-math.log(self.distill_weight))
            )
        else:
            self.register_parameter("log_distill_scale", None)

        self.class_statistics = ClassFeatureStatistics(
            num_classes=self.num_classes,
            feature_dim=self.embed_dim,
            history_size=self.statistics_history,
            statistic_type=statistic_type,
            covariance_rank=covariance_rank,
            min_replay_samples=min_replay_samples,
            variance_prior_strength=variance_prior_strength,
            variance_floor=variance_floor,
            variance_ceiling=variance_ceiling,
            sketch_shrinkage=sketch_shrinkage,
        )

        self.last_replay_class_count = 0
        self.last_active_teacher_lags = tuple()
        self.last_raw_distillation = 0.0
        self.last_effective_distill_weight = float(self.distill_weight)
        self.last_log_distill_scale = (
            -math.log(self.distill_weight)
            if self.distill_weight_mode == "uncertainty"
            else 0.0
        )
        self._prompt_slot_by_block = {
            block: slot for slot, block in enumerate(self.prompt_blocks)
        }

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen modules must also remain behaviorally frozen.
        self.backbone.eval()
        for teacher in self.teacher_generators:
            teacher.eval()
        return self

    @torch.no_grad()
    def encode_clean(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the single cacheable no-prompt backbone pass."""
        self.backbone.eval()
        clean_tokens = self.backbone.forward_features(inputs)
        return clean_tokens[:, 0].detach()

    def _forward_prompted(
            self,
            inputs: torch.Tensor,
            clean_cls: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        x = backbone.patch_embed(inputs)
        batch_size = x.size(0)
        cls_token = backbone.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        if x.size(1) != backbone.pos_embed.size(1):
            raise ValueError(
                "input token count does not match the backbone positional "
                f"embedding: {x.size(1)} != {backbone.pos_embed.size(1)}"
            )
        x = backbone.pos_drop(x + backbone.pos_embed)

        generated_prompts = self.prompt_generator(clean_cls.detach())
        prompt_position = backbone.pos_embed[:, :1, :].unsqueeze(1)
        generated_prompts = generated_prompts + prompt_position

        for block_index, block in enumerate(backbone.blocks):
            prompt_slot = self._prompt_slot_by_block.get(block_index)
            if prompt_slot is None:
                x = block(x)
                continue
            prompt = generated_prompts[:, prompt_slot]
            x = torch.cat((x[:, :1], prompt, x[:, 1:]), dim=1)
            x = block(x)
            x = torch.cat(
                (x[:, :1], x[:, self.prompt_length + 1:]), dim=1
            )

        x = backbone.norm(x)
        return x[:, 0]

    def _teacher_loss(
            self,
            student_prompt: torch.Tensor,
            teacher_prompt: torch.Tensor) -> torch.Tensor:
        if self.distill_metric == "mse":
            return F.mse_loss(student_prompt, teacher_prompt)
        student_flat = student_prompt.flatten(start_dim=1)
        teacher_flat = teacher_prompt.flatten(start_dim=1)
        return (
            1.0 - F.cosine_similarity(student_flat, teacher_flat, dim=1)
        ).mean()

    def _allocate_teacher_budgets(
            self,
            eligibility_slots: Tuple[int, ...]) -> Tuple[int, ...]:
        capacities = [
            self.class_statistics.eligible_count(slot)
            for slot in eligibility_slots
        ]
        budgets = [0 for _ in eligibility_slots]
        remaining = self.replay_class_budget
        while remaining > 0:
            progressed = False
            for index, capacity in enumerate(capacities):
                if remaining == 0:
                    break
                if budgets[index] < capacity:
                    budgets[index] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        return tuple(budgets)

    def _eligibility_slot(self, teacher_slot: int) -> int:
        """Snapshot slot that decides which classes this teacher may replay.

        'previous' steps one snapshot further back, so a class introduced while
        the teacher itself was still learning it is not treated as an old class.
        Unlike a live-count rule, this is unaffected by a blurry stream that
        trickles old-class samples into every step.
        """
        if self.replay_eligibility == "previous":
            return teacher_slot + 1
        return teacher_slot

    @staticmethod
    def _affine_parameters(generator: MLPPromptGenerator):
        """Fold condition_norm's affine into the Linear.

        With one Linear layer, g(a) = W (gamma * ahat + beta) + c is affine in
        the normalised input ahat, with weight W diag(gamma) and bias
        W beta + c.
        """
        norm = generator.condition_norm
        linear = generator.mlp[0]
        return (
            linear.weight * norm.weight.unsqueeze(0),
            linear.weight @ norm.bias + linear.bias,
        )

    def _closed_form_teacher_loss(
            self,
            snapshot_slot: int,
            class_ids: torch.Tensor) -> torch.Tensor:
        """Exact population MSE between the student and teacher generators.

        For an affine generator and z ~ (mu, Sigma),

            E|| dA z + db ||^2 = || dA mu + db ||^2 + tr(dA Sigma dA^T)

        so no anchors are drawn and the objective carries no estimator noise.
        The stored low-rank factor makes the trace term cost O(C r d p) instead
        of materialising a d x d covariance.
        """
        with torch.cuda.amp.autocast(enabled=False):
            student_weight, student_bias = self._affine_parameters(
                self.prompt_generator
            )
            with torch.no_grad():
                teacher_weight, teacher_bias = self._affine_parameters(
                    self.teacher_generators[snapshot_slot]
                )
            delta_weight = student_weight.float() - teacher_weight.float()
            delta_bias = student_bias.float() - teacher_bias.float()

            means, factor, residual = self.class_statistics.distribution(
                snapshot_slot, class_ids
            )
            means = means.float()
            residual = residual.float()

            # || dA mu + db ||^2 : discrepancy at each class mean.
            mean_term = (
                means @ delta_weight.t() + delta_bias
            ).square().sum(dim=1)

            # tr(dA diag(residual) dA^T) is one dot product per class.
            column_energy = delta_weight.square().sum(dim=0)
            variance_term = residual @ column_energy

            if factor is not None:
                factor = factor.float()
                class_count, rank, feature_dim = factor.shape
                projected = factor.reshape(
                    class_count * rank, feature_dim
                ) @ delta_weight.t()
                variance_term = variance_term + (
                    projected.square().sum(dim=1).view(class_count, rank)
                ).sum(dim=1)

            # F.mse_loss averages over prompt elements; match that scale so the
            # closed-form and sampled objectives are numerically comparable.
            return (
                (mean_term + variance_term) / float(delta_weight.size(0))
            ).mean()

    def prompt_distillation_loss(
            self,
            current_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.distillation_enabled:
            self.last_active_teacher_lags = tuple()
            self.last_replay_class_count = 0
            return next(self.prompt_generator.parameters()).new_zeros(())

        del current_labels

        # A teacher is usable when it has been frozen and the snapshot it reads
        # its eligible classes from is populated. Those are different snapshots
        # under replay_eligibility='previous'.
        active = []
        for lag in self.teacher_lags:
            slot = lag - 1
            eligibility_slot = self._eligibility_slot(slot)
            if (
                    bool(self.teacher_valid[slot].item())
                    and self.class_statistics.replay_available(
                        eligibility_slot)):
                active.append((lag, slot, eligibility_slot))

        self.last_active_teacher_lags = tuple(lag for lag, _s, _e in active)
        if not active:
            self.last_replay_class_count = 0
            return next(self.prompt_generator.parameters()).new_zeros(())

        budgets = self._allocate_teacher_budgets(
            tuple(eligibility for _lag, _slot, eligibility in active)
        )
        weighted_losses = []
        total_anchors = 0
        for (_lag, slot, eligibility_slot), budget in zip(active, budgets):
            if budget == 0:
                continue
            if self.distill_mode == "closed_form":
                class_ids = self.class_statistics.select_classes(
                    eligibility_slot=eligibility_slot,
                    class_budget=budget,
                )
                if class_ids.numel() == 0:
                    continue
                teacher_loss = self._closed_form_teacher_loss(slot, class_ids)
                anchor_count = int(class_ids.numel())
            else:
                anchors, _class_ids = self.class_statistics.sample_anchors(
                    snapshot_slot=slot,
                    class_budget=budget,
                    anchors_per_class=self.anchors_per_class,
                    eligibility_slot=eligibility_slot,
                )
                if anchors.numel() == 0:
                    continue
                student_prompt = self.prompt_generator(anchors)
                with torch.no_grad():
                    teacher_prompt = self.teacher_generators[slot](anchors)
                teacher_loss = self._teacher_loss(
                    student_prompt, teacher_prompt
                )
                anchor_count = int(anchors.size(0))
            weighted_losses.append(teacher_loss * float(anchor_count))
            total_anchors += anchor_count

        self.last_replay_class_count = total_anchors
        if total_anchors == 0:
            return next(self.prompt_generator.parameters()).new_zeros(())
        return torch.stack(weighted_losses).sum() / float(total_anchors)

    def _weight_distillation(self, raw_distillation: torch.Tensor):
        """Prepare the distillation term for the configured weighting mode.

        GradNorm needs the raw distance because its detached coefficient is
        computed by the trainer from both CE and distillation gradients.
        """
        self.last_raw_distillation = float(raw_distillation.detach().item())
        if not self.distillation_enabled or self.last_replay_class_count == 0:
            # No anchors this step: emitting the bare log-scale term here would
            # let s drift upward on steps that carry no distillation signal.
            self.last_effective_distill_weight = 0.0
            self.last_log_distill_scale = (
                float(self.log_distill_scale.detach().clamp(-9.21, 9.21).item())
                if self.distill_weight_mode == "uncertainty"
                else 0.0
            )
            return raw_distillation * 0.0
        if self.distill_weight_mode == "gradnorm":
            self.last_effective_distill_weight = 0.0
            self.last_log_distill_scale = 0.0
            return raw_distillation
        if self.distill_weight_mode == "fixed":
            self.last_effective_distill_weight = self.distill_weight
            self.last_log_distill_scale = 0.0
            return self.distill_weight * raw_distillation
        log_scale = self.log_distill_scale.clamp(-9.21, 9.21)
        weight = torch.exp(-log_scale)
        self.last_effective_distill_weight = float(weight.detach().item())
        self.last_log_distill_scale = float(log_scale.detach().item())
        return weight * raw_distillation + log_scale

    def reset_distillation_observation(self) -> None:
        """Clear per-update diagnostics without entering the teacher branch."""
        self.last_active_teacher_lags = tuple()
        self.last_replay_class_count = 0
        self.last_raw_distillation = 0.0
        self.last_effective_distill_weight = 0.0
        self.last_log_distill_scale = (
            float(self.log_distill_scale.detach().clamp(-9.21, 9.21).item())
            if self.distill_weight_mode == "uncertainty"
            else 0.0
        )

    def forward(
            self,
            inputs: torch.Tensor,
            clean_cls: Optional[torch.Tensor] = None,
            return_distill: bool = False,
            current_labels: Optional[torch.Tensor] = None,
            **kwargs):
        del kwargs
        if clean_cls is None:
            clean_cls = self.encode_clean(inputs)
        prompted_cls = self._forward_prompted(inputs, clean_cls)
        logits = self.backbone.fc(prompted_cls)
        if not return_distill:
            # Keep the legacy uncertainty parameter present in the DDP graph
            # during delayed batches without sampling anchors or running a
            # teacher. This is exactly zero and does not update s.
            if self.distill_weight_mode == "uncertainty":
                logits = logits + self.log_distill_scale * 0.0
            return logits
        raw_distillation = self.prompt_distillation_loss(current_labels)
        return logits, self._weight_distillation(raw_distillation)

    @torch.no_grad()
    def update_class_statistics(
            self, clean_cls: torch.Tensor, labels: torch.Tensor) -> None:
        if self.statistics_space == "normalized":
            # The generator's condition_norm always applies this, and it is
            # parameter-free. Storing statistics after it removes the shared
            # mean direction that otherwise dominates the second moment, and it
            # is what makes the closed-form objective exact.
            clean_cls = F.layer_norm(clean_cls, (self.embed_dim,))
        self.class_statistics.update(clean_cls, labels)

    @torch.no_grad()
    def snapshot_teacher(self) -> bool:
        """Roll teacher/statistic histories, then store the current window."""
        for slot in range(self.history_size - 1, 0, -1):
            self.teacher_generators[slot].load_state_dict(
                self.teacher_generators[slot - 1].state_dict()
            )
            self.teacher_valid[slot].copy_(self.teacher_valid[slot - 1])
        self.teacher_generators[0].load_state_dict(
            self.prompt_generator.state_dict()
        )
        self.teacher_generators[0].eval()
        self.teacher_valid[0].fill_(True)
        ready = self.class_statistics.snapshot()
        self.snapshot_count += 1
        return ready

    def process_task_count(self) -> None:
        # Called after the last optimized batch of an internal sample window.
        self.snapshot_teacher()
        self.task_count += 1

    def update(self) -> None:
        """No routing head needs refreshing before evaluation."""

    def loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(output, target)
