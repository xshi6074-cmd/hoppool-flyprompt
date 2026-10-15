"""Online class distributions shared by prompt and classifier replay."""
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


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
