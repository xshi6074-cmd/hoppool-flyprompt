import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


def resolve_distill_weight_mode(mode: str, legacy_learnable: bool) -> str:
    """Resolve the legacy flag without making GradNorm silently ambiguous."""
    mode = str(mode)
    if mode not in {"fixed", "uncertainty", "gradnorm"}:
        raise ValueError(
            "distill_weight_mode must be fixed, uncertainty or gradnorm, got "
            f"{mode}"
        )
    if legacy_learnable:
        if mode == "gradnorm":
            raise ValueError(
                "--learnable_distill_weight is the legacy uncertainty alias "
                "and cannot be combined with --distill_weight_mode gradnorm"
            )
        return "uncertainty"
    return mode


@dataclass(frozen=True)
class GradNormResult:
    """Detached diagnostics for one GradNorm-lite controller update."""

    valid: bool
    weight: float = 0.0
    target_weight: float = 0.0
    classification_grad_norm: float = 0.0
    distillation_grad_norm: float = 0.0
    actual_grad_ratio: float = 0.0
    gradient_cosine: float = 0.0
    boundary_hit: bool = False
    reason: str = ""


class DistillationDelayGate:
    """Whole-batch delay measured in global stream samples after snapshots."""

    def __init__(self, delay_samples: int):
        if int(delay_samples) < 0:
            raise ValueError(
                f"distill_delay_samples must be non-negative, got {delay_samples}"
            )
        self.delay_samples = int(delay_samples)
        self.snapshot_count = 0
        self.samples_since_snapshot = 0

    @property
    def state(self) -> str:
        if self.snapshot_count == 0:
            return "waiting_teacher"
        if self.samples_since_snapshot < self.delay_samples:
            return "delay"
        return "active"

    @property
    def active(self) -> bool:
        return self.state == "active"

    def finish_batch(self, observed_snapshot_count: int, batch_samples: int) -> bool:
        """Record one completed batch and return whether a snapshot occurred."""
        observed_snapshot_count = int(observed_snapshot_count)
        batch_samples = int(batch_samples)
        if observed_snapshot_count < self.snapshot_count:
            raise ValueError(
                "snapshot_count cannot move backwards: "
                f"{observed_snapshot_count} < {self.snapshot_count}"
            )
        if batch_samples < 0:
            raise ValueError(f"batch_samples must be non-negative, got {batch_samples}")

        if observed_snapshot_count > self.snapshot_count:
            self.snapshot_count = observed_snapshot_count
            self.samples_since_snapshot = 0
            return True
        if self.snapshot_count > 0:
            self.samples_since_snapshot += batch_samples
        return False


class GradNormLiteController:
    """Set one auxiliary weight from its gradient norm relative to CE.

    The controller is deliberately not an ``nn.Module``: its weight is a
    detached control value, not a parameter optimized through the task loss.
    """

    def __init__(
            self,
            target_ratio: float,
            log_ema: float,
            weight_min: float,
            weight_max: float,
            eps: float = 1e-12):
        if not math.isfinite(target_ratio) or target_ratio <= 0:
            raise ValueError(
                f"distill_grad_ratio must be positive and finite, got {target_ratio}"
            )
        if not math.isfinite(log_ema) or not 0 <= log_ema < 1:
            raise ValueError(
                f"distill_grad_ema must be in [0, 1), got {log_ema}"
            )
        if not math.isfinite(weight_min) or weight_min <= 0:
            raise ValueError(
                f"distill_weight_min must be positive and finite, got {weight_min}"
            )
        if not math.isfinite(weight_max) or weight_max < weight_min:
            raise ValueError(
                "distill_weight_max must be finite and no smaller than "
                f"distill_weight_min, got {weight_max} < {weight_min}"
            )
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"eps must be positive and finite, got {eps}")

        self.target_ratio = float(target_ratio)
        self.log_ema = float(log_ema)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.eps = float(eps)
        self._ema_log_weight: Optional[float] = None

    def reset(self) -> None:
        self._ema_log_weight = None

    @staticmethod
    def _gradient(
            loss: torch.Tensor,
            reference_parameter: torch.Tensor) -> Optional[torch.Tensor]:
        if not loss.requires_grad:
            return None
        gradient, = torch.autograd.grad(
            loss,
            reference_parameter,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        return gradient

    @staticmethod
    def _global_statistics(
            classification_gradient: torch.Tensor,
            distillation_gradient: torch.Tensor) -> torch.Tensor:
        classification_gradient = classification_gradient.detach().float()
        distillation_gradient = distillation_gradient.detach().float()
        statistics = torch.stack((
            classification_gradient.square().sum(),
            distillation_gradient.square().sum(),
            (classification_gradient * distillation_gradient).sum(),
        ))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        return statistics

    def compute(
            self,
            classification_loss: torch.Tensor,
            distillation_loss: torch.Tensor,
            reference_parameter: torch.Tensor) -> GradNormResult:
        classification_gradient = self._gradient(
            classification_loss, reference_parameter
        )
        distillation_gradient = self._gradient(
            distillation_loss, reference_parameter
        )
        if classification_gradient is None or distillation_gradient is None:
            return GradNormResult(valid=False, reason="missing_gradient")

        statistics = self._global_statistics(
            classification_gradient, distillation_gradient
        )
        if not bool(torch.isfinite(statistics).all().item()):
            return GradNormResult(valid=False, reason="nonfinite_gradient")

        classification_norm = float(statistics[0].sqrt().item())
        distillation_norm = float(statistics[1].sqrt().item())
        if classification_norm <= self.eps:
            return GradNormResult(valid=False, reason="zero_classification_gradient")
        if distillation_norm <= self.eps:
            return GradNormResult(valid=False, reason="zero_distillation_gradient")

        unclipped_target = (
            self.target_ratio * classification_norm
            / (distillation_norm + self.eps)
        )
        target_weight = min(
            self.weight_max, max(self.weight_min, unclipped_target)
        )
        boundary_hit = (
            unclipped_target <= self.weight_min
            or unclipped_target >= self.weight_max
        )
        target_log_weight = math.log(target_weight)
        if self._ema_log_weight is None:
            self._ema_log_weight = target_log_weight
        else:
            self._ema_log_weight = (
                self.log_ema * self._ema_log_weight
                + (1.0 - self.log_ema) * target_log_weight
            )
        weight = math.exp(self._ema_log_weight)

        denominator = classification_norm * distillation_norm
        gradient_cosine = float(statistics[2].item()) / max(
            denominator, self.eps
        )
        gradient_cosine = min(1.0, max(-1.0, gradient_cosine))
        actual_ratio = weight * distillation_norm / (
            classification_norm + self.eps
        )
        return GradNormResult(
            valid=True,
            weight=weight,
            target_weight=target_weight,
            classification_grad_norm=classification_norm,
            distillation_grad_norm=distillation_norm,
            actual_grad_ratio=actual_ratio,
            gradient_cosine=gradient_cosine,
            boundary_hit=boundary_hit,
        )
