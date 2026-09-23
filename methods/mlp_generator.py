import gc
import logging
import math
from typing import Tuple

import torch
import torch.distributed as dist

from methods.frozen_vit import FrozenViTTrainer
from utils.distill_gradnorm import (
    DistillationDelayGate,
    GradNormLiteController,
    GradNormResult,
    resolve_distill_weight_mode,
)


logger = logging.getLogger()


class MLPGenerator(FrozenViTTrainer):
    """Online trainer for the clean-CLS-conditioned MLP prompt generator.

    One transformed stream batch produces one detached clean CLS cache.  The
    same transformed images and clean CLS are then reused for all online
    optimizer iterations, while every prompted backbone pass is recomputed so
    it reflects the latest generator parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # _Trainer copies every parsed argument onto self, and the model is not
        # built until setup_distributed_model(), so the configured values must
        # be read here and the resolved model state reported there.
        self.distill_weight = float(getattr(self, "distill_weight", 1.0))
        if self.distill_weight < 0:
            raise ValueError(
                f"distill_weight must be non-negative, got {self.distill_weight}"
            )
        self.learnable_distill_weight = bool(
            getattr(self, "learnable_distill_weight", False)
        )
        self.distill_weight_mode = resolve_distill_weight_mode(
            getattr(self, "distill_weight_mode", "fixed"),
            self.learnable_distill_weight,
        )

        self.distill_delay_samples = int(
            getattr(self, "distill_delay_samples", 100)
        )
        self.distill_grad_ratio = float(
            getattr(self, "distill_grad_ratio", 0.25)
        )
        self.distill_grad_ema = float(
            getattr(self, "distill_grad_ema", 0.9)
        )
        self.distill_weight_min = float(
            getattr(self, "distill_weight_min", 1e-4)
        )
        self.distill_weight_max = float(
            getattr(self, "distill_weight_max", 1e4)
        )
        self._distill_delay_gate = DistillationDelayGate(
            self.distill_delay_samples
        )
        configured_controller = GradNormLiteController(
            target_ratio=self.distill_grad_ratio,
            log_ema=self.distill_grad_ema,
            weight_min=self.distill_weight_min,
            weight_max=self.distill_weight_max,
        )
        self._gradnorm_controller = (
            configured_controller
            if self.distill_weight_mode == "gradnorm"
            else None
        )

        # Keep the method defensive when it is instantiated outside the main
        # registry (for example, in a focused model/method contract test).
        requested_step_num = kwargs.get("step_num", None)
        if requested_step_num is None or requested_step_num <= 0:
            requested_step_num = getattr(self, "n_tasks", None)
        if requested_step_num is None or requested_step_num <= 1:
            raise ValueError(
                "MLPGenerator requires step_num > 1 so hard-teacher windows "
                "can be formed"
            )
        self.step_num = int(requested_step_num)

        self.last_classification_loss = 0.0
        self.last_distillation_loss = 0.0
        self.last_distill_weight = float(self.distill_weight)
        self.last_weighted_distillation = 0.0
        self.last_distillation_objective = 0.0
        self.last_true_total_loss = 0.0
        self.last_gradnorm = GradNormResult(
            valid=False, reason="not_evaluated"
        )
        self.last_distill_batch_state = "waiting_teacher"
        self._reset_loss_accumulators()

    def _reset_loss_accumulators(self):
        """Running totals so a whole run can be judged from the log, not grep."""
        self._acc_steps = 0
        self._acc_active_steps = 0
        self._acc_ce = 0.0
        self._acc_raw_distill = 0.0
        self._acc_weighted_distill = 0.0
        self._acc_distill_objective = 0.0
        self._acc_true_total = 0.0
        self._acc_anchors = 0
        self._acc_gradnorm_valid = 0
        self._acc_gradnorm_boundary_hits = 0
        self._acc_gradnorm_ratio = 0.0
        self._acc_gradnorm_in_band = 0
        self._acc_bad_objectives = 0

    def setup_distributed_model(self):
        super().setup_distributed_model()

        # The weighting lives on the model so DDP synchronises the learned
        # scale; the method only reads it back for reporting.
        model = self.model_without_ddp
        self.distill_weight = float(model.distill_weight)
        self.learnable_distill_weight = bool(model.learnable_distill_weight)
        self.distill_weight_mode = str(model.distill_weight_mode)
        self.last_distill_weight = self.distill_weight

        if not model.distillation_enabled:
            logger.info(
                "[MLPGenerator] teacher distillation disabled "
                "(--distill_weight 0): no-distillation control run"
            )
            return

        logger.info(
            "[MLPGenerator] distillation | mode=%s | metric=%s | "
            "weight_mode=%s | initial_weight=%s | delay_samples=%d | "
            "statistics=%s/%s | anchors_per_class=%d | replay_eligibility=%s",
            model.distill_mode,
            model.distill_metric,
            model.distill_weight_mode,
            model.distill_weight,
            self.distill_delay_samples,
            model.class_statistics.statistic_type,
            model.statistics_space,
            model.anchors_per_class,
            model.replay_eligibility,
        )
        if model.distill_weight_mode == "gradnorm":
            logger.info(
                "[MLPGenerator] GradNorm-lite | reference="
                "prompt_generator.mlp[-1].weight | target_ratio=%.4f | "
                "log_ema=%.4f | weight_bounds=[%.4g, %.4g]",
                self.distill_grad_ratio,
                self.distill_grad_ema,
                self.distill_weight_min,
                self.distill_weight_max,
            )
        if model.distill_mode == "closed_form":
            logger.info(
                "[MLPGenerator] closed_form: no anchors are drawn; "
                "statistics_space forced to 'normalized' and "
                "anchors_per_class is ignored"
            )

    def _init_internal_step_scheduler(self):
        """Spread hard-teacher windows across all configured training epochs."""
        if self.step_num is None or not hasattr(self, "total_samples"):
            return
        if self.step_num <= 1:
            raise ValueError(f"step_num must be > 1, got {self.step_num}")
        sampler = getattr(self, "train_sampler", None)
        task_indices = getattr(sampler, "indices", None)
        if task_indices is None:
            visits_per_epoch = int(self.total_samples)
        elif getattr(sampler, "distributed", False):
            replicas = int(sampler.num_replicas)
            visits_per_epoch = sum(
                (len(indices) // replicas) * replicas
                for indices in task_indices
            )
        else:
            visits_per_epoch = sum(len(indices) for indices in task_indices)

        total_stream_visits = visits_per_epoch * int(self.num_epochs)
        if total_stream_visits <= 0:
            return
        self.samples_per_step = max(1, total_stream_visits // self.step_num)
        self.current_step = 0
        self.current_step_seen_samples = 0

    def _map_stream_labels(self, labels: torch.Tensor) -> torch.Tensor:
        mapped = [
            self.exposed_classes.index(int(label.item())) for label in labels
        ]
        return torch.tensor(mapped, device=self.device, dtype=torch.long)

    def _batch_logit_mask(self, labels: torch.Tensor) -> torch.Tensor:
        if self.no_batchmask:
            return self.mask
        logit_mask = torch.full_like(self.mask, -torch.inf)
        logit_mask[torch.unique(labels)] = 0
        return logit_mask

    def _gather_statistics(
            self,
            clean_cls: torch.Tensor,
            labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Give every DDP rank identical global class statistics."""
        if not dist.is_available() or not dist.is_initialized():
            return clean_cls, labels

        world_size = dist.get_world_size()
        local_size = torch.tensor(
            [clean_cls.size(0)], device=clean_cls.device, dtype=torch.long
        )
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size)
        max_size = max(int(size.item()) for size in gathered_sizes)

        feature_padding = max_size - clean_cls.size(0)
        if feature_padding > 0:
            clean_cls = torch.cat(
                (
                    clean_cls,
                    clean_cls.new_zeros(
                        (feature_padding,) + tuple(clean_cls.shape[1:])
                    ),
                ),
                dim=0,
            )
            labels = torch.cat(
                (labels, labels.new_zeros((feature_padding,))), dim=0
            )

        gathered_features = [torch.zeros_like(clean_cls) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(labels) for _ in range(world_size)]
        dist.all_gather(gathered_features, clean_cls.contiguous())
        dist.all_gather(gathered_labels, labels.contiguous())

        features = torch.cat([
            item[:int(size.item())]
            for item, size in zip(gathered_features, gathered_sizes)
        ], dim=0)
        global_labels = torch.cat([
            item[:int(size.item())]
            for item, size in zip(gathered_labels, gathered_sizes)
        ], dim=0)
        return features, global_labels

    def online_step(self, images, labels, idx):
        del idx
        self.add_new_class(labels)

        update_count = int(self.online_iter)
        if update_count <= 0:
            raise ValueError(
                f"online_iter must yield at least one update, got {self.online_iter}"
            )

        images = images.to(self.device)
        mapped_labels = self._map_stream_labels(labels)

        # Transform once.  Re-transforming inside the update loop would make a
        # single cached clean CLS inconsistent with later prompted forwards.
        transformed_images = self.train_transform(images)
        self.model.train()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            clean_cls = self.model_without_ddp.encode_clean(transformed_images)
        clean_cls = clean_cls.detach()

        statistics_features, statistics_labels = self._gather_statistics(
            clean_cls.float(), mapped_labels
        )
        self.model_without_ddp.update_class_statistics(
            statistics_features, statistics_labels
        )

        logit_mask = self._batch_logit_mask(mapped_labels)
        # Freeze the decision for all online_iter updates of this stream batch.
        # Delayed batches take the logits-only path, so they do not select
        # anchors, run a hard teacher, or update GradNorm state.
        self.last_distill_batch_state = self._distill_delay_gate.state
        distillation_active_for_batch = (
            self.model_without_ddp.distillation_enabled
            and self._distill_delay_gate.active
        )
        total_loss = 0.0
        total_accuracy = 0.0
        for _ in range(update_count):
            loss, accuracy = self._online_train_cached(
                transformed_images,
                mapped_labels,
                clean_cls,
                logit_mask,
                distillation_active_for_batch,
            )
            total_loss += loss
            total_accuracy += accuracy

        # _Trainer calls process_task_count() here when the sample window ends,
        # so the boundary batch and all of its optimizer updates enter the new
        # hard teacher.  Whole batches are deliberately not split at a boundary.
        self._maybe_advance_internal_step(int(statistics_labels.numel()))
        snapshots_after = self.model_without_ddp.snapshot_count
        snapshot_happened = self._distill_delay_gate.finish_batch(
            snapshots_after, int(statistics_labels.numel())
        )
        if snapshot_happened:
            if self._gradnorm_controller is not None:
                self._gradnorm_controller.reset()
            eligible = int(
                self.model_without_ddp.class_statistics.replay_eligible[0]
                .sum()
                .item()
            )
            logger.info(
                "[MLPGenerator] hard snapshot=%d internal_step=%d "
                "eligible_classes=%d | distill_delay_reset=%d",
                snapshots_after,
                self.current_step,
                eligible,
                self._distill_delay_gate.samples_since_snapshot,
            )

        del images, labels, transformed_images, clean_cls
        gc.collect()
        return total_loss / update_count, total_accuracy / update_count

    def _online_train_cached(
            self,
            images: torch.Tensor,
            labels: torch.Tensor,
            clean_cls: torch.Tensor,
            logit_mask: torch.Tensor,
            distillation_active: bool) -> Tuple[float, float]:
        self.model.train()
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            if distillation_active:
                logits, prepared_distillation = self.model(
                    images,
                    clean_cls=clean_cls,
                    return_distill=True,
                    current_labels=labels,
                )
            else:
                self.model_without_ddp.reset_distillation_observation()
                logits = self.model(
                    images,
                    clean_cls=clean_cls,
                    return_distill=False,
                )
                prepared_distillation = logits.new_zeros(())
            masked_logits = logits + logit_mask
            classification_loss = self.criterion(masked_logits, labels)

        model = self.model_without_ddp
        self.last_gradnorm = GradNormResult(
            valid=False,
            reason=("delay_gate" if not distillation_active else "not_gradnorm"),
        )
        if model.distill_weight_mode == "gradnorm" and distillation_active:
            reference_parameter = model.prompt_generator.mlp[-1].weight
            self.last_gradnorm = self._gradnorm_controller.compute(
                classification_loss,
                prepared_distillation,
                reference_parameter,
            )
            if self.last_gradnorm.valid:
                model.last_effective_distill_weight = self.last_gradnorm.weight
                detached_weight = prepared_distillation.new_tensor(
                    self.last_gradnorm.weight
                )
                distillation_objective = (
                    detached_weight * prepared_distillation
                )
            else:
                # Multiplying NaN by zero remains NaN, so use a fresh finite
                # scalar when the gradient check rejects this auxiliary step.
                model.last_effective_distill_weight = 0.0
                distillation_objective = classification_loss.new_zeros(())
        else:
            # fixed and uncertainty are prepared by the model. In uncertainty
            # mode this includes the legacy +s term and is the true objective.
            distillation_objective = prepared_distillation

        loss = classification_loss + distillation_objective

        _, predictions = masked_logits.topk(
            self.topk, dim=1, largest=True, sorted=True
        )
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        self.last_classification_loss = float(classification_loss.detach().item())
        self.last_distillation_loss = float(
            model.last_raw_distillation
        )
        self.last_distill_weight = float(
            model.last_effective_distill_weight
        )
        self.last_distillation_objective = float(
            distillation_objective.detach().item()
        )
        self.last_true_total_loss = float(loss.detach().item())
        if model.distill_weight_mode == "gradnorm" and not self.last_gradnorm.valid:
            self.last_weighted_distillation = 0.0
        else:
            self.last_weighted_distillation = (
                self.last_distill_weight * self.last_distillation_loss
            )
        anchors = int(model.last_replay_class_count)
        self._acc_steps += 1
        self._acc_ce += self.last_classification_loss
        self._acc_raw_distill += self.last_distillation_loss
        self._acc_weighted_distill += self.last_weighted_distillation
        self._acc_distill_objective += self.last_distillation_objective
        self._acc_true_total += self.last_true_total_loss
        self._acc_bad_objectives += int(
            not math.isfinite(self.last_true_total_loss)
            or self.last_true_total_loss < 0
        )
        self._acc_anchors += anchors
        if anchors > 0:
            self._acc_active_steps += 1
        if self.last_gradnorm.valid:
            self._acc_gradnorm_valid += 1
            self._acc_gradnorm_boundary_hits += int(
                self.last_gradnorm.boundary_hit
            )
            self._acc_gradnorm_ratio += self.last_gradnorm.actual_grad_ratio
            self._acc_gradnorm_in_band += int(
                0.125 <= self.last_gradnorm.actual_grad_ratio <= 0.5
            )
        correct = torch.sum(predictions == labels.unsqueeze(1)).item()
        accuracy = correct / labels.size(0)
        return float(loss.item()), accuracy

    def report_training(self, sample_num, train_loss, train_acc):
        super().report_training(sample_num, train_loss, train_acc)
        model = self.model_without_ddp
        steps = max(self._acc_steps, 1)
        logger.info(
            "Loss | ce %.6f | distill_raw %.6f | distill_w %.4f | "
            "distill_weighted %.6f | distill_objective %.6f | "
            "true_total %.6f | anchors %d | lags %s",
            self.last_classification_loss,
            self.last_distillation_loss,
            self.last_distill_weight,
            self.last_weighted_distillation,
            self.last_distillation_objective,
            self.last_true_total_loss,
            model.last_replay_class_count,
            model.last_active_teacher_lags,
        )
        logger.info(
            "Distill gate | batch_state=%s | delay_state=%s | "
            "samples_since_snapshot=%d | delay_samples=%d | snapshot=%d",
            self.last_distill_batch_state,
            self._distill_delay_gate.state,
            self._distill_delay_gate.samples_since_snapshot,
            self.distill_delay_samples,
            self._distill_delay_gate.snapshot_count,
        )
        if model.distill_weight_mode == "gradnorm":
            gradnorm = self.last_gradnorm
            logger.info(
                "GradNorm | valid=%s | reason=%s | g_ce=%.6g | g_d=%.6g | "
                "target_ratio=%.4f | actual_ratio=%.6g | "
                "grad_cosine=%.6g | target_weight=%.6g | weight=%.6g | "
                "boundary_hit=%s",
                gradnorm.valid,
                gradnorm.reason or "ok",
                gradnorm.classification_grad_norm,
                gradnorm.distillation_grad_norm,
                self.distill_grad_ratio,
                gradnorm.actual_grad_ratio,
                gradnorm.gradient_cosine,
                gradnorm.target_weight,
                gradnorm.weight,
                gradnorm.boundary_hit,
            )
        elif model.distill_weight_mode == "uncertainty":
            logger.info(
                "Uncertainty weight | s=%.6f | scale_exp(-s)=%.6g | "
                "effective_weight=%.6g | "
                "weighted_distill=%.6f | distill_objective=%.6f | "
                "true_total=%.6f",
                model.last_log_distill_scale,
                math.exp(-model.last_log_distill_scale),
                self.last_distill_weight,
                self.last_weighted_distillation,
                self.last_distillation_objective,
                self.last_true_total_loss,
            )
        logger.info(
            "Loss(mean over %d updates) | ce %.6f | distill_raw %.6f | "
            "distill_weighted %.6f | distill_objective %.6f | "
            "true_total %.6f | distill/ce %.4f | active_steps %d/%d | "
            "mean_anchors %.2f | gradnorm_valid %d | boundary_hits %d "
            "(%.2f%%) | mean_actual_ratio %.6g | ratio_in_[.125,.5] "
            "%d/%d (%.2f%%) | bad_objectives %d",
            self._acc_steps,
            self._acc_ce / steps,
            self._acc_raw_distill / steps,
            self._acc_weighted_distill / steps,
            self._acc_distill_objective / steps,
            self._acc_true_total / steps,
            self._acc_weighted_distill / max(self._acc_ce, 1e-12),
            self._acc_active_steps,
            self._acc_steps,
            self._acc_anchors / steps,
            self._acc_gradnorm_valid,
            self._acc_gradnorm_boundary_hits,
            100.0 * self._acc_gradnorm_boundary_hits
            / max(self._acc_gradnorm_valid, 1),
            self._acc_gradnorm_ratio / max(self._acc_gradnorm_valid, 1),
            self._acc_gradnorm_in_band,
            self._acc_gradnorm_valid,
            100.0 * self._acc_gradnorm_in_band
            / max(self._acc_gradnorm_valid, 1),
            self._acc_bad_objectives,
        )
