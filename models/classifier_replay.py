"""Class-conditional feature replay for a shared linear classifier.

Uses the same Welford / low-rank distribution bank as the MLP prompt generator.
Replay features are detached final prompted CLS vectors, so replay gradients
only regularise the online classifier, not the prompt adapter or frozen ViT.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.feature_statistics import ClassFeatureStatistics


class ClassifierReplay(nn.Module):
    def __init__(self, online_fc, task_num, **kwargs):
        super().__init__()
        self.weight_mode = kwargs.get("classifier_replay_weight_mode", "fixed")
        self.fixed_weight = float(kwargs.get("distill_weight", 1.0))
        if self.weight_mode not in {"fixed", "gradnorm", "learned_gradnorm"}:
            raise ValueError("Invalid classifier replay weight mode")
        if self.fixed_weight < 0 or not math.isfinite(self.fixed_weight):
            raise ValueError("Classifier distill_weight must be finite and non-negative")
        if self.weight_mode == "learned_gradnorm":
            ratio = float(kwargs.get("distill_grad_ratio", 0.25))
            if not 0 < ratio < 1:
                raise ValueError("Learned classifier gradient ratio must start strictly between 0 and 1")
            self.ratio_reg = float(kwargs.get("classifier_replay_ratio_reg", 1.0))
            if not math.isfinite(self.ratio_reg) or self.ratio_reg <= 0:
                raise ValueError("classifier_replay_ratio_reg must be finite and positive")
            self.register_buffer("ratio_prior", torch.tensor(ratio))
            self.ratio_logit = nn.Parameter(torch.tensor(math.log(ratio / (1 - ratio))))
        else:
            self.register_parameter("ratio_logit", None)
        self.teacher_lags = tuple(sorted(set(kwargs.get("teacher_lags", (1,)))))
        if not self.teacher_lags or self.teacher_lags[0] < 1 or self.teacher_lags[-1] >= task_num:
            raise ValueError("Classifier replay needs positive teacher_lags smaller than step_num")
        self.replay_class_budget = int(kwargs.get("replay_class_budget", 8))
        self.anchors_per_class = int(kwargs.get("anchors_per_class", 8))
        if self.replay_class_budget < len(self.teacher_lags) or self.anchors_per_class < 1:
            raise ValueError("Invalid classifier replay class / anchor budget")
        self.replay_eligibility = kwargs.get("replay_eligibility", "previous")
        self.distill_mode = kwargs.get("distill_mode", "sample")
        self.distill_metric = kwargs.get("distill_metric", "mse")
        if self.replay_eligibility not in {"teacher", "previous"}:
            raise ValueError("Invalid replay_eligibility")
        if self.distill_mode not in {"sample", "closed_form"}:
            raise ValueError("Invalid distill_mode")
        if self.distill_metric not in {"mse", "cosine"}:
            raise ValueError("Invalid classifier replay distill_metric")
        if self.distill_mode == "closed_form" and self.distill_metric != "mse":
            raise ValueError("Classifier closed_form replay supports raw-logit MSE only")
        # Classifier inputs already include the ViT's final LayerNorm. Apply no
        # extra input transform: the stored variable must be exactly FC's input.
        if kwargs.get("statistics_space", "raw") != "raw":
            raise ValueError("Classifier replay stores the actual FC input; use statistics_space raw")
        history_size = self.teacher_lags[-1]
        self.teachers = nn.ModuleList([copy.deepcopy(online_fc) for _ in range(history_size)])
        self.teachers.requires_grad_(False)
        self.register_buffer("teacher_valid", torch.zeros(history_size, dtype=torch.bool))
        self.register_buffer("teacher_class_count", torch.zeros(history_size, dtype=torch.long))
        self.register_buffer("seen_class_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("snapshot_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("replay_seed", torch.tensor((torch.initial_seed() + 7919) % (2**63 - 1)))
        self.register_buffer("draw_count", torch.zeros((), dtype=torch.long))
        self.statistics = ClassFeatureStatistics(
            num_classes=online_fc.out_features,
            feature_dim=online_fc.in_features,
            history_size=history_size + 1,
            statistic_type=kwargs.get("statistic_type", "diagonal"),
            covariance_rank=int(kwargs.get("covariance_rank", 16)),
            min_replay_samples=int(kwargs.get("min_replay_samples", 2)),
            variance_prior_strength=float(kwargs.get("variance_prior_strength", 16)),
            variance_floor=float(kwargs.get("variance_floor", 1e-5)),
            variance_ceiling=kwargs.get("variance_ceiling", 10.0),
            sketch_shrinkage=kwargs.get("sketch_shrinkage", "none"),
        )
        self.last_replay_count = 0
        self.last_teacher_lags = ()

    @property
    def enabled(self):
        return self.weight_mode != "fixed" or self.fixed_weight > 0

    @torch.no_grad()
    def update_statistics(self, features, labels):
        self.statistics.update(features.detach().float(), labels)
        if labels.numel():
            self.seen_class_count.copy_(torch.maximum(self.seen_class_count, labels.max() + 1))

    @torch.no_grad()
    def snapshot(self, online_fc):
        for slot in range(len(self.teachers) - 1, 0, -1):
            self.teachers[slot].load_state_dict(self.teachers[slot - 1].state_dict())
            self.teacher_valid[slot].copy_(self.teacher_valid[slot - 1])
            self.teacher_class_count[slot].copy_(self.teacher_class_count[slot - 1])
        self.teachers[0].load_state_dict(online_fc.state_dict())
        self.teacher_valid[0].fill_(True)
        self.teacher_class_count[0].copy_(self.seen_class_count)
        self.statistics.snapshot()
        self.snapshot_count.add_(1)

    def _closed_form(self, online_fc, slot, classes):
        count = int(self.teacher_class_count[slot])
        teacher = self.teachers[slot]
        delta_w = online_fc.weight[:count].float() - teacher.weight[:count].float()
        delta_b = online_fc.bias[:count].float() - teacher.bias[:count].float()
        means, factor, residual = self.statistics.distribution(slot, classes)
        value = (means.float() @ delta_w.t() + delta_b).square().sum(1)
        value = value + residual.float() @ delta_w.square().sum(0)
        if factor is not None:
            value = value + torch.einsum("crd,od->cro", factor.float(), delta_w).square().sum((1, 2))
        return (value / count).mean()

    def loss(self, online_fc):
        # Keep replay draws independent of future image augmentation RNG.
        device = online_fc.weight.device
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            seed = (int(self.replay_seed) + int(self.draw_count)) % (2**63 - 1)
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            result = self._loss(online_fc)
        self.draw_count.add_(1)
        return result

    def _loss(self, online_fc):
        self.last_replay_count = 0
        self.last_teacher_lags = ()
        active = []
        for lag in self.teacher_lags:
            slot = lag - 1
            eligible = slot + int(self.replay_eligibility == "previous")
            if bool(self.teacher_valid[slot]) and self.statistics.replay_available(eligible):
                active.append((lag, slot, eligible))
        # Same round-robin total class budget as the existing generator.
        capacities = [self.statistics.eligible_count(e) for _, _, e in active]
        budgets = [0] * len(active)
        remaining = self.replay_class_budget
        while remaining and any(b < c for b, c in zip(budgets, capacities)):
            for i, capacity in enumerate(capacities):
                if remaining and budgets[i] < capacity:
                    budgets[i] += 1
                    remaining -= 1
        losses = []
        used_lags = []
        with torch.autocast(device_type=online_fc.weight.device.type, enabled=False):
            for (lag, slot, eligible), budget in zip(active, budgets):
                if not budget:
                    continue
                if self.distill_mode == "closed_form":
                    classes = self.statistics.select_classes(eligible, budget)
                    loss = self._closed_form(online_fc, slot, classes)
                    size = classes.numel()
                else:
                    anchors, _ = self.statistics.sample_anchors(slot, budget, self.anchors_per_class, eligible)
                    count = int(self.teacher_class_count[slot])
                    student = F.linear(anchors.float(), online_fc.weight.float(), online_fc.bias.float())[:, :count]
                    with torch.no_grad():
                        teacher = self.teachers[slot](anchors.float())[:, :count]
                    loss = (F.mse_loss(student, teacher) if self.distill_metric == "mse" else
                            (1 - F.cosine_similarity(student, teacher, dim=1)).mean())
                    size = anchors.size(0)
                losses.append(loss * size)
                self.last_replay_count += size
                used_lags.append(lag)
        self.last_teacher_lags = tuple(used_lags)
        if not losses:
            return online_fc.weight.new_zeros(())
        return torch.stack(losses).sum() / self.last_replay_count
