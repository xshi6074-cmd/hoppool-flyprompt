import gc
import logging
import os

import torch
import torch.distributed as dist

from methods._trainer import _Trainer
from utils.prompt_probe import PromptProbeRecorder
from utils.distill_gradnorm import DistillationDelayGate, GradNormLiteController

logger = logging.getLogger()

FROZEN_METHODS = {"baseline", "shared_prompt", "gate", "hfpool", "hfpool_lora", "spatial_gelu"}
PROMPT_METHODS = {"shared_prompt", "hfpool", "hfpool_lora", "spatial_gelu"}


class FrozenViTTrainer(_Trainer):
    """Online trainer for frozen-ViT models with a trainable FC head.

    Used directly by ``baseline``, ``shared_prompt`` and ``gate``. Evaluation
    uses the online FC, or the online + EMA ensemble when the model has
    ``use_ema``. Subclasses customise the hooks:

    - ``prepare_online_batch``: work shared by all ``online_iter`` updates.
    - ``clip_gradients``: called between backward and the optimizer step.
    - ``after_online_updates``: called once per stream batch after training.
    - ``evaluation_forward_kwargs``: extra model inputs at evaluation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offline_probe_enabled = bool(getattr(self, "offline_probe", False))
        self.cls_probe_enabled = bool(getattr(self, "cls_probe", False))
        if self.cls_probe_enabled:
            if self.method not in FROZEN_METHODS or self.distributed:
                raise ValueError("CLS probing requires a frozen-ViT classifier and one GPU per job")
            if int(getattr(self, "cls_probe_per_class", 16)) < 2:
                raise ValueError("CLS probing needs at least two examples per class")
        if getattr(self, "ema_diagnostics", False) and (self.distributed or self.topk != 1):
            raise ValueError("EMA diagnostics requires one GPU and topk=1")
        if self.offline_probe_enabled:
            if self.method not in FROZEN_METHODS or self.distributed or self.topk != 1:
                raise ValueError("Offline probing requires a frozen-ViT classifier, one GPU per job and topk=1")
            from utils.offline_probe import ProbeSettings, ObservedSamples
            self.offline_probe_settings = ProbeSettings.from_arguments(self.kwargs)
            self.offline_probe_observed = ObservedSamples()
        elif self.cls_probe_enabled:
            from utils.offline_probe import ObservedSamples
            self.offline_probe_observed = ObservedSamples()
        self.task_id = 0
        self.prompt_probe_every_batches = int(getattr(self, "prompt_probe_every_batches", 0))
        self.prompt_probe_examples = int(getattr(self, "prompt_probe_examples", 8))
        if self.prompt_probe_every_batches < 0:
            raise ValueError("prompt_probe_every_batches cannot be negative")
        if self.prompt_probe_every_batches and self.method not in PROMPT_METHODS:
            raise ValueError("Prompt probing requires a method with capture_prompts")
        self.prompt_probe_batch_count = 0
        self.prompt_probe_seen_samples = 0
        self.prompt_probe_recorder = None
        self.classifier_replay_enabled = bool(getattr(self, "classifier_replay", False))
        if self.classifier_replay_enabled and self.method not in FROZEN_METHODS:
            raise ValueError("Classifier replay requires a frozen-ViT classifier")
        self._classifier_replay_gate = DistillationDelayGate(int(getattr(self, "distill_delay_samples", 100)))
        self._classifier_replay_controller = None
        if self.classifier_replay_enabled:
            mode = getattr(self, "classifier_replay_weight_mode", "fixed")
            if mode != "fixed":
                if self.distributed:
                    raise ValueError("Classifier gradient-norm weighting currently supports one GPU per job")
                self._classifier_replay_controller = GradNormLiteController(
                    target_ratio=float(getattr(self, "distill_grad_ratio", 0.25)),
                    log_ema=float(getattr(self, "distill_grad_ema", 0.9)),
                    weight_min=float(getattr(self, "distill_weight_min", 1e-4)),
                    weight_max=float(getattr(self, "distill_weight_max", 1e4)),
                )
        self._replay_acc_steps = self._replay_acc_active = self._replay_acc_invalid = 0
        self._replay_acc_ce = self._replay_acc_raw = self._replay_acc_weighted = self._replay_acc_balance = 0.0

    def _init_internal_step_scheduler(self):
        super()._init_internal_step_scheduler()
        if not self.classifier_replay_enabled or not hasattr(self, "total_samples"):
            return
        # Match the MLP generator's sample-only windows over ALL epoch visits.
        sampler = getattr(self, "train_sampler", None)
        indices = getattr(sampler, "indices", None)
        if indices is None:
            visits = int(self.total_samples)
        elif getattr(sampler, "distributed", False):
            replicas = int(sampler.num_replicas)
            visits = sum((len(items) // replicas) * replicas for items in indices)
        else:
            visits = sum(len(items) for items in indices)
        self.samples_per_step = max(1, visits * int(self.num_epochs) // self.step_num)

    def online_step(self, images, labels, idx):
        if getattr(self, "offline_probe_enabled", False) or getattr(self, "cls_probe_enabled", False):
            self.offline_probe_observed.add(idx, labels)
        self.add_new_class(labels)
        _loss, _acc, _iter = 0.0, 0.0, 0

        prepared = self.prepare_online_batch(images)
        replay = getattr(self.model_without_ddp, "classifier_replay", None)
        replay_enabled = replay is not None and replay.enabled
        replay_active = replay_enabled and self._classifier_replay_gate.active
        for update_index in range(int(self.online_iter)):
            replay_kwargs = (dict(record_replay_statistics=update_index == 0,
                                  replay_active=replay_active) if replay_enabled else {})
            if prepared is None:
                loss, acc = self.online_train([images.clone(), labels.clone()], **replay_kwargs)
            else:
                train_images, forward_kwargs = prepared
                loss, acc = self.online_train(
                    [train_images, labels.clone()],
                    inputs_transformed=True,
                    forward_kwargs=forward_kwargs,
                    **replay_kwargs,
                )
            _loss += loss
            _acc += acc
            _iter += 1

        self.after_online_updates(images, labels)

        # Advance the internal step schedule from seen samples only
        # (task-boundary-free).
        self._maybe_advance_internal_step(images.size(0) * self.world_size)
        if replay_enabled:
            if self._classifier_replay_gate.finish_batch(int(replay.snapshot_count), images.size(0) * self.world_size):
                if self._classifier_replay_controller is not None:
                    self._classifier_replay_controller.reset()
                logger.info("Classifier replay snapshot | snapshot=%d | step=%d | eligible=%d",
                            int(replay.snapshot_count), self.current_step,
                            replay.statistics.eligible_count(0))
        self._maybe_record_prompt_probe(images.size(0) * self.world_size)

        del images, labels
        gc.collect()
        return _loss / _iter, _acc / _iter

    def prepare_online_batch(self, images):
        """Return ``(transformed_images, forward_kwargs)`` to reuse, or None."""
        return None

    def after_online_updates(self, images, labels):
        pass

    def clip_gradients(self):
        pass

    def evaluation_forward_kwargs(self, x):
        return {}

    def _maybe_record_prompt_probe(self, batch_size):
        interval = getattr(self, "prompt_probe_every_batches", 0)
        if not interval:
            return
        self.prompt_probe_batch_count += 1
        self.prompt_probe_seen_samples += batch_size
        if self.prompt_probe_batch_count % interval or not self.is_main_process():
            return
        if self.prompt_probe_recorder is None:
            self.prompt_probe_recorder = PromptProbeRecorder(
                self.test_dataset, self.prompt_probe_examples, self.method
            )
        was_training = self.model_without_ddp.training
        self.model_without_ddp.eval()
        try:
            # Probe in FP32 so cosine/drift is not quantized by training AMP.
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                self.prompt_probe_recorder.record(
                    self.model_without_ddp, self.device, self.prompt_probe_seen_samples
                )
        finally:
            self.model_without_ddp.train(was_training)

    @staticmethod
    @torch.no_grad()
    def _gather_replay_statistics(features, labels):
        if not dist.is_available() or not dist.is_initialized():
            return features, labels
        world = dist.get_world_size()
        size = labels.new_tensor([labels.numel()])
        sizes = [torch.zeros_like(size) for _ in range(world)]
        dist.all_gather(sizes, size)
        maximum = max(int(item) for item in sizes)
        padding = maximum - labels.numel()
        features = torch.cat((features, features.new_zeros(padding, features.size(1))))
        labels = torch.cat((labels, labels.new_zeros(padding)))
        all_features = [torch.empty_like(features) for _ in range(world)]
        all_labels = [torch.empty_like(labels) for _ in range(world)]
        dist.all_gather(all_features, features.contiguous())
        dist.all_gather(all_labels, labels.contiguous())
        return (torch.cat([item[:int(size)] for item, size in zip(all_features, sizes)]),
                torch.cat([item[:int(size)] for item, size in zip(all_labels, sizes)]))

    def _classifier_replay_objective(self, ce, raw):
        replay = self.model_without_ddp.classifier_replay
        balance = ce.new_zeros(())
        ratio = 0.0
        reason = "fixed"
        self.last_classifier_replay_norms = None
        if replay.last_replay_count == 0 or not self._classifier_replay_gate.active:
            return ce.new_zeros(()), balance, 0.0, ratio, "no_replay"
        if replay.weight_mode == "fixed":
            weight = replay.fixed_weight
        else:
            controller = self._classifier_replay_controller
            norms = controller.compute(ce, raw, self.model_without_ddp.backbone.fc.weight)
            self.last_classifier_replay_norms = norms
            reason = norms.reason
            if not norms.valid:
                return ce.new_zeros(()), balance, 0.0, ratio, reason
            if replay.weight_mode == "gradnorm":
                weight = norms.weight
                weighted = raw * weight
            else:
                # Learn rho (initially .25), not the detached normalisation
                # factor. A KL prior prevents the trivial rho -> 0 solution.
                # No higher-order gradients pass through the gradient norms.
                rho = replay.ratio_logit.sigmoid()
                normalisation = norms.weight / controller.target_ratio
                coefficient = (rho * normalisation).clamp(controller.weight_min, controller.weight_max)
                weighted = raw * coefficient
                prior = replay.ratio_prior
                balance = replay.ratio_reg * (
                    prior * (prior.log() - torch.nn.functional.logsigmoid(replay.ratio_logit))
                    + (1 - prior) * ((1 - prior).log() - torch.nn.functional.logsigmoid(-replay.ratio_logit)))
                weight = float(coefficient.detach())
            ratio = weight * norms.distillation_grad_norm / norms.classification_grad_norm
            return weighted, balance, weight, ratio, reason
        return raw * weight, balance, weight, ratio, reason

    def online_train(self, data, inputs_transformed=False, forward_kwargs=None,
                     record_replay_statistics=False, replay_active=False):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = x.to(self.device)
        y = y.to(self.device)

        if not inputs_transformed:
            x = self.train_transform(x)

        self.optimizer.zero_grad()
        replay = getattr(self.model_without_ddp, "classifier_replay", None)
        if replay is not None and replay.enabled:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logit, features, raw = self.model(x, return_replay=True, replay_active=replay_active,
                                                  **(forward_kwargs or {}))
                logit = logit + (self.mask if self.no_batchmask else logit_mask)
                ce = self.criterion(logit, y)
            if record_replay_statistics:
                features, global_labels = self._gather_replay_statistics(features.float(), y)
                replay.update_statistics(features, global_labels)
            weighted, balance, weight, ratio, reason = self._classifier_replay_objective(ce, raw)
            loss = ce + weighted + balance
            self._replay_acc_steps += 1
            self._replay_acc_active += int(replay_active and replay.last_replay_count > 0)
            self._replay_acc_invalid += int(reason not in {"fixed", "", "no_replay"})
            self._replay_acc_ce += float(ce.detach())
            self._replay_acc_raw += float(raw.detach())
            self._replay_acc_weighted += float(weighted.detach())
            self._replay_acc_balance += float(balance.detach())
            self.last_classifier_replay = dict(ce=float(ce.detach()), raw=float(raw.detach()),
                weighted=float(weighted.detach()), balance=float(balance.detach()),
                true_total=float(loss.detach()),
                weight=weight, grad_ratio=ratio, reason=reason,
                target_ratio=(float(replay.ratio_logit.detach().sigmoid())
                              if replay.ratio_logit is not None else
                              float(getattr(self, "distill_grad_ratio", .25))),
                replay_count=replay.last_replay_count if replay_active else 0,
                teacher_lags=replay.last_teacher_lags if replay_active else ())
            norms = self.last_classifier_replay_norms
            if norms is not None:
                self.last_classifier_replay.update(
                    ce_grad_norm=norms.classification_grad_norm,
                    distill_grad_norm=norms.distillation_grad_norm,
                    gradient_cosine=norms.gradient_cosine,
                    weight_boundary_hit=norms.boundary_hit)
        elif not self.no_batchmask:
            logit, loss = self.model_forward(x, y, mask=logit_mask, forward_kwargs=forward_kwargs)
        else:
            logit, loss = self.model_forward(x, y, forward_kwargs=forward_kwargs)

        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.clip_gradients()
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        step_accepted = self.scaler.get_scale() >= scale_before
        if step_accepted:
            self.update_schedule()

        if step_accepted and getattr(self.model_without_ddp, "has_ema", False):
            self.model_without_ddp.update_ema_fc()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct / total_num_data

    def report_training(self, sample_num, train_loss, train_acc):
        super().report_training(sample_num, train_loss, train_acc)
        if not self.classifier_replay_enabled:
            return
        replay = self.model_without_ddp.classifier_replay
        if not replay.enabled:
            return
        info = getattr(self, "last_classifier_replay", {})
        logger.info("Classifier replay | samples=%d | statistics=%s | shrinkage=%s | mode=%s | "
                    "weight_mode=%s | state=%s | snapshot=%d | telemetry=%s",
                    sample_num, replay.statistics.statistic_type, replay.statistics.sketch_shrinkage,
                    replay.distill_mode, replay.weight_mode, self._classifier_replay_gate.state,
                    int(replay.snapshot_count), info)

    def model_forward(self, x, y, mask=None, forward_kwargs=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x, **(forward_kwargs or {}))
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)

        return logit, loss

    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        head_accumulators = {}

        model = self.model_without_ddp
        model.update()

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                forward_kwargs = self.evaluation_forward_kwargs(x)
                probabilities = False
                if getattr(model, "has_ema", False):
                    logit_ls = model.forward_with_ema(x, **forward_kwargs)
                    logit_ls = [logit + self.mask for logit in logit_ls]
                    if model.use_ema:
                        logit = self._ensemble_logits(logit_ls)
                        probabilities = "softmax" in self.ensemble_method
                    else:
                        logit = logit_ls[0]
                    if getattr(model, "ema_diagnostics", False) and end:
                        from utils.ema_diagnostics import diagnostic_scores, probability_loss
                        for name, (scores, is_probability) in diagnostic_scores(logit_ls, model.ema_ratio).items():
                            item = head_accumulators.setdefault(name, dict(correct=0, count=0, loss=0.,
                                cls_correct=torch.zeros(self.n_classes), cls_count=torch.zeros(self.n_classes)))
                            prediction = scores.argmax(dim=-1)
                            count, correct = self._interpret_pred(y, prediction)
                            head_loss = (probability_loss(scores, y) if is_probability else
                                         torch.nn.functional.cross_entropy(scores, y))
                            item["correct"] += int((prediction == y).sum())
                            item["count"] += len(y)
                            item["loss"] += float(head_loss) * len(y)
                            item["cls_correct"] += correct.cpu()
                            item["cls_count"] += count.cpu()
                else:
                    logit = model(x, **forward_kwargs) + self.mask

                if probabilities:
                    from utils.ema_diagnostics import probability_loss
                    loss = probability_loss(logit, y)
                else:
                    loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        if head_accumulators:
            from utils.ema_diagnostics import save_head_metrics
            save_head_metrics(os.path.join(self.log_dir, f"seed_{self.rnd_seed}_ema.jsonl"),
                              task_id, self.exposed_classes, head_accumulators)
        return eval_dict

    def _ensemble_logits(self, logit_ls):
        if not hasattr(self, 'ensemble_method'):
            self.ensemble_method = "softmax_max_prob"

        if "softmax" in self.ensemble_method:
            logit_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]

        logit_stack = torch.stack(logit_ls, dim=-1)  # Shape: [batch_size, n_classes, n_heads]

        if "mean" in self.ensemble_method:
            return logit_stack.mean(dim=-1)
        elif "max_prob" in self.ensemble_method:
            return logit_stack.max(dim=-1)[0]
        elif "min_entropy" in self.ensemble_method:
            entropies = -torch.sum(logit_stack * torch.log(logit_stack + 1e-8), dim=1)  # [batch_size, n_heads]
            min_entropy_indices = torch.argmin(entropies, dim=-1)  # [batch_size]
            batch_indices = torch.arange(logit_stack.size(0), device=logit_stack.device)
            return logit_stack[batch_indices, :, min_entropy_indices]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def online_before_task(self, task_id):
        pass

    def save_offline_probe_checkpoint(self, task_id, samples_cnt, eval_dict):
        if not (getattr(self, "offline_probe_enabled", False) or getattr(self, "cls_probe_enabled", False)):
            return
        from pathlib import Path
        from utils.offline_probe import save_snapshot
        for enabled, suffix in ((getattr(self, "offline_probe_enabled", False), "lp"),
                                (getattr(self, "cls_probe_enabled", False), "cls")):
            if enabled:
                directory = Path(self.log_dir) / f"seed_{self.rnd_seed}_{suffix}"
                save_snapshot(directory, self.model_without_ddp, self.train_dataset,
                              self.offline_probe_observed, self.exposed_classes,
                              task_id, samples_cnt, eval_dict, self.kwargs,
                              getattr(self, "ensemble_method", "softmax_max_prob"))

    def run_offline_probe(self):
        if getattr(self.model_without_ddp, "ema_diagnostics", False):
            from utils.ema_diagnostics import summarize_heads
            summarize_heads(os.path.join(self.log_dir, f"seed_{self.rnd_seed}_ema.jsonl"))
        if getattr(self, "cls_probe_enabled", False):
            from utils.cls_probe import run_cls_probe
            run_cls_probe(self.model_without_ddp, self.train_dataset, self.test_transform_tensor,
                          os.path.join(self.log_dir, f"seed_{self.rnd_seed}_cls"), self.device,
                          self.rnd_seed, int(getattr(self, "cls_probe_per_class", 16)), self.batchsize)
        if not getattr(self, "offline_probe_enabled", False):
            return
        from pathlib import Path
        from utils.offline_probe import run_probes
        directory = Path(self.log_dir) / f"seed_{self.rnd_seed}_lp"
        logger.info("[Post] Refitting offline linear heads from %s", directory)
        run_probes(self.model_without_ddp, self.train_dataset, self.test_dataset,
                   self.test_transform_tensor, directory, self.device,
                   self.offline_probe_settings, seed=self.rnd_seed,
                   cache_features=bool(getattr(self, "offline_probe_cache_features", False)))

    def online_after_task(self, cur_iter):
        """Hook called after each benchmark task.

        ``task_id`` is kept for logging/analysis only; the model's internal
        step state advances exclusively via ``_maybe_advance_internal_step``.
        """
        if self.prompt_probe_recorder is not None and self.is_main_process():
            self.prompt_probe_recorder.save(os.path.join(
                self.log_dir, f"seed_{self.rnd_seed}_prompt_probe.npz"
            ))
        if self.classifier_replay_enabled:
            count = max(1, self._replay_acc_steps)
            logger.info("Classifier replay summary | steps=%d | active=%d | invalid_norms=%d | "
                        "mean_ce=%g | mean_raw=%g | mean_weighted=%g | mean_balance=%g",
                        self._replay_acc_steps, self._replay_acc_active, self._replay_acc_invalid,
                        self._replay_acc_ce/count, self._replay_acc_raw/count,
                        self._replay_acc_weighted/count, self._replay_acc_balance/count)
        self.task_id += 1
