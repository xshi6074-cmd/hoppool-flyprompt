import logging

import torch
import torch.distributed as dist

from methods.frozen_vit import FrozenViTTrainer
from utils.train_utils import select_scheduler

logger = logging.getLogger()


class HFPool(FrozenViTTrainer):
    """Trainer for Hopfield-pooling prompts.

    ``--hopfield_grad_clip`` clips the trainable Hopfield gradients after AMP
    unscaling; it is the stability control for jointly trained parts, which
    otherwise collapsed (Q+K+O). With ``--hopfield_input_mode full_vit`` one
    augmentation and one clean ViT pass are cached per stream batch and reused
    by every online update; the prompted suffix is still recomputed each time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hopfield_grad_clip = bool(getattr(self, "hopfield_grad_clip", False))
        self.hopfield_grad_clip_norm = float(getattr(self, "hopfield_grad_clip_norm", 1.0))
        if not self.hopfield_grad_clip_norm > 0:
            raise ValueError(
                "hopfield_grad_clip_norm must be positive, got "
                f"{self.hopfield_grad_clip_norm}"
            )
        if self.hopfield_grad_clip:
            logger.info(
                "Hopfield gradient clipping enabled: max_norm=%s",
                self.hopfield_grad_clip_norm,
            )
        self.hopfield_lora_lr = float(getattr(self, "hopfield_lora_lr", 0.005))
        if self.hopfield_lora_lr <= 0:
            raise ValueError("hopfield_lora_lr must be positive")
        self.lora_seen_samples = 0
        self.lora_samples_since_merge = 0
        self.full_o_seen_samples = 0

    def setup_distributed_model(self):
        super().setup_distributed_model()
        if self.model_without_ddp.adapter.o_mode != "full":
            self._configure_lora_optimizer()

    def _configure_lora_optimizer(self):
        if self.opt_name != "adam":
            raise ValueError("HFPool LoRA modes currently require --opt_name adam")
        lora_params = [
            parameter
            for lora in self.model_without_ddp.adapter.lora_output_weights()
            for parameter in (lora.lora_A, lora.lora_B)
        ]
        lora_ids = {id(parameter) for parameter in lora_params}
        other_params = [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in lora_ids
        ]
        self.optimizer = torch.optim.Adam([
            {"params": other_params, "lr": self.lr},
            {"params": lora_params, "lr": self.hopfield_lora_lr,
             "_hopfield_lora_group": True},
        ], weight_decay=0)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
        logger.info(
            "HFPool LoRA optimizer | lr=%g | lora_lr=%g | rank=%d | alpha=%g | FP32",
            self.lr, self.hopfield_lora_lr,
            self.model_without_ddp.adapter.lora_rank,
            self.model_without_ddp.adapter.lora_alpha,
        )

    def reset_opt(self):
        if self.model_without_ddp.adapter.o_mode == "full":
            return super().reset_opt()
        self._configure_lora_optimizer()

    def update_schedule(self, reset=False):
        if reset and self.model_without_ddp.adapter.o_mode != "full":
            for group in self.optimizer.param_groups:
                group["lr"] = (self.hopfield_lora_lr if group.get("_hopfield_lora_group")
                               else self.lr)
                group.pop("initial_lr", None)
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            return
        super().update_schedule(reset=reset)

    @torch.no_grad()
    def _effective_output_weights(self):
        return [
            projection.weight.detach().clone()
            for projection in self.model_without_ddp.adapter.output_projections()
        ]

    def online_step(self, images, labels, idx):
        mode = self.model_without_ddp.adapter.o_mode
        if mode == "full" and self.model_without_ddp.adapter.trainable != {"o"}:
            return super().online_step(images, labels, idx)
        before = self._effective_output_weights()
        batch_size = images.size(0) * self.world_size
        result = super().online_step(images, labels, idx)
        if mode == "full":
            self.full_o_seen_samples += batch_size
        after = self._effective_output_weights()
        update_sq = sum(
            torch.sum((new - old).square()).item()
            for old, new in zip(before, after)
        )
        base_sq = sum(torch.sum(old.square()).item() for old in before)
        logger.info(
            "HFPool output update | mode=%s | seen=%d | batch=%d | "
            "relative_delta_w=%.8g",
            mode, (self.full_o_seen_samples if mode == "full"
                   else self.lora_seen_samples), batch_size,
            (update_sq / max(base_sq, 1e-30)) ** 0.5,
        )
        return result

    def after_online_updates(self, images, labels):
        mode = self.model_without_ddp.adapter.o_mode
        if mode == "full":
            return
        batch_size = images.size(0) * self.world_size
        self.lora_seen_samples += batch_size
        self.lora_samples_since_merge += batch_size
        if mode == "lora_merge_batch" or (
                mode == "lora_merge_1024" and self.lora_samples_since_merge >= 1024):
            self._merge_lora()

    def online_after_task(self, cur_iter):
        if (self.model_without_ddp.adapter.o_mode == "lora_merge_1024"
                and self.task_id == self.n_tasks - 1
                and self.lora_samples_since_merge):
            self._merge_lora()
        super().online_after_task(cur_iter)

    @torch.no_grad()
    def _merge_lora(self):
        adapter = self.model_without_ddp.adapter
        stats = adapter.merge_lora_output_weights()
        for lora in adapter.lora_output_weights():
            for parameter in (lora.lora_A, lora.lora_B):
                self.optimizer.state.pop(parameter, None)
                parameter.grad = None
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(parameter, src=0)
        delta_sq = sum(delta_norm ** 2 for delta_norm, _ in stats)
        base_sq = sum(base_norm ** 2 for _, base_norm in stats)
        logger.info(
            "HFPool LoRA merge | mode=%s | seen=%d | interval=%d | "
            "relative_merge=%.8g | layer_relative_merge=%s",
            adapter.o_mode, self.lora_seen_samples, self.lora_samples_since_merge,
            (delta_sq / max(base_sq, 1e-30)) ** 0.5,
            [round(delta / max(base, 1e-15), 8) for delta, base in stats],
        )
        self.lora_samples_since_merge = 0

    def prepare_online_batch(self, images):
        model = self.model_without_ddp
        if not model.uses_full_vit_input:
            return None
        if self.online_iter != 3:
            raise ValueError(
                "hopfield_input_mode='full_vit' requires --online_iter 3 "
                f"for this ablation, got {self.online_iter}"
            )
        self.model.train()
        train_images = self.train_transform(images.clone().to(self.device))
        context = model.prepare_hopfield_context(train_images)
        return train_images, {"hopfield_context": context}

    def clip_gradients(self):
        if not self.hopfield_grad_clip:
            return
        # AMP scales gradients before backward. Unscale exactly once before
        # clipping so max_norm is expressed in the gradients' true units.
        self.scaler.unscale_(self.optimizer)
        parameters = [
            parameter
            for parameter in self.model_without_ddp.adapter.parameters()
            if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.hopfield_grad_clip_norm)
