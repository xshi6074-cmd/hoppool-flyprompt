import logging

import torch

from methods.frozen_vit import FrozenViTTrainer

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
