import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import create_backbone


logger = logging.getLogger()


class FrozenViTClassifier(nn.Module):
    """Frozen pretrained ViT with a trainable online FC head.

    Subclasses return a feature adapter from ``build_adapter``. The adapter is
    called as ``adapter(backbone, inputs, **kwargs)`` and returns the final CLS
    feature; without one the model is the frozen ViT + FC baseline. The
    optional EMA bank is one global bank shared by the whole stream.
    """

    def __init__(self,
                 task_num: int = 10,
                 num_classes: int = 100,
                 backbone_name: Optional[str] = None,
                 use_ema: bool = False,
                 ema_ratio: Iterable[float] = (0.9, 0.99),
                 **kwargs):
        super().__init__()
        if backbone_name is None:
            raise ValueError("backbone_name must be specified")

        self.task_num = task_num
        self.num_classes = num_classes
        self.task_count = 0
        self.use_ema = bool(use_ema)
        self.ema_ratio = [float(ratio) for ratio in ema_ratio]
        if self.use_ema and not self.ema_ratio:
            raise ValueError("ema_ratio must not be empty when use_ema=True")

        self.backbone = create_backbone(
            backbone_name,
            num_classes=num_classes,
            pretrained=kwargs.get("pretrained", True),
            backbone_path=kwargs.get("backbone_path"),
        )
        self.embed_dim = self.backbone.num_features
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad = True

        # Built before the EMA heads so parameter initialisation consumes the
        # random stream in the same order as the former FlyPrompt flags.
        self.adapter = self.build_adapter(**kwargs)

        self.ema_fc = None
        if self.use_ema:
            self.ema_fc = nn.ModuleList([
                nn.Linear(self.embed_dim, num_classes, bias=True)
                for _ in self.ema_ratio
            ])
            for param in self.ema_fc.parameters():
                param.requires_grad = False
            self._copy_online_fc_to_ema()

    def build_adapter(self, **kwargs) -> Optional[nn.Module]:
        return None

    def extract_features(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.adapter is None:
            return self.backbone.forward_features(inputs)[:, 0]
        return self.adapter(self.backbone, inputs, **kwargs)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.backbone.fc(self.extract_features(inputs, **kwargs))

    def forward_with_ema(self, inputs: torch.Tensor, **kwargs):
        features = self.extract_features(inputs, **kwargs)
        outputs = [self.backbone.fc(features)]
        if self.use_ema:
            outputs.extend(fc(features) for fc in self.ema_fc)
        return outputs

    @torch.no_grad()
    def _copy_online_fc_to_ema(self):
        for fc in self.ema_fc:
            fc.weight.copy_(self.backbone.fc.weight)
            fc.bias.copy_(self.backbone.fc.bias)

    @torch.no_grad()
    def update_ema_fc(self):
        if not self.use_ema:
            return
        online_fc = self.backbone.fc
        for fc, ratio in zip(self.ema_fc, self.ema_ratio):
            fc.weight.mul_(ratio).add_(online_fc.weight, alpha=1.0 - ratio)
            fc.bias.mul_(ratio).add_(online_fc.bias, alpha=1.0 - ratio)

    def process_task_count(self):
        self.task_count += 1

    def update(self):
        """No routing head needs refreshing before evaluation."""

    def loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(output, target)


class Baseline(FrozenViTClassifier):
    """Frozen ViT + shared online FC: no prompt, gate or routing."""
