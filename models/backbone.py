import logging
from pathlib import Path

import timm

from . import vit as custom_vit


logger = logging.getLogger()


def create_backbone(
    backbone_name,
    *,
    deep_prompts=False,
    num_pooling_heads=2,
    num_pooling_blocks=5,
    prompt_length=5,
    pretrained=True,
    num_classes=None,
    backbone_path=None,
    **model_kwargs,
):
    """Create a backbone, optionally loading weights from an explicit file.

    The Hopfield-related arguments remain accepted for call-site compatibility,
    but Hopfield is now attached by FlyPrompt rather than built into the ViT.
    """
    checkpoint_path = None
    if backbone_path:
        path = Path(backbone_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Backbone checkpoint not found: {path}")
        checkpoint_path = str(path.resolve())
        logger.info("Using explicit backbone checkpoint: %s", checkpoint_path)

    if hasattr(custom_vit, backbone_name):
        if checkpoint_path is not None:
            model_kwargs["checkpoint_path"] = checkpoint_path
        return getattr(custom_vit, backbone_name)(
            pretrained=pretrained,
            num_classes=num_classes,
            **model_kwargs,
        )

    timm_kwargs = dict(model_kwargs)
    if checkpoint_path is not None:
        # Avoid downloading timm's default pretrained weights before loading
        # the explicitly supplied local checkpoint.
        timm_kwargs["checkpoint_path"] = checkpoint_path
        pretrained = False
    return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        **timm_kwargs,
    )
