"""Spatially pooled GELU features with a learned prompt output mapping."""
from models.hfpool import HFPool


class SpatialGELU(HFPool):
    """Keep CLS plus a 3x3 patch summary; train its output mapping, optionally via LoRA."""

    def build_adapter(self, hopfield_trainable=None, hopfield_o_mode=None,
                      hopfield_pooling_mode=None, **kwargs):
        trainable = ["o"] if hopfield_trainable is None else hopfield_trainable
        if set(trainable) != {"o"}:
            raise ValueError("spatial_gelu trains the output mapping only")
        if hopfield_pooling_mode not in (None, "spatial_gelu"):
            raise ValueError("spatial_gelu requires spatial GELU pooling")
        return super().build_adapter(hopfield_trainable=trainable,
                                     hopfield_o_mode=hopfield_o_mode or "full",
                                     hopfield_pooling_mode="spatial_gelu", **kwargs)
