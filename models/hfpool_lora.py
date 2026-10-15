"""Hopfield prompt generation with low-rank output updates."""
from models.hfpool import HFPool


class HFPoolLoRA(HFPool):
    """Output-only HFPool; merge after each complete stream batch by default."""

    def build_adapter(self, hopfield_trainable=None, hopfield_o_mode=None,
                      hopfield_pooling_mode=None, **kwargs):
        trainable = ["o"] if hopfield_trainable is None else hopfield_trainable
        mode = hopfield_o_mode or "lora_merge_batch"
        if set(trainable) != {"o"} or mode == "full":
            raise ValueError("hfpool_lora requires output-only low-rank updates")
        if hopfield_pooling_mode not in (None, "attention"):
            raise ValueError("hfpool_lora uses attention pooling; use spatial_gelu for GELU pooling")
        return super().build_adapter(hopfield_trainable=trainable, hopfield_o_mode=mode,
                                     hopfield_pooling_mode="attention", **kwargs)
