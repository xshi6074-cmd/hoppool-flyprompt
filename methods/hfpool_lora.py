"""Online output-LoRA training, including complete-batch merge/reset schedules."""
from methods.hfpool import HFPool


class HFPoolLoRA(HFPool):
    """Use HFPool's shared optimizer/merge hooks with the hfpool_lora model."""
