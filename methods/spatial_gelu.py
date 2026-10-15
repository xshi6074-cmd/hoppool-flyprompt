"""Online training for spatial GELU prompt generation."""
from methods.hfpool import HFPool


class SpatialGELU(HFPool):
    """Use the same complete-batch output update protocol as HFPool/LoRA."""
