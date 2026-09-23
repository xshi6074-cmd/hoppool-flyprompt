from .codaprompt import CodaPrompt
from .dualprompt import DualPrompt
from .flyprompt import FlyPrompt
from .frozen_vit import FrozenViTTrainer
from .hfpool import HFPool
from .mlp_generator import MLPGenerator
from .l2p import L2P
from .mvp import MVP
from .ranpac import RanPAC
from .slca import SLCA
from .hide_norga_trainer import HiDeGCLTrainer, NoRGaGCLTrainer
from .sdlora import SDLoRAGCL
from .sprompt import SPrompt as SPromptTrainer

METHODS = {
    "codaprompt": CodaPrompt,
    "dualprompt": DualPrompt,
    "flyprompt": FlyPrompt,
    "baseline": FrozenViTTrainer,
    "shared_prompt": FrozenViTTrainer,
    "gate": FrozenViTTrainer,
    "hfpool": HFPool,
    "mlp_generator": MLPGenerator,
    "l2p": L2P,
    "mvp": MVP,
    "ranpac": RanPAC,
    "slca": SLCA,
    "hide": HiDeGCLTrainer,
    "hide_lora": HiDeGCLTrainer,
    "hide_adapter": HiDeGCLTrainer,
    "norga": NoRGaGCLTrainer,
    "sdlora": SDLoRAGCL,
    "sprompt": SPromptTrainer,
}
