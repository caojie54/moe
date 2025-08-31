"""
CoreLoRA Configuration
"""
from dataclasses import dataclass, field

from ..lora import LoraConfig
from ..utils.peft_types import PeftType


@dataclass
class MoCoreLoraConfig(LoraConfig):
    """
    MoCoreLORA Configuration
    """
    num_experts: int = field(default=4, metadata={"help": "The number of experts in MoE."})

    def __post_init__(self):
        self.peft_type = PeftType.MoCoreLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
