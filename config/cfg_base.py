from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

@dataclass()
class Config:
    # Reproducibility
    seed: int = 0

    # Model checkpoint
    model_ckpt_path: str = "checkpoints/unet.pt"
    resume_ckpt_path: Optional[str] = None


