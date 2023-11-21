from diffusers import UNet2DModel
from torch import nn
from typing import Tuple, Optional, Union
import torch
    

class UNet(nn.Module):
    def __init__(self, image_size: Tuple[int, int]) -> None:
        super().__init__()

        if image_size > 64:
            self.model_fn = UNet2DModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        else:
            self.model_fn = UNet2DModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=4,
            block_out_channels=(256, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            mid_block_scale_factor=1.4142135623730951,
            center_input_sample=False,
            act_fn="swish",
            time_embedding_type="positional",
        )

    def forward(self, *args, **kwargs):
        return self.model_fn(*args, **kwargs, return_dict=True).sample
