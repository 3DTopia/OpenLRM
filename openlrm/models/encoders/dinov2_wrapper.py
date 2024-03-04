# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from accelerate.logging import get_logger


logger = get_logger(__name__)


class Dinov2Wrapper(nn.Module):
    """
    Dino v2 wrapper using original implementation, hacked with modulation.
    """
    def __init__(self, model_name: str, modulation_dim: int = None, freeze: bool = True):
        super().__init__()
        self.modulation_dim = modulation_dim
        self.model = self._build_dinov2(model_name, modulation_dim=modulation_dim)
        if freeze:
            if modulation_dim is not None:
                raise ValueError("Modulated Dinov2 requires training, freezing is not allowed.")
            self._freeze()

    def _freeze(self):
        logger.warning(f"======== Freezing Dinov2Wrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dinov2(model_name: str, modulation_dim: int = None, pretrained: bool = True):
        from importlib import import_module
        dinov2_hub = import_module(".dinov2.hub.backbones", package=__package__)
        model_fn = getattr(dinov2_hub, model_name)
        logger.debug(f"Modulation dim for Dinov2 is {modulation_dim}.")
        model = model_fn(modulation_dim=modulation_dim, pretrained=pretrained)
        return model

    @torch.compile
    def forward(self, image: torch.Tensor, mod: torch.Tensor = None):
        # image: [N, C, H, W]
        # mod: [N, D] or None
        # RGB image with [0,1] scale and properly sized
        if self.modulation_dim is None:
            assert mod is None, "Unexpected modulation input in dinov2 forward."
            outs = self.model(image, is_training=True)
        else:
            assert mod is not None, "Modulation input is required in modulated dinov2 forward."
            outs = self.model(image, mod=mod, is_training=True)
        ret = torch.cat([
            outs["x_norm_clstoken"].unsqueeze(dim=1),
            outs["x_norm_patchtokens"],
        ], dim=1)
        return ret
