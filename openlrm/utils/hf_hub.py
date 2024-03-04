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


import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def wrap_model_hub(model_cls: nn.Module):
    class HfModel(model_cls, PyTorchModelHubMixin):
        def __init__(self, config: dict):
            super().__init__(**config)
            self.config = config
    return HfModel
