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


import os
import argparse
from omegaconf import OmegaConf
import torch.nn as nn
from accelerate import Accelerator
import safetensors

import sys
sys.path.append(".")

from openlrm.utils.hf_hub import wrap_model_hub
from openlrm.models import model_dict


def auto_load_model(cfg, model: nn.Module) -> int:

    ckpt_root = os.path.join(
        cfg.saver.checkpoint_root,
        cfg.experiment.parent, cfg.experiment.child,
    )
    if not os.path.exists(ckpt_root):
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")
    ckpt_dirs = os.listdir(ckpt_root)
    if len(ckpt_dirs) == 0:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_root}")
    ckpt_dirs.sort()

    load_step = f"{cfg.convert.global_step}" if cfg.convert.global_step is not None else ckpt_dirs[-1]
    load_model_path = os.path.join(ckpt_root, load_step, 'model.safetensors')
    print(f"Loading from {load_model_path}")
    safetensors.torch.load_model(model, load_model_path)

    return int(load_step)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./assets/config.yaml')
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(unknown)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    """
    [cfg.convert]
    global_step: int
    save_dir: str
    """

    accelerator = Accelerator()

    hf_model_cls = wrap_model_hub(model_dict[cfg.experiment.type])
    hf_model = hf_model_cls(dict(cfg.model))
    loaded_step = auto_load_model(cfg, hf_model)
    dump_path = os.path.join(
        f"./exps/releases",
        cfg.experiment.parent, cfg.experiment.child,
        f'step_{loaded_step:06d}',
    )
    print(f"Saving locally to {dump_path}")
    os.makedirs(dump_path, exist_ok=True)
    hf_model.save_pretrained(
        save_directory=dump_path,
        config=hf_model.config,
    )
