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


import sys
sys.path.append('.')

import argparse
from accelerate import Accelerator

from openlrm.utils.hf_hub import wrap_model_hub
from openlrm.models import model_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--local_ckpt', type=str, required=True)
    parser.add_argument('--repo_id', type=str, required=True)
    args, unknown = parser.parse_known_args()

    accelerator = Accelerator()

    hf_model_cls = wrap_model_hub(model_dict[args.model_type])
    hf_model = hf_model_cls.from_pretrained(args.local_ckpt)
    hf_model.push_to_hub(
        repo_id=args.repo_id,
        config=hf_model.config,
        private=True,
    )
