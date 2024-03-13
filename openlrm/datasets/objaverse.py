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
from typing import Union
import random
import numpy as np
import torch
from megfile import smart_path_join, smart_open

from .base import BaseDataset
from .cam_utils import build_camera_standard, build_camera_principle, camera_normalization_objaverse
from ..utils.proxy import no_proxy

__all__ = ['ObjaverseDataset']


class ObjaverseDataset(BaseDataset):

    def __init__(self, root_dirs: list[str], meta_path: str,
                 sample_side_views: int,
                 render_image_res_low: int, render_image_res_high: int, render_region_size: int,
                 source_image_res: int, normalize_camera: bool,
                 normed_dist_to_center: Union[float, str] = None, num_all_views: int = 32):
        super().__init__(root_dirs, meta_path)
        self.sample_side_views = sample_side_views
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        self.render_region_size = render_region_size
        self.source_image_res = source_image_res
        self.normalize_camera = normalize_camera
        self.normed_dist_to_center = normed_dist_to_center
        self.num_all_views = num_all_views

    @staticmethod
    def _load_pose(file_path):
        pose = np.load(smart_open(file_path, 'rb'))
        pose = torch.from_numpy(pose).float()
        return pose

    @no_proxy
    def inner_get_item(self, idx):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """
        uid = self.uids[idx]
        root_dir = self._locate_datadir(self.root_dirs, uid, locator="intrinsics.npy")
        
        pose_dir = os.path.join(root_dir, uid, 'pose')
        rgba_dir = os.path.join(root_dir, uid, 'rgba')
        intrinsics_path = os.path.join(root_dir, uid, 'intrinsics.npy')

        # load intrinsics
        intrinsics = np.load(smart_open(intrinsics_path, 'rb'))
        intrinsics = torch.from_numpy(intrinsics).float()

        # sample views (incl. source view and side views)
        sample_views = np.random.choice(range(self.num_all_views), self.sample_side_views + 1, replace=False)
        poses, rgbs, bg_colors = [], [], []
        source_image = None
        for view in sample_views:
            pose_path = smart_path_join(pose_dir, f'{view:03d}.npy')
            rgba_path = smart_path_join(rgba_dir, f'{view:03d}.png')
            pose = self._load_pose(pose_path)
            bg_color = random.choice([0.0, 0.5, 1.0])
            rgb = self._load_rgba_image(rgba_path, bg_color=bg_color)
            poses.append(pose)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            if source_image is None:
                source_image = self._load_rgba_image(rgba_path, bg_color=1.0)
        assert source_image is not None, "Really bad luck!"
        poses = torch.stack(poses, dim=0)
        rgbs = torch.cat(rgbs, dim=0)

        if self.normalize_camera:
            poses = camera_normalization_objaverse(self.normed_dist_to_center, poses)

        # build source and target camera features
        source_camera = build_camera_principle(poses[:1], intrinsics.unsqueeze(0)).squeeze(0)
        render_camera = build_camera_standard(poses, intrinsics.repeat(poses.shape[0], 1, 1))

        # adjust source image resolution
        source_image = torch.nn.functional.interpolate(
            source_image, size=(self.source_image_res, self.source_image_res), mode='bicubic', align_corners=True).squeeze(0)
        source_image = torch.clamp(source_image, 0, 1)

        # adjust render image resolution and sample intended rendering region
        render_image_res = np.random.randint(self.render_image_res_low, self.render_image_res_high + 1)
        render_image = torch.nn.functional.interpolate(
            rgbs, size=(render_image_res, render_image_res), mode='bicubic', align_corners=True)
        render_image = torch.clamp(render_image, 0, 1)
        anchors = torch.randint(
            0, render_image_res - self.render_region_size + 1, size=(self.sample_side_views + 1, 2))
        crop_indices = torch.arange(0, self.render_region_size, device=render_image.device)
        index_i = (anchors[:, 0].unsqueeze(1) + crop_indices).view(-1, self.render_region_size, 1)
        index_j = (anchors[:, 1].unsqueeze(1) + crop_indices).view(-1, 1, self.render_region_size)
        batch_indices = torch.arange(self.sample_side_views + 1, device=render_image.device).view(-1, 1, 1)
        cropped_render_image = render_image[batch_indices, :, index_i, index_j].permute(0, 3, 1, 2)

        return {
            'uid': uid,
            'source_camera': source_camera,
            'render_camera': render_camera,
            'source_image': source_image,
            'render_image': cropped_render_image,
            'render_anchors': anchors,
            'render_full_resolutions': torch.tensor([[render_image_res]], dtype=torch.float32).repeat(self.sample_side_views + 1, 1),
            'render_bg_colors': torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1),
        }
