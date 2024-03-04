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
import os
import argparse
import mcubes
import trimesh
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from accelerate.logging import get_logger

from .base_inferrer import Inferrer
from openlrm.datasets.cam_utils import build_camera_principle, build_camera_standard, surrounding_views_linspace, create_intrinsics
from openlrm.utils.logging import configure_logger
from openlrm.runners import REGISTRY_RUNNERS
from openlrm.utils.video import images_to_video
from openlrm.utils.hf_hub import wrap_model_hub


logger = get_logger(__name__)


def parse_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--infer', type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get('APP_INFER') is not None:
        args.infer = os.environ.get('APP_INFER')
    if os.environ.get('APP_MODEL_NAME') is not None:
        cli_cfg.model_name = os.environ.get('APP_MODEL_NAME')

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(cfg_train.experiment.parent, cfg_train.experiment.child, os.path.basename(cli_cfg.model_name).split('_')[-1])
        cfg.video_dump = os.path.join("exps", 'videos', _relative_path)
        cfg.mesh_dump = os.path.join("exps", 'meshes', _relative_path)

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault('video_dump', os.path.join("dumps", cli_cfg.model_name, 'videos'))
        cfg.setdefault('mesh_dump', os.path.join("dumps", cli_cfg.model_name, 'meshes'))

    cfg.merge_with(cli_cfg)

    """
    [required]
    model_name: str
    image_input: str
    export_video: bool
    export_mesh: bool

    [special]
    source_size: int
    render_size: int
    video_dump: str
    mesh_dump: str

    [default]
    render_views: int
    render_fps: int
    mesh_size: int
    mesh_thres: float
    frame_size: int
    logger: str
    """

    cfg.setdefault('logger', 'INFO')

    # assert not (args.config is not None and args.infer is not None), "Only one of config and infer should be provided"
    assert cfg.model_name is not None, "model_name is required"
    if not os.environ.get('APP_ENABLED', None):
        assert cfg.image_input is not None, "image_input is required"
        assert cfg.export_video or cfg.export_mesh, \
            "At least one of export_video or export_mesh should be True"
        cfg.app_enabled = False
    else:
        cfg.app_enabled = True

    return cfg


@REGISTRY_RUNNERS.register('infer.lrm')
class LRMInferrer(Inferrer):

    EXP_TYPE: str = 'lrm'

    def __init__(self):
        super().__init__()

        self.cfg = parse_configs()
        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )

        self.model = self._build_model(self.cfg).to(self.device)

    def _build_model(self, cfg):
        from openlrm.models import model_dict
        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model

    def _default_source_camera(self, dist_to_center: float = 2.0, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -dist_to_center],
            [0, 1, 0, 0],
        ]], dtype=torch.float32, device=device)
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(self, n_views: int, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(n_views=n_views, device=device)
        render_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)
        render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    def infer_planes(self, image: torch.Tensor, source_cam_dist: float):
        N = image.shape[0]
        source_camera = self._default_source_camera(dist_to_center=source_cam_dist, batch_size=N, device=self.device)
        planes = self.model.forward_planes(image, source_camera)
        assert N == planes.shape[0]
        return planes

    def infer_video(self, planes: torch.Tensor, frame_size: int, render_size: int, render_views: int, render_fps: int, dump_video_path: str):
        N = planes.shape[0]
        render_cameras = self._default_render_cameras(n_views=render_views, batch_size=N, device=self.device)
        render_anchors = torch.zeros(N, render_cameras.shape[1], 2, device=self.device)
        render_resolutions = torch.ones(N, render_cameras.shape[1], 1, device=self.device) * render_size
        render_bg_colors = torch.ones(N, render_cameras.shape[1], 1, device=self.device, dtype=torch.float32) * 1.

        frames = []
        for i in range(0, render_cameras.shape[1], frame_size):
            frames.append(
                self.model.synthesizer(
                    planes=planes,
                    cameras=render_cameras[:, i:i+frame_size],
                    anchors=render_anchors[:, i:i+frame_size],
                    resolutions=render_resolutions[:, i:i+frame_size],
                    bg_colors=render_bg_colors[:, i:i+frame_size],
                    region_size=render_size,
                )
            )
        # merge frames
        frames = {
            k: torch.cat([r[k] for r in frames], dim=1)
            for k in frames[0].keys()
        }
        # dump
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        for k, v in frames.items():
            if k == 'images_rgb':
                images_to_video(
                    images=v[0],
                    output_path=dump_video_path,
                    fps=render_fps,
                    gradio_codec=self.cfg.app_enabled,
                )

    def infer_mesh(self, planes: torch.Tensor, mesh_size: int, mesh_thres: float, dump_mesh_path: str):
        grid_out = self.model.synthesizer.forward_grid(
            planes=planes,
            grid_size=mesh_size,
        )
        
        vtx, faces = mcubes.marching_cubes(grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(), mesh_thres)
        vtx = vtx / (mesh_size - 1) * 2 - 1

        vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=self.device).unsqueeze(0)
        vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
        vtx_colors = (vtx_colors * 255).astype(np.uint8)
        
        mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

        # dump
        os.makedirs(os.path.dirname(dump_mesh_path), exist_ok=True)
        mesh.export(dump_mesh_path)

    def infer_single(self, image_path: str, source_cam_dist: float, export_video: bool, export_mesh: bool, dump_video_path: str, dump_mesh_path: str):
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_views = self.cfg.render_views
        render_fps = self.cfg.render_fps
        mesh_size = self.cfg.mesh_size
        mesh_thres = self.cfg.mesh_thres
        frame_size = self.cfg.frame_size
        source_cam_dist = self.cfg.source_cam_dist if source_cam_dist is None else source_cam_dist

        # prepare image: [1, C_img, H_img, W_img], 0-1 scale
        image = torch.from_numpy(np.array(Image.open(image_path))).to(self.device)
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        if image.shape[1] == 4:  # RGBA
            image = image[:, :3, ...] * image[:, 3:, ...] + (1 - image[:, 3:, ...])
        image = torch.nn.functional.interpolate(image, size=(source_size, source_size), mode='bicubic', align_corners=True)
        image = torch.clamp(image, 0, 1)

        with torch.no_grad():
            planes = self.infer_planes(image, source_cam_dist=source_cam_dist)

            results = {}
            if export_video:
                frames = self.infer_video(planes, frame_size=frame_size, render_size=render_size, render_views=render_views, render_fps=render_fps, dump_video_path=dump_video_path)
                results.update({
                    'frames': frames,
                })
            if export_mesh:
                mesh = self.infer_mesh(planes, mesh_size=mesh_size, mesh_thres=mesh_thres, dump_mesh_path=dump_mesh_path)
                results.update({
                    'mesh': mesh,
                })

    def infer(self):

        image_paths = []
        if os.path.isfile(self.cfg.image_input):
            omit_prefix = os.path.dirname(self.cfg.image_input)
            image_paths.append(self.cfg.image_input)
        else:
            omit_prefix = self.cfg.image_input
            for root, dirs, files in os.walk(self.cfg.image_input):
                for file in files:
                    if file.endswith('.png'):
                        image_paths.append(os.path.join(root, file))
            image_paths.sort()

        # alloc to each DDP worker
        image_paths = image_paths[self.accelerator.process_index::self.accelerator.num_processes]

        for image_path in tqdm(image_paths, disable=not self.accelerator.is_local_main_process):

            # prepare dump paths
            image_name = os.path.basename(image_path)
            uid = image_name.split('.')[0]
            subdir_path = os.path.dirname(image_path).replace(omit_prefix, '')
            subdir_path = subdir_path[1:] if subdir_path.startswith('/') else subdir_path
            dump_video_path = os.path.join(
                self.cfg.video_dump,
                subdir_path,
                f'{uid}.mov',
            )
            dump_mesh_path = os.path.join(
                self.cfg.mesh_dump,
                subdir_path,
                f'{uid}.ply',
            )

            self.infer_single(
                image_path,
                source_cam_dist=None,
                export_video=self.cfg.export_video,
                export_mesh=self.cfg.export_mesh,
                dump_video_path=dump_video_path,
                dump_mesh_path=dump_mesh_path,
            )
