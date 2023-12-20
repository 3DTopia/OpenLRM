# Copyright (c) 2023, Zexin He
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
import math
import os
import imageio
import mcubes
import trimesh
import numpy as np
import argparse
from PIL import Image

from .models.generator import LRMGenerator
from .cam_utils import build_camera_principle, build_camera_standard, center_looking_at_camera_pose


class LRMInferrer:
    def __init__(self, model_name: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _checkpoint = self._load_checkpoint(model_name)
        _model_weights, _model_kwargs = _checkpoint['weights'], _checkpoint['kwargs']['model']
        self.model = self._build_model(_model_kwargs, _model_weights).eval()

        self.infer_kwargs = _checkpoint['kwargs']['infer']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _load_checkpoint(self, model_name: str, cache_dir = './.cache'):
        # download checkpoint if not exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cache_dir, f'{model_name}.pth')):
            # os.system(f'wget -O {os.path.join(cache_dir, f"{model_name}.pth")} https://zxhezexin.com/modelzoo/openlrm/{model_name}.pth')
            # raise FileNotFoundError(f"Checkpoint {model_name} not found in {cache_dir}")
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(repo_id='zxhezexin/OpenLRM', filename=f'{model_name}.pth', local_dir=cache_dir)
        else:
            local_path = os.path.join(cache_dir, f'{model_name}.pth')
        checkpoint = torch.load(local_path, map_location=self.device)
        return checkpoint

    def _build_model(self, model_kwargs, model_weights):
        model = LRMGenerator(**model_kwargs).to(self.device)
        model.load_state_dict(model_weights)
        print(f"======== Loaded model from checkpoint ========")
        return model

    @staticmethod
    def _get_surrounding_views(M: int = 160, radius: float = 2.0, height: float = 0.8):
        # M: number of surrounding views
        # radius: camera dist to center
        # height: height of the camera
        # return: (M, 3, 4)
        assert M > 0
        assert radius > 0

        camera_positions = []
        projected_radius = math.sqrt(radius ** 2 - height ** 2)
        for i in range(M):
            theta = 2 * math.pi * i / M - math.pi / 2
            x = projected_radius * math.cos(theta)
            y = projected_radius * math.sin(theta)
            z = height
            camera_positions.append([x, y, z])
        camera_positions = torch.tensor(camera_positions, dtype=torch.float32)
        extrinsics = center_looking_at_camera_pose(camera_positions)

        return extrinsics

    @staticmethod
    def _default_intrinsics():
        # return: (3, 2)
        fx = fy = 384
        cx = cy = 256
        w = h = 512
        intrinsics = torch.tensor([
            [fx, fy],
            [cx, cy],
            [w, h],
        ], dtype=torch.float32)
        return intrinsics

    def _default_source_camera(self, batch_size: int = 1):
        # return: (N, D_cam_raw)
        dist_to_center = 2
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -dist_to_center],
            [0, 1, 0, 0],
        ]], dtype=torch.float32)
        canonical_camera_intrinsics = self._default_intrinsics().unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(self, batch_size: int = 1):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = self._get_surrounding_views()
        render_camera_intrinsics = self._default_intrinsics().unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)
        render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    @staticmethod
    def images_to_video(images, output_path, fps, verbose=False):
        # images: (T, C, H, W)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        frames = []
        for i in range(images.shape[0]):
            frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
                f"Frame shape mismatch: {frame.shape} vs {images.shape}"
            assert frame.min() >= 0 and frame.max() <= 255, \
                f"Frame value out of range: {frame.min()} ~ {frame.max()}"
            frames.append(frame)
        imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='mpeg4', quality=10)
        if verbose:
            print(f"Saved video to {output_path}")

    def infer_single(self, image: torch.Tensor, render_size: int, mesh_size: int, export_video: bool, export_mesh: bool):
        # image: [1, C_img, H_img, W_img]
        mesh_thres = 1.0
        chunk_size = 2
        batch_size = 1

        source_camera = self._default_source_camera(batch_size).to(self.device)
        render_cameras = self._default_render_cameras(batch_size).to(self.device)

        with torch.no_grad():
            planes = self.model.forward_planes(image, source_camera)
            results = {}

            if export_video:
                # forward synthesizer per mini-batch
                frames = []
                for i in range(0, render_cameras.shape[1], chunk_size):
                    frames.append(
                        self.model.synthesizer(
                            planes,
                            render_cameras[:, i:i+chunk_size],
                            render_size,
                        )
                    )
                # merge frames
                frames = {
                    k: torch.cat([r[k] for r in frames], dim=1)
                    for k in frames[0].keys()
                }
                # update results
                results.update({
                    'frames': frames,
                })

            if export_mesh:
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

                results.update({
                    'mesh': mesh,
                })

            return results

    def infer(self, source_image: str, dump_path: str, source_size: int, render_size: int, mesh_size: int, export_video: bool, export_mesh: bool):

        source_image_size = source_size if source_size > 0 else self.infer_kwargs['source_size']

        image = torch.tensor(np.array(Image.open(source_image))).permute(2, 0, 1).unsqueeze(0) / 255.0
        # if RGBA, blend to RGB
        if image.shape[1] == 4:
            image = image[:, :3, ...] * image[:, 3:, ...] + (1 - image[:, 3:, ...])
        image = torch.nn.functional.interpolate(image, size=(source_image_size, source_image_size), mode='bicubic', align_corners=True)
        image = torch.clamp(image, 0, 1)
        results = self.infer_single(
            image.cuda(),
            render_size=render_size if render_size > 0 else self.infer_kwargs['render_size'],
            mesh_size=mesh_size,
            export_video=export_video,
            export_mesh=export_mesh,
        )

        image_name = os.path.basename(source_image)
        uid = image_name.split('.')[0]

        os.makedirs(dump_path, exist_ok=True)

        # dump video
        if 'frames' in results:
            renderings = results['frames']
            for k, v in renderings.items():
                if k == 'images_rgb':
                    self.images_to_video(
                        v[0],
                        os.path.join(dump_path, f'{uid}.mov'),
                        fps=40,
                    )
                else:
                    # torch.save(v[0], os.path.join(dump_path, f'{uid}_{k}.pth'))
                    pass

        # dump mesh
        if 'mesh' in results:
            mesh = results['mesh']
            # save ply format mesh
            mesh.export(os.path.join(dump_path, f'{uid}.ply'), 'ply')


if __name__ == '__main__':

    """
    Example usage:
    python -m lrm.inferrer --model_name lrm-base-obj-v1 --source_image ./assets/sample_input/owl.png --export_video --export_mesh
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='lrm-base-obj-v1')
    parser.add_argument('--source_image', type=str, default='./assets/sample_input/owl.png')
    parser.add_argument('--dump_path', type=str, default='./dumps')
    parser.add_argument('--source_size', type=int, default=-1)
    parser.add_argument('--render_size', type=int, default=-1)
    parser.add_argument('--mesh_size', type=int, default=384)
    parser.add_argument('--export_video', action='store_true')
    parser.add_argument('--export_mesh', action='store_true')
    args = parser.parse_args()

    with LRMInferrer(model_name=args.model_name) as inferrer:
        inferrer.infer(
            source_image=args.source_image,
            dump_path=args.dump_path,
            source_size=args.source_size,
            render_size=args.render_size,
            mesh_size=args.mesh_size,
            export_video=args.export_video,
            export_mesh=args.export_mesh,
        )
