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
from PIL import Image
import numpy as np
import gradio as gr


def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")

def prepare_working_dir():
    import tempfile
    working_dir = tempfile.TemporaryDirectory()
    return working_dir

def init_preprocessor():
    from openlrm.utils.preprocess import Preprocessor
    global preprocessor
    preprocessor = Preprocessor()

def preprocess_fn(image_in: np.ndarray, remove_bg: bool, recenter: bool, working_dir):
    image_raw = os.path.join(working_dir.name, "raw.png")
    with Image.fromarray(image_in) as img:
        img.save(image_raw)
    image_out = os.path.join(working_dir.name, "rembg.png")
    success = preprocessor.preprocess(image_path=image_raw, save_path=image_out, rmbg=remove_bg, recenter=recenter)
    assert success, f"Failed under preprocess_fn!"
    return image_out


def demo_openlrm(infer_impl):

    def core_fn(image: str, source_cam_dist: float, working_dir):
        dump_video_path = os.path.join(working_dir.name, "output.mp4")
        dump_mesh_path = os.path.join(working_dir.name, "output.ply")
        infer_impl(
            image_path=image,
            source_cam_dist=source_cam_dist,
            export_video=True,
            export_mesh=False,
            dump_video_path=dump_video_path,
            dump_mesh_path=dump_mesh_path,
        )
        return dump_video_path

    def example_fn(image: np.ndarray):
        from gradio.utils import get_cache_folder
        working_dir = get_cache_folder()
        image = preprocess_fn(
            image_in=image,
            remove_bg=True,
            recenter=True,
            working_dir=working_dir,
        )
        video = core_fn(
            image=image,
            source_cam_dist=2.0,
            working_dir=working_dir,
        )
        return image, video


    _TITLE = '''OpenLRM: Open-Source Large Reconstruction Models'''

    _DESCRIPTION = '''
        <div>
            <a style="display:inline-block" href='https://github.com/3DTopia/OpenLRM'><img src='https://img.shields.io/github/stars/3DTopia/OpenLRM?style=social'/></a>
            <a style="display:inline-block; margin-left: .5em" href="https://huggingface.co/zxhezexin"><img src='https://img.shields.io/badge/Model-Weights-blue'/></a>
        </div>
        OpenLRM is an open-source implementation of Large Reconstruction Models.

        <strong>Image-to-3D in 10 seconds with A100!</strong>

        <strong>Disclaimer:</strong> This demo uses `openlrm-mix-base-1.1` model with 288x288 rendering resolution here for a quick demonstration.
    '''

    with gr.Blocks(analytics_enabled=False) as demo:

        # HEADERS
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        with gr.Row():
            gr.Markdown(_DESCRIPTION)

        # DISPLAY
        with gr.Row():

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_input_image"):
                    with gr.TabItem('Input Image'):
                        with gr.Row():
                            input_image = gr.Image(label="Input Image", image_mode="RGBA", width="auto", sources="upload", type="numpy", elem_id="content_image")

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_processed_image"):
                    with gr.TabItem('Processed Image'):
                        with gr.Row():
                            processed_image = gr.Image(label="Processed Image", image_mode="RGBA", type="filepath", elem_id="processed_image", width="auto", interactive=False)

            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_render_video"):
                    with gr.TabItem('Rendered Video'):
                        with gr.Row():
                            output_video = gr.Video(label="Rendered Video", format="mp4", width="auto", autoplay=True)

        # SETTING
        with gr.Row():
            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_attrs"):
                    with gr.TabItem('Settings'):
                        with gr.Column(variant='panel'):
                            gr.Markdown(
                                """
                                <strong>Best Practice</strong>:
                                    Centered objects in reasonable sizes. Try adjusting source camera distances.
                                """
                            )
                            checkbox_rembg = gr.Checkbox(True, label='Remove background')
                            checkbox_recenter = gr.Checkbox(True, label='Recenter the object')
                            slider_cam_dist = gr.Slider(1.0, 3.5, value=2.0, step=0.1, label="Source Camera Distance")
                            submit = gr.Button('Generate', elem_id="openlrm_generate", variant='primary')

        # EXAMPLES
        with gr.Row():
            examples = [
                ['assets/sample_input/owl.png'],
                ['assets/sample_input/building.png'],
                ['assets/sample_input/mailbox.png'],
                ['assets/sample_input/fire.png'],
                ['assets/sample_input/girl.png'],
                ['assets/sample_input/lamp.png'],
                ['assets/sample_input/hydrant.png'],
                ['assets/sample_input/hotdogs.png'],
                ['assets/sample_input/traffic.png'],
                ['assets/sample_input/ceramic.png'],
            ]
            gr.Examples(
                examples=examples,
                inputs=[input_image], 
                outputs=[processed_image, output_video],
                fn=example_fn,
                cache_examples=bool(os.getenv('SPACE_ID')),
                examples_per_page=20,
            )

        working_dir = gr.State()
        submit.click(
            fn=assert_input_image,
            inputs=[input_image],
            queue=False,
        ).success(
            fn=prepare_working_dir,
            outputs=[working_dir],
            queue=False,
        ).success(
            fn=preprocess_fn,
            inputs=[input_image, checkbox_rembg, checkbox_recenter, working_dir],
            outputs=[processed_image],
        ).success(
            fn=core_fn,
            inputs=[processed_image, slider_cam_dist, working_dir],
            outputs=[output_video],
        )

        demo.queue()
        demo.launch()


def launch_gradio_app():

    os.environ.update({
        "APP_ENABLED": "1",
        "APP_MODEL_NAME": "zxhezexin/openlrm-mix-base-1.1",
        "APP_INFER": "./configs/infer-gradio.yaml",
        "APP_TYPE": "infer.lrm",
        "NUMBA_THREADING_LAYER": 'omp',
    })

    from openlrm.runners import REGISTRY_RUNNERS
    from openlrm.runners.infer.base_inferrer import Inferrer
    InferrerClass : Inferrer = REGISTRY_RUNNERS[os.getenv("APP_TYPE")]
    with InferrerClass() as inferrer:
        init_preprocessor()
        if not bool(os.getenv('SPACE_ID')):
            from openlrm.utils.proxy import no_proxy
            demo = no_proxy(demo_openlrm)
        else:
            demo = demo_openlrm
        demo(infer_impl=inferrer.infer_single)


if __name__ == '__main__':

    launch_gradio_app()
