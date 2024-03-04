# OpenLRM: Open-Source Large Reconstruction Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](LICENSE)
[![Weight License](https://img.shields.io/badge/Weight%20License-CC%20By%20NC%204.0-red)](LICENSE_WEIGHT)
[![LRM](https://img.shields.io/badge/LRM-Arxiv%20Link-green)](https://arxiv.org/abs/2311.04400)

[![HF Models](https://img.shields.io/badge/Models-Huggingface%20Models-bron)](https://huggingface.co/zxhezexin)
[![HF Demo](https://img.shields.io/badge/Demo-Huggingface%20Demo-blue)](https://huggingface.co/spaces/zxhezexin/OpenLRM)

<img src="assets/rendered_video/teaser.gif" width="75%" height="auto"/>

<div style="text-align: left">
    <img src="assets/mesh_snapshot/crop.owl.ply00.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.owl.ply01.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.building.ply00.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.building.ply01.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.rose.ply00.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.rose.ply01.png" width="12%" height="auto"/>
</div>

## News

- [2024.03.05] The [Huggingface demo](https://huggingface.co/spaces/zxhezexin/OpenLRM) now uses `openlrm-mix-base-1.1` model by default. Please refer to the [model card](model_card.md) for details on the updated model architecture and training settings.
- [2024.03.04] Version update v1.1. Release model weights trained on both Objaverse and MVImgNet. Codebase is majorly refactored for better usability and extensibility. Please refer to [v1.1.0](https://github.com/3DTopia/OpenLRM/releases/tag/v1.1.0) for details.
- [2024.01.09] Updated all v1.0 models trained on Objaverse. Please refer to [HF Models](https://huggingface.co/zxhezexin) and overwrite previous model weights.
- [2023.12.21] [Hugging Face Demo](https://huggingface.co/spaces/zxhezexin/OpenLRM) is online. Have a try!
- [2023.12.20] Release weights of the base and large models trained on Objaverse.
- [2023.12.20] We release this project OpenLRM, which is an open-source implementation of the paper [LRM](https://arxiv.org/abs/2311.04400).

## Setup

### Installation
```
git clone https://github.com/3DTopia/OpenLRM.git
cd OpenLRM
```

### Environment
- Install requirements for OpenLRM first.
  ```
  pip install -r requirements.txt
  ```
- Please then follow the [xFormers installation guide](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers) to enable memory efficient attention inside [DINOv2 encoder](openlrm/models/encoders/dinov2/layers/attention.py).

## Quick Start

### Pretrained Models

- Model weights are released on [Hugging Face](https://huggingface.co/zxhezexin).
- Weights will be downloaded automatically when you run the inference script for the first time.
- Please be aware of the [license](LICENSE_WEIGHT) before using the weights.

| Model | Training Data | Layers | Feat. Dim | Trip. Dim. | In. Res. | Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| openlrm-obj-small-1.1 | Objaverse | 12 | 512 | 32 | 224 | [HF](https://huggingface.co/zxhezexin/openlrm-obj-small-1.1) |
| openlrm-obj-base-1.1 | Objaverse | 12 | 768 | 48 | 336 | [HF](https://huggingface.co/zxhezexin/openlrm-obj-base-1.1) |
| openlrm-obj-large-1.1 | Objaverse | 16 | 1024 | 80 | 448 | [HF](https://huggingface.co/zxhezexin/openlrm-obj-large-1.1) |
| openlrm-mix-small-1.1 | Objaverse + MVImgNet | 12 | 512 | 32 | 224 | [HF](https://huggingface.co/zxhezexin/openlrm-mix-small-1.1) |
| openlrm-mix-base-1.1 | Objaverse + MVImgNet | 12 | 768 | 48 | 336 | [HF](https://huggingface.co/zxhezexin/openlrm-mix-base-1.1) |
| openlrm-mix-large-1.1 | Objaverse + MVImgNet | 16 | 1024 | 80 | 448 | [HF](https://huggingface.co/zxhezexin/openlrm-mix-large-1.1) |

Model cards with additional details can be found in [model_card.md](model_card.md).

### Prepare Images
- We put some sample inputs under `assets/sample_input`, and you can quickly try them.
- Prepare RGBA images or RGB images with white background (with some background removal tools, e.g., [Rembg](https://github.com/danielgatis/rembg), [Clipdrop](https://clipdrop.co)).

### Inference
- Run the inference script to get 3D assets.
- You may specify which form of output to generate by setting the flags `EXPORT_VIDEO=true` and `EXPORT_MESH=true`.
- Please set default `INFER_CONFIG` according to the model you want to use. E.g., `infer-b.yaml` for base models and `infer-s.yaml` for small models.
- An example usage is as follows:

  ```
  # Example usage
  EXPORT_VIDEO=true
  EXPORT_MESH=true
  INFER_CONFIG="./configs/infer-b.yaml"
  MODEL_NAME="zxhezexin/openlrm-mix-base-1.1"
  IMAGE_INPUT="./assets/sample_input/owl.png"

  python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
  ```

## Training
To be released soon.

## Acknowledgement

- We thank the authors of the [original paper](https://arxiv.org/abs/2311.04400) for their great work! Special thanks to Kai Zhang and Yicong Hong for assistance during the reproduction.
- This project is supported by Shanghai AI Lab by providing the computing resources.
- This project is advised by Ziwei Liu and Jiaya Jia.

## Citation

If you find this work useful for your research, please consider citing:
```
@article{hong2023lrm,
  title={Lrm: Large reconstruction model for single image to 3d},
  author={Hong, Yicong and Zhang, Kai and Gu, Jiuxiang and Bi, Sai and Zhou, Yang and Liu, Difan and Liu, Feng and Sunkavalli, Kalyan and Bui, Trung and Tan, Hao},
  journal={arXiv preprint arXiv:2311.04400},
  year={2023}
}
```

```
@misc{openlrm,
  title = {OpenLRM: Open-Source Large Reconstruction Models},
  author = {Zexin He and Tengfei Wang},
  year = {2023},
  howpublished = {\url{https://github.com/3DTopia/OpenLRM}},
}
```

## License

- OpenLRM as a whole is licensed under the [Apache License, Version 2.0](LICENSE), while certain components are covered by [NVIDIA's proprietary license](LICENSE_NVIDIA). Users are responsible for complying with the respective licensing terms of each component.
- Model weights are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE_WEIGHT). They are provided for research purposes only, and CANNOT be used commercially.
