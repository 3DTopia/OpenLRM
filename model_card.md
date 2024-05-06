# Model Card for OpenLRM V1.1

## Overview

- This model card is for the [OpenLRM](https://github.com/3DTopia/OpenLRM) project, which is an open-source implementation of the paper [LRM](https://arxiv.org/abs/2311.04400).
- Information contained in this model card corresponds to [Version 1.1](https://github.com/3DTopia/OpenLRM/releases).

## Model Details

- Training data

    | Model | Training Data |
    | :---: | :---: |
    | [openlrm-obj-small-1.1](https://huggingface.co/zxhezexin/openlrm-obj-small-1.1) | Objaverse |
    | [openlrm-obj-base-1.1](https://huggingface.co/zxhezexin/openlrm-obj-base-1.1) | Objaverse |
    | [openlrm-obj-large-1.1](https://huggingface.co/zxhezexin/openlrm-obj-large-1.1) | Objaverse |
    | [openlrm-mix-small-1.1](https://huggingface.co/zxhezexin/openlrm-mix-small-1.1) | Objaverse + MVImgNet |
    | [openlrm-mix-base-1.1](https://huggingface.co/zxhezexin/openlrm-mix-base-1.1) | Objaverse + MVImgNet |
    | [openlrm-mix-large-1.1](https://huggingface.co/zxhezexin/openlrm-mix-large-1.1) | Objaverse + MVImgNet |

- Model architecture (version==1.1)

    | Type  | Layers | Feat. Dim | Attn. Heads | Triplane Dim. | Input Res. | Image Encoder     | Encoder Dim. | Size  |
    | :---: | :----: | :-------: | :---------: | :-----------: | :--------: | :---------------: | :----------: | :---: |
    | small |   12   |    512    |      8      |      32       |    224     | dinov2_vits14_reg |      384     | 446M  |
    | base  |   12   |    768    |     12      |      48       |    336     | dinov2_vitb14_reg |      768     | 1.04G |
    | large |   16   |   1024    |     16      |      80       |    448     | dinov2_vitb14_reg |      768     | 1.81G |

- Training settings

    | Type  | Rend. Res. | Rend. Patch | Ray Samples |
    | :---: | :--------: | :---------: | :---------: |
    | small |    192     |     64      |     96      |
    | base  |    288     |     96      |     96      |
    | large |    384     |    128      |    128      |

## Notable Differences from the Original Paper

- We do not use the deferred back-propagation technique in the original paper.
- We used random background colors during training.
- The image encoder is based on the [DINOv2](https://github.com/facebookresearch/dinov2) model with register tokens.
- The triplane decoder contains 4 layers in our implementation.

## License

- The model weights are released under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE_WEIGHT).
- They are provided for research purposes only, and CANNOT be used commercially.

## Disclaimer

This model is an open-source implementation and is NOT the official release of the original research paper. While it aims to reproduce the original results as faithfully as possible, there may be variations due to model implementation, training data, and other factors.

### Ethical Considerations

- This model should be used responsibly and ethically, and should not be used for malicious purposes.
- Users should be aware of potential biases in the training data.
- The model should not be used under the circumstances that could lead to harm or unfair treatment of individuals or groups.

### Usage Considerations

- The model is provided "as is" without warranty of any kind.
- Users are responsible for ensuring that their use complies with all relevant laws and regulations.
- The developers and contributors of this model are not liable for any damages or losses arising from the use of this model.

---

*This model card is subject to updates and modifications. Users are advised to check for the latest version regularly.*
