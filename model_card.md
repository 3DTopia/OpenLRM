# Model Card for OpenLRM

## Overview

This model card is for the [OpenLRM](https://github.com/OpenLRM/OpenLRM) project, which is an open-source implementation of the paper [LRM](https://arxiv.org/abs/2311.04400).

## Model Details

| Model | Training Data | Layers | Feat. Dim | Trip. Dim. | Render Res. | Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LRM-Base-Obj | Objaverse | 12 | 1024 | 40 | 192 | [HF](https://huggingface.co/zxhezexin/OpenLRM) |
| LRM-Large-Obj | Objaverse | 16 | 1024 | 80 | 384 | [HF](https://huggingface.co/zxhezexin/OpenLRM) |
| LRM-Base | Objaverse + MVImgNet | 12 | 1024 | 40 | 192 | To be released |
| LRM-Large | Objaverse + MVImgNet | 16 | 1024 | 80 | 384 | To be released |

## Differences from the Original Paper

- We do not use the deferred back-propagation technique in the original paper.
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
