# TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding

![TESTA Visualization](figs/testa.png)

Currently, the repository contains the code for pre-training a general-purpose video-language model and fine-tuning it on downstream video understanding tasks including video-paragraph retrieval and VideoQA.

![TESTA Arch](figs/arch.png)
## Installation

This is the codebase of the video understanding project under the joint project of the LANCO Lab (Peking University) and Huawei.
To install the dependencies, run
```bash
# create 
conda env create -f environment.yml
# activate
conda activate testa
```

## Data preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.

## Models

### Pre-trained model

zero-shot performance (32 frames):

| Model                 | QuerYD R@1 | DiDeMo R@1 | ActivityNet Caption R@1 | GFLOPs | Checkpoint                                                                                             |
|-----------------------|------------|------------|-------------------------|--------|--------------------------------------------------------------------------------------------------------|
| TESTA-base (ViT-B/16) | 64.4       | 64.9       | 37.1                    | 786    | [testa_model_base_pretrain.pth](https://huggingface.co/ShuhuaiRen/TESTA_model_base_pretrain/tree/main) |

### Fine-tuned model

To be uploaded...

## Training and Evaluation
Please refer to the [RUN.md](docs/RUN.md) for detailed instructions on training, evaluating and reproducing the results.


## Acknowledgement
The codebase relies on resources from [BLIP](https://github.com/salesforce/BLIP), [ToMe](https://github.com/facebookresearch/ToMe),and [TimeSFormer](https://github.com/facebookresearch/TimeSformer). We thank the original authors for their open-sourcing.
