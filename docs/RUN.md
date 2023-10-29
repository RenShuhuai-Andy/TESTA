## Pre-training
### Pre-Training from Scratch

Multimodal pre-training from scratch with a TimeSFormer vision encoder and a BERT text encoder:

```bash
python -m torch.distributed.run --nproc_per_node=8 video_pretrain.py --config ./configs/pretrain_timesformer.yaml --output_dir output/pretrain_video --mixed_precision_method apex
```

To change the pre-training datasets, just modify the `train_file` list in the config file. 

If you want to separate image-text datasets (deduplicated at low frame rates) and video-text datasets (sampled at high frame rates) for efficient pre-training, please use the `--sep_img` option and put the paths of the image datasets in the `img_config` file. Here is an example:

```bash
python -m torch.distributed.run --nproc_per_node=8 video_pretrain.py --config ./configs/pretrain_timesformer.yaml --sep_img --img_config ./configs/pretrain_timesformer_coco.yaml
--output_dir output/pretrain_timesformer_sep_img --mixed_precision_method apex
```

### Continual Pre-Training from the BLIP Image-Language Model

Alternatively, this repository supports continually pre-training the BLIP pre-trained model on video data:
```bash
python -m torch.distributed.run --nproc_per_node=8 video_pretrain.py --config ./configs/pretrain_timesformer_from_blip.yaml --output_dir output/pretrain_blip --mixed_precision_method apex
```
We support two ways to intialize the video encoder from the image encoder of BLIP (please modify in the `attention_type` line of the config file): 

1. 'space-only' (concating the features of each frame before the language decoder) 
2. 'divided_space_time' (adding temporal attention modules before the original spatial attention modules in each ViT layer).

For the BLIP checkpoints, please download them from the orignial repository (https://github.com/salesforce/BLIP) and put the local path in the `pretrained` line in the config file.



## Fine tuning

### Video-text retrieval:
```bash
python -m torch.distributed.run --nproc_per_node=8 train_video_retrieval.py --config configs/retrieval_queryd_f32.yaml --output_dir ./output/video_retrieval/queryd --mixed_precision_method apex
python -m torch.distributed.run --nproc_per_node=8 train_video_retrieval.py --config configs/retrieval_didemo_f32.yaml --output_dir ./output/video_retrieval/didemo --mixed_precision_method apex
python -m torch.distributed.run --nproc_per_node=8 train_video_retrieval.py --config configs/retrieval_activitynet_f32.yaml --output_dir ./output/video_retrieval/activitynet --mixed_precision_method apex
python -m torch.distributed.run --nproc_per_node=8 train_video_retrieval.py --config configs/retrieval_condensedmovies_f32.yaml --output_dir ./output/video_retrieval/condensedmovies --mixed_precision_method apex
```

PS. If you want to finetune 96 frames for ActivityNet (`--config configs/retrieval_activitynet_f96.yaml`), please add `--low_resource_eval` argument to avoid memory explosion.

### Video QA:
```bash
python -m torch.distributed.run --nproc_per_node=8 train_video_qa.py --config ./configs/vqa_activitynet.yaml --output_dir output/video_qa/activitynet --mixed_precision_method apex
```

## Zero-shot Evaluation

### Video-text retrieval:
```bash
python -m torch.distributed.run --nproc_per_node=8 train_video_retrieval.py --config configs/retrieval_queryd_f32.yaml --output_dir ./output/video_retrieval/queryd --mixed_precision_method apex --evaluate
python -m torch.distributed.run --nproc_per_node=8 train_video_retrieval.py --config configs/retrieval_didemo_f32.yaml --output_dir ./output/video_retrieval/didemo --mixed_precision_method apex --evaluate
python -m torch.distributed.run --nproc_per_node=8 train_video_retrieval.py --config configs/retrieval_activitynet_f32.yaml --output_dir ./output/video_retrieval/activitynet --mixed_precision_method apex --evaluate
python -m torch.distributed.run --nproc_per_node=8 train_video_retrieval.py --config configs/retrieval_condensedmovies_f32.yaml --output_dir ./output/video_retrieval/condensedmovies --mixed_precision_method apex --evaluate
```

### Video QA:
```bash
python -m torch.distributed.run --nproc_per_node=8 train_video_qa.py --config ./configs/vqa_activitynet.yaml --output_dir output/video_qa/activitynet --mixed_precision_method apex --evaluate
```
