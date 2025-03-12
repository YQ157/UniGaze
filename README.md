# UniGaze
This is the official repository for the paper [UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training](https://arxiv.org/pdf/2502.02307)


# Installation
we tested on XX enviornment
- python 3.8
- torch 2.0

please install the required packages by pip install -r requirements

## Overview
- this repo contains the [MAE](https://github.com/facebookresearch/mae) pre-training and gaze estimation training.
### Milestones:
- :white_check_mark: Release pre-trained MAE checkpoints (B, L, H) and gaze estimation training code.
- [ ] Release UniGaze models trained on the *joint dataset* for inference.
- [ ] Release inference code or gaze estimation demo.
- [ ] Release the MAE pre-training code.



## Pre-training (MAE)
(skip for now)


# UniGaze
This is the official repository for the paper [UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training](https://arxiv.org/pdf/2502.02307).

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Overview
This repository contains code for both **MAE pre-training** and **gaze estimation training**.


---

## Pre-training (MAE)
*(Details to be added later.)*

---



## Training (Gaze Estimation)
Refer to [Gaze estimation training.md](./gaze_estimation/README_GAZE.md)



## Inference

## Using Trained Gaze Estimation Models
We provide **UniGaze-B, UniGaze-L, and UniGaze-H**, trained on **joint datasets** to enhance robustness and generalizability.

### Available Models
| Backbone | Config Name | Checkpoint | Training Data |
|----------|------------|------------|---------------|
| UniGaze-B | `configs/model/unigaze_b.yaml` | [Download](#) | Joint datasets |
| UniGaze-L | `configs/model/unigaze_l.yaml` | [Download](#) | Joint datasets |
| UniGaze-H | `configs/model/unigaze_h.yaml` | [Download](#) | Joint datasets |


```bash
projdir=<...>/UniGaze/gaze_estimation
cd ${projdir}
model=configs/model/mae/mae_vit_h_14_pretrain_1023/299epoch.yaml 
ckpt_resume=<path to the trained gaze estimator checkpoint>

python draw_predict_video_wild.py \
    --model_cfg_path ${model} \
    -i ./input_video \
    --ckpt_resume ${ckpt_resume}
```


# Citation
```
@article{qin2025unigaze,
  title={UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training},
  author={Qin, Jiawei and Zhang, Xucong and Sugano, Yusuke},
  journal={arXiv preprint arXiv:2502.02307},
  year={2025}
}

@inproceedings{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proc. CVPR},
  pages={16000--16009},
  year={2022}
}
```