# UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training
<a href="https://jqin-home.github.io/">Jiawei Qin</a><sup>1</sup>, 
<a href="https://www.ccmitss.com/zhang">Xucong Zhang</a><sup>2</sup>, 
<a href="https://www.yusuke-sugano.info/">Yusuke Sugano</a><sup>1</sup>, 

*<sup>1</sup>The University of Tokyo, <sup>2</sup>Computer Vision Lab, Delft University of Technology 

> [[arxiv]](https://arxiv.org/pdf/2502.02307)





<!-- [*UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training*](https://arxiv.org/pdf/2502.02307)  -->

<!-- <a href="https://jqin-home.github.io/">Jiawei Qin</a><sup>1</sup>, 
<a href="https://www.ccmitss.com/zhang">Xucong Zhang</a><sup>2</sup>, 
<a href="https://www.yusuke-sugano.info/">Yusuke Sugano</a><sup>1</sup>, 

*<sup>1</sup>The University of Tokyo, <sup>2</sup>Computer Vision Lab, Delft University of Technology  -->




<!-- <h4 align="left">
<a href="">Project Page</a>
</h4> -->



## Overview
This repository contains the official PyTorch implementation of both **MAE pre-training** and **gaze estimation training**.



### Todo:
:white_check_mark: Release pre-trained MAE checkpoints (B, L, H) and gaze estimation training code.
- [ ] Release UniGaze models trained on the *joint dataset* for inference.
- [ ] Release inference code or gaze estimation demo.
- [ ] Release the MAE pre-training code.

---


## Installation
<!-- 
we tested on:
- python 3.8
- torch 2.0.1
- torchvision 0.15.2
- numpy 1.24.2
- timm 1.0.9 -->

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Pre-training (MAE)
*(Coming soon)*

---



## Training (Gaze Estimation)
Refer to [Gaze estimation training.md](./gaze_estimation/README.md)


## Inference

## Using Trained Gaze Estimation Models
*(Coming soon)*

<!-- We provide **UniGaze-B, UniGaze-L, and UniGaze-H**, trained on **joint datasets** to enhance robustness and generalizability.

### Available Models
| Backbone | Config Name | Checkpoint | Training Data |
|----------|------------|------------|---------------|
| UniGaze-B | `configs/model/mae_b_16_gaze.yaml` | [Download](#) | |
| UniGaze-L | `configs/model/mae_l_16_gaze.yaml` | [Download](#) | |
| UniGaze-H | `configs/model/mae_h_14_gaze.yaml` | [Download](#) | |


```bash
projdir=<...>/UniGaze/gaze_estimation
cd ${projdir}
model=configs/model/mae_b_16_gaze.yaml 
ckpt_resume=<path to the trained gaze estimator checkpoint>

python draw_predict_video_wild.py \
    --model_cfg_path ${model} \
    -i ./input_video \
    --ckpt_resume ${ckpt_resume}
``` -->


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