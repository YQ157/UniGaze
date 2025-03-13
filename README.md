# UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training

> UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training

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
This repository contains the official PyTorch implementation of both **MAE pre-training** and **unigaze**.



### Todo:
- :white_check_mark: Release pre-trained MAE checkpoints (B, L, H) and gaze estimation training code.
- :white_check_mark: Release UniGaze models for inference.
- :white_check_mark: Code for predicting gaze of videos
- Gaze estimation demo.
- Release the MAE pre-training code.

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
Refer to [UniGaze training.md](./unigaze/README.md)


## Usage of UniGaze


### Available Models
We provide below models.

|   Filename   | Backbone |   Training Data   | Checkpoint |
|--------------|----------|-------------------|------------|
|`unigaze_b16_joint.pth.tar`  | UniGaze-B | *Joint Datasets* | [Download (Google Drive)](https://drive.google.com/file/d/1xdPbzAX8d3cPAMChFjRThryIWVp9Ng_f/view?usp=sharing) |
|`unigaze_L16_joint.pth.tar`  | UniGaze-L | *Joint Datasets* | [Download (Google Drive)](https://drive.google.com/file/d/1JR20_iGTU8pSXtKIC-_swiSRImWLAbBC/view?usp=sharing) |
|`unigaze_h14_joint.pth.tar`  | UniGaze-H | *Joint Datasets* | [Download (Google Drive)](https://drive.google.com/file/d/16z_Y8_yi53xTw_-5Pw9H4jjAOPebIFdA/view?usp=sharing) | 
|`unigaze_h14_cross_X.pth.tar`| UniGaze-H |  ETH-XGaze       | [Download (Google Drive)](https://drive.google.com/file/d/1BVYGOK5NwXUPr63DnbYGeQ_yqlevv9VR/view?usp=sharing) |

### Loading Models
Download and put these models under `./unigaze/logs`.
Please refer to [load_gaze_model.ipynb](./unigaze/load_gaze_model.ipynb) for loading the model.

### Predicting Gaze from videos

```bash
projdir=<...>/UniGaze/unigaze
cd ${projdir}
model=configs/model/mae_b_16_gaze.yaml 
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