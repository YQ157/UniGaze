# UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training
> [[arxiv]](https://arxiv.org/pdf/2502.02307)

<a href="https://jqin-home.github.io/">Jiawei Qin</a><sup>1</sup>, 
<a href="https://www.ccmitss.com/zhang">Xucong Zhang</a><sup>2</sup>, 
<a href="https://www.yusuke-sugano.info/">Yusuke Sugano</a><sup>1</sup>, 

*<sup>1</sup>The University of Tokyo, <sup>2</sup>Computer Vision Lab, Delft University of Technology 




<!-- <h4 align="left">
<a href="">Project Page</a>
</h4> -->



## Overview
This repository contains the official PyTorch implementation of both **MAE pre-training** and **unigaze**.



### Todo:
- :white_check_mark: Release pre-trained MAE checkpoints (B, L, H) and gaze estimation training code.
- :white_check_mark: Release UniGaze models for inference.
- :white_check_mark: Code for predicting gaze from videos
- Gaze estimation demo.
- Release the MAE pre-training code.

---


## Installation


To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Pre-training (MAE)
*(Coming soon)*

---


## Training (Gaze Estimation)

For detailed training instructions, please refer to [UniGaze Training](./unigaze/README.md).

---

## Usage of UniGaze


### Available Models

We provide the following trained models:

|   Filename   | Backbone |   Training Data   | Checkpoint |
|--------------|----------|-------------------|------------|
|`unigaze_b16_joint.pth.tar`  | UniGaze-B | *Joint Datasets* | [Download (Google Drive)](https://drive.google.com/file/d/1xdPbzAX8d3cPAMChFjRThryIWVp9Ng_f/view?usp=sharing) |
|`unigaze_L16_joint.pth.tar`  | UniGaze-L | *Joint Datasets* | [Download (Google Drive)](https://drive.google.com/file/d/1JR20_iGTU8pSXtKIC-_swiSRImWLAbBC/view?usp=sharing) |
|`unigaze_h14_joint.pth.tar`  | UniGaze-H | *Joint Datasets* | [Download (Google Drive)](https://drive.google.com/file/d/16z_Y8_yi53xTw_-5Pw9H4jjAOPebIFdA/view?usp=sharing) | 
|`unigaze_h14_cross_X.pth.tar`| UniGaze-H |  ETH-XGaze       | [Download (Google Drive)](https://drive.google.com/file/d/1BVYGOK5NwXUPr63DnbYGeQ_yqlevv9VR/view?usp=sharing) |



### Loading Pretrained Models
- Download the pretrained models and place them in `./unigaze/logs`.
- For instructions on how to load and use these models, please refer to [load_gaze_model.ipynb](./unigaze/load_gaze_model.ipynb).

### Predicting Gaze from Videos
To predict gaze direction from videos, use the following script:

```bash
projdir=<...>/UniGaze/unigaze
cd ${projdir}
python predict_gaze_video.py \
    --model_cfg_path configs/model/mae_b_16_gaze.yaml  \
    -i ./input_video \
    --ckpt_resume logs/unigaze_b16_joint.pth.tar
``` 


---

# Citation
If you find our work useful for your research, please consider citing:

```
@article{qin2025unigaze,
  title={UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training},
  author={Qin, Jiawei and Zhang, Xucong and Sugano, Yusuke},
  journal={arXiv preprint arXiv:2502.02307},
  year={2025}
}
```
We also acknowledge the excellent work on [MAE](https://github.com/facebookresearch/mae.git).



## Contact
If you have any questions, feel free to contact Jiawei Qin at jqin@iis.u-tokyo.ac.jp.
