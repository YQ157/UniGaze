# Masked Autoencoder Pre-training on Normalized Facial Datasets

This repo modifies the original **MAE** implementation to large-scale, normalized face images.  
Only the _dataset_ and _dataloader_ logic have been changed; all MAE core code and hyper-parameters remain intact.


> Make sure to use **timm == 0.3.2** and apply the fix described in this [issue #420](https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842).


## Data Preparation

Pre-processing scripts for CelebV-Text, VGGFace2, VFHQ, FaceSynthetics, and SFHQ-T2I live in [`facedata_preparation/readme`](../facedata_preparation/README.md).  
Each script outputs one or more **224 × 224** h5 files that this repo can read directly.



## Nomralized datasets structure (in h5 file)

Create a yaml file: `MAE/configs/data_path.yaml`, pointing to the directory where the datasets are saved.

```yaml
---
data:
    ## real datasets
    vggface2_224: . # modify to yours
    vfhq_224: .
    celebv_224: .

    ## synthetic datasets
    facesyn_224: .
    sfhq_t2i_224: .
    ffhq_albedo_nv_256: .

    ## XGaze synthetic dataset
    xgaze_mvs_dense_224: .
```

where each dataset has structures:
```
vggface2_224:
    group_0.h5
    group_1.h5
    group_2.h5
    group_3.h5
    ...
    group_17.h5
    group_test.h5
vfhq_224:
    part1.h5
    part2.h5
    part3.h5
    part4.h5
    part5.h5
    part6.h5
celebv_224:
    part_1.h5
    part_2.h5
    part_3.h5
    part_4.h5
    part_5.h5
    part_6.h5
    part_7.h5
sfhq_t2i_224:
    sfhqt2i_224.h5
facesyn_224:
    FaceSynthetics_224.h5
```

For more details, please refer to the datasets defined in `MAE/datasets`.

### More datasets
You can create your custom dataset class as long as the iteration of `__getitem__` returns
```python
def __getitem__(self,index):
    ...
    entry = {
        'image': self.preprocess_image(image),
        'gaze': np.array([0,0]).astype('float'),
        'head': head_label,
    }
    return entry
```

## Run MAE

### Single GPU
```bash

cd ./MAE

vgg2_yaml=configs/data/vggface2_224.yaml
vfhq_yaml=configs/data/vfhq_224.yaml
celebv_yaml=configs/data/celebv_224.yaml
facesyn_yaml=configs/data/facesyn_224.yaml
sfhq_t2i_224_yaml=configs/data/sfhq_t2i_224.yaml
# ffhqnv_yaml=configs/data/ffhqnv_256.yaml
# xgaze_mvs_dense_224_yaml=configs/data/xgaze_mvs_dense_224.yaml

data_yamls="${vfhq_yaml} ${celebv_yaml} ${vgg2_yaml} ${facesyn_yaml} ${sfhq_t2i_224_yaml}"
output_dir=./logs

python main_pretrain.py \
    --batch_size 64 \
    --accum_iter 8 \
    --num_workers 32 \
    --pin_mem \
    --model mae_vit_huge_patch14 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_yamls ${data_yamls} \
    --output_dir ${output_dir} 
```

### Multiple GPU (DDP)
```bash
torchrun --nnodes 1 --nproc_per_node 4 main_pretrain.py \
    --batch_size 64 \
    --accum_iter 8 \
    --num_workers 32 \
    --pin_mem \
    --model mae_vit_huge_patch14 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_yamls ${data_yamls} \
    --output_dir ${output_dir} 
```