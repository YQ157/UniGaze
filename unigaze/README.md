


# Training (Gaze Estimation)

### Model Cards
| Model Name | `model` Config YAML | Checkpoint | Details |
|------------|------------|------------|---------|
| MAE-B16 | `configs/model/mae_b_16_gaze.yaml` | [Download (Google Drive)](https://drive.google.com/drive/folders/1vz38f90jPrMwb_lByzJfaMgH6BEtB49f?usp=sharing) | Backbone: ViT-Base |
| MAE-L16 | `configs/model/mae_l_16_gaze.yaml` | [Download (Google Drive)](https://drive.google.com/drive/folders/1-diS5Ff826wysQeeXiBiDo2bn_5FB3a0?usp=sharing) | Backbone: ViT-Large |
| MAE-H14 | `configs/model/mae_h_14_gaze.yaml` | [Download (Google Drive)](https://drive.google.com/drive/folders/1W-SMVOLhj9PFU3XWrfbMCqY544DI7vtV?usp=sharing) | Backbone: ViT-Huge |


Put them in this structure (refer to their yaml config file)
```
UniGaze/unigaze
    ├── checkpoints/
        ├── mae_b16/
        │   ├── mae_b16_checkpoint-299.pth
        ├── mae_l16/
        │   ├── mae_l16_checkpoint-299.pth
        ├── mae_h14/
        │   ├── mae_h14_checkpoint-299.pth
```





### Create a `data_path.yaml`
This file is ignored so you have to create your own
```
UniGaze/unigaze
    ├── configs/
        ├── data_path.yaml
        ├── <others>
```
It should look like this, please refer to `datasets/xgaze.py` (etc.) for more details
```yaml
data:
    xgaze_v2_224: <path to your data>
    mpii: 
    gazecapture_train_224: .
    gazecapture_test: .
    eyediap_cs: .
    eyediap_ft: .
    gaze360_224_train: .
    gaze360_224_test: .
```

### Experimental Settings

| `data` Config Name | Training Data | Testing Data |
|-------------|--------------|--------------|
| `configs/exp/joint/X_GC_M_ED_g360.yaml` | Joint dataset (Train set) | Joint dataset (Test set)  |
| `configs/exp/cross/train_X.yaml` | XGaze | XGaze test set |

You can create your own train-test settings in a similar way:
- `configs/exp/`: the yaml files here define the overall train/test settings
- `configs/data/`: the yaml files here define the details of the datasets


### Training Instructions

1. Specify the `model` based on the above table
2. Specify the `data` based on the above table
3. Run the training script as follows:


#### Distributed Data Parallel (DDP)

Create a `run.sh` file like below and `sh run.sh`

```bash
projdir=<...>/UniGaze/unigaze
cd ${projdir}
OUTPUT_DIR=${projdir}/logs

exp_name=mae_H14_cross_X ## name can be arbitrary, just for logging

## default
exp=blank.yaml
trainer=configs/trainer/simple_trainer.yaml

model=configs/model/mae_h_14_gaze.yaml
data=configs/exp/cross/train_X.yaml

epochs=12
save_epoch=12

valid_epoch=3
eval_epoch=12

export MASTER_PORT=$((10000 + RANDOM % 10000)) 
echo "MASTER_PORT: ${MASTER_PORT}"

torchrun --nnodes 1 --nproc_per_node 4 --master_port ${MASTER_PORT} main.py \
        use_autocast=True \
        exp=${exp} \
        batch_size=32 \
        test_batch_size=100 \
        epochs=${epochs} \
        valid_epoch=${valid_epoch} \
        eval_epoch=${eval_epoch} \
        save_epoch=${save_epoch} \
        exp.loss=configs/loss/l1_loss.yaml \
        exp.exp_name=${exp_name} \
        exp.model=${model} \
        exp.trainer=${trainer} \
        exp.data=${data} \
        exp.scheduler=configs/schedulers/OneCycleLR.yaml \
```


#### Single GPU Training
```bash
<...>

python main.py \
    use_autocast=True \
    exp=${exp} \
    batch_size=32 \
    test_batch_size=100 \
    epochs=${epochs} \
    valid_epoch=${valid_epoch} \
    eval_epoch=${eval_epoch} \
    save_epoch=${save_epoch} \
    exp.loss=configs/loss/l1_loss.yaml \
    exp.exp_name=${exp_name} \
    exp.model=${model} \
    exp.trainer=${trainer} \
    exp.data=${data} \
    exp.scheduler=configs/schedulers/OneCycleLR.yaml \

```