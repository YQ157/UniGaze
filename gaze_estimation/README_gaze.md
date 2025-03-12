


# Training (Gaze Estimation)

### Model Cards
| Model Name | Config YAML | Checkpoint | Details |
|------------|------------|------------|---------|
| UniGaze-B | `configs/model/unigaze_b.yaml` | [Download](#) | Small backbone, efficient model |
| UniGaze-L | `configs/model/unigaze_l.yaml` | [Download](#) | Large backbone, improved accuracy |
| UniGaze-H | `configs/model/unigaze_h.yaml` | [Download](#) | High-capacity model, best accuracy |



### Experimental Settings
| Config Name | Training Data | Testing Data |
|-------------|--------------|--------------|
| `train_joint.yaml` | Joint dataset | Multiple benchmarks |
| `train_xgaze.yaml` | XGaze | XGaze test set |
| `train_mpii.yaml` | MPII | MPII test set |

### Training Instructions

1. Place the **pre-trained MAE** checkpoints into `./checkpoints`.
2. Specify the **model config name**.
3. Specify the **experimental setting config name**.
4. Run the training script as follows:


#### Distributed Data Parallel (DDP)
```bash
projdir=<...>/UniGaze/gaze_estimation
cd ${projdir}
OUTPUT_DIR=${projdir}/logs
exp=blank.yaml
exp_name=mae_H14_cross_X
model=configs/model/mae/mae_vit_h_14_pretrain_1023/299epoch.yaml
data=configs/exp/1026_cross/train_X.yaml
trainer=configs/trainer/simple_trainer.yaml

epochs=6
save_epoch=6
valid_epoch=3
eval_epoch=6
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