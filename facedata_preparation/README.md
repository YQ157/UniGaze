# Face-Dataset Normalization Pipelines

This folder hosts the **pre-processing / normalization code** used by **UniGaze** for several large-scale face-image datasets.

| Dataset | Official Source / Mirror |
|---------|--------------------------|
| CelebV-Text | <https://github.com/celebv-text/CelebV-Text/issues/8> | 
| VGGFace2 |  [AcademicTorrents] and [Kaggle] | 
| VFHQ | “VFHQ_zips (2.8 TB)” on the [VFHQ project page] | `main_vfhq.py` 
| FaceSynthetics | <https://github.com/microsoft/FaceSynthetics> | 
| SFHQ-T2I |  <https://github.com/SelfishGene/SFHQ-T2I-dataset> (or [Kaggle]) | 

<div style="font-size:0.9em">

[AcademicTorrents]: https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b  
[Kaggle]: https://www.kaggle.com/datasets/hearfool/vggface2  
[VFHQ project page]: https://liangbinxie.github.io/projects/vfhq/  
[Kaggle]: https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models?resource=download
</div>

> **For novel-view rendered datasets?**  
> - FFHQ-NV &nbsp;→ see **[AlbedoGAN]** and **[Gaze-NV-Rendering]**  
> - XGaze-Dense → see **[XGaze3D]**

[AlbedoGAN]: https://github.com/aashishrai3799/Towards-Realistic-Generative-3D-Face-Models/
[Gaze-NV-Rendering]: https://github.com/ut-vision/Gaze-NV-Rendering  
[XGaze3D]: https://github.com/ut-vision/XGaze3D


## Download original datasets from their respective webstie
You may do not want to bother trying the long-time reconstruction and rendering process, you can also add any new facial datasets you want. 
Just follow a similar normalization process.

## Run the below scripts
The main parts of these scripts are almost the same, they only differ in how to load the original data and how to save them,


## 1. CelebV-Text
#### Input
Folder layout of the original dataset (keep archives **compressed**):
```yaml
<celebv_text_dir>/
    readme.txt
    video/
        sp0000.tar
        …
        sp0069.tar
    audio/ # not used
```
#### run script
```bash
cd UniGaze/facedata_preparation
python main_celeb_v.py \
  --input_data  <celebv_text_dir> \
  --output_dir  <your-output-dir> \
  --img_sample_rate 15        # sample 1 frame every 15
```
#### output
Originally it has 70 tar files, we split them to 7 parts, each of which will be saved to an h5 file.
```yaml
<your-output-dir>/
    part_1.h5
    part_2.h5
    part_3.h5
    part_4.h5
    part_5.h5
    part_6.h5
    part_7.h5
```


## 2. VFHQ
#### Input
Folder layout of the original dataset, you don't need to unzip them.
```yaml
<vfhq_dir>\
    part1: 30 zip files, group100.zip ~ group129.zip
    part2: 30 zip files, group130.zip ~ group159.zip
    part3: 30 zip files, group160.zip ~ group206.zip
    part4: 30 zip files, group207.zip ~ group236.zip
    part5: 30 zip files, group237.zip ~ group323.zip
    part6: 31 zip files, group324.zip ~ group422.zip
```


#### run script
```bash
for p in 1 2 3 4 5 6 
do 
python main_vfhq.py \
    --input_dir <vfhq_dir>/part${p} \
    --output_dir <your-output-dir> \
    --img_sample_rate 15  # sample 1 frame every 15

done
```
#### output
```yaml
<your-output-dir>/
    part1.h5
    part2.h5
    part3.h5
    part4.h5
    part5.h5
    part6.h5
```


## 3. VGGFace2
#### Input
```yaml
<vggface2_dir>/
  data/
  meta/
  samples/
    train/
        n000002/*.jpg
        ...
        n009279/*.jpg
        …
    test/
        n000001/*.jpg
        ...
        n009294/*.jpg
        …
```
#### run script
```bash
vggface2_dir=<vggface2_dir>
for p in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
do 
python main_vggface2.py \
    --input_dir ${vggface2_dir}/samples/train \
    --supp_data ./vggface2_indices/group_${p}.yaml \
    --num_per_subject 20 \
    --output_dir <your-output-dir> 
    
done

python main_vggface2.py \
    --input_dir ${vggface2_dir}/samples/test \
    --supp_data ./vggface2_indices/group_test.yaml \
    --num_per_subject 20 \
    --output_dir <your-output-dir> 

```

#### output
```yaml
<your-output-dir>/
    group_0.h5
    group_1.h5
    group_2.h5
    group_3.h5
    ...
    group_17.h5
    group_test.h5
```


## 4. FaceSynthetics

Download the full dataset of 100,000 images (32GB): `dataset_100000.zip`.
#### run script
```bash
python main_face_syn.py \
  --input_path  <facesyn_dir>/dataset_100000.zip \
  --output_dir  <your-output-dir>
```
The output will be `<your-output-dir>/FaceSynthetics_224.h5`.

## 5. SFHQ-T2I
Download the `archive.zip`.
#### run script
```bash
python main_sfhq_t2i.py \
  --input_path  <sfhqt2i_dir>/archive.zip \
  --output_dir  <your-output-dir>
```
The output will be `<your-output-dir>/sfhqt2i_224.h5`.


---
