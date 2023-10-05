# Linear-Covariance Loss for End-to-End Learning of 6D Pose Estimation
## Requirements:
pytorch>=1.12.1, python>=3.8 and use following commands to setup other depends
```
# root priviledge might be required for some commands
bash scripts/install-depends.sh        # dependencies, glfw, assimp and python depends
bash scripts/build-ceres.sh            # install ceres dependencies and build ceres-solver, will leave a "/ceres-build", which can be safely deleted
                                       # if failed to build ceres, please refer to their official guides
python lib/pnp/setup_ceres.py          # compile extension for weighted PnP solver
```

## Dataset structure:
Download dataset from the [BOP website](https://bop.felk.cvut.cz/datasets/) and background images (VOCtrainval_11-May-2012.tar) from [VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).

The structure should looks like:
```
.
├──datasets:
   ├──VOCdevkit/VOC2012/JPEGImages # VOC images
   ├──BOP_DATASETS
      ├──lm
         ├──test
      ├──lmo   # or ycbv
         ├──models
         ├──models_eval
         ├──test
         ├──train_real # for lmo, it's a soft link to lm/test
            ├──......
            ├──z_crop  # generated with gen_z.py
         ├──train_pbr
            ├──......
            ├──z_crop  # generated with gen_z.py
```

Generate depth patches:
```
python tools/gen_z.py --dataset lmo --data_dir datasets/BOP_DATASETS/lmo/train_pbr
python tools/gen_z.py --dataset lmo --data_dir datasets/BOP_DATASETS/lmo/train_real

python tools/gen_z.py --dataset ycbv --data_dir datasets/BOP_DATASETS/ycbv/train_pbr
python tools/gen_z.py --dataset ycbv --data_dir datasets/BOP_DATASETS/ycbv/train_real
```

A "z_crop" folder will be created in data_dir after this step, as listed before.

download pretrained resnet34-333f7ec4.pth to ./assets
```
wget --directory-prefix assets https://download.pytorch.org/models/resnet34-333f7ec4.pth
```

## train:
```
python train.py --config PATH_TO_CONFIG --obj OBJECT_ID --output OUTPUT_DIR
```
Example:
```
python train.py --config configs/glmo.yaml --obj 1 --output output
```

## test:
```
python test.py --config PATH_TO_TRAINING_CONFIG --obj OBJECT_ID --weight TRAINED_WEIGHT_PATH --output OUTPUT_DIR
```
Example:
```
python test.py --config configs/glmo.yaml --obj 1 --weight assets/models/glmo/01/model.pth --output output
```
## models:
[Google drive](https://drive.google.com/drive/folders/1twkJ4oNdBAFXA8rMpv9uzwFG0bI5MxPt?usp=drive_link)


