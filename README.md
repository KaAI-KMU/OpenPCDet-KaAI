# OpenPCDet with KaAI dataset

OpenPCDet에서 KaAI dataset을 이용해 모델을 훈련시킬 수 있도록 구현하였습니다.<br/><br/>

## Prepare for using KaAI dataset

### Creat data infos
```
python -m pcdet.datasets.kaai.kaai_dataset create_kaai_infos_w/o_gt_database tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### Train
```
python tools/train.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}
```

## Pre-Annotation for KaAI dataset

### ImageSet generation
    python -m pcdet.datasets.kaai.kaai_dataset generate_imagesets tools/cfgs/dataset_configs/kaai_dataset.yaml 1

### Creat data infos
    python -m pcdet.datasets.kaai.kaai_dataset create_kaai_infos_w/o_gt_database tools/cfgs/dataset_configs/kitti_dataset.yaml

### Run pre-annotation
```
python tools/pre_annotation.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}
```
    # example
    python tools/pre_annotation.py --cfg_file cfgs/kaai_models/pv_rcnn.yaml --ckpt pv_rcnn_80_epochs.pth

원하는 모델과 checkpoint를 쓰고 위 코드를 터미널에 입력하면 됩니다.
checkpoint는 [OpenPCDet 원본 깃허브](https://github.com/open-mmlab/OpenPCDet#model-zoo)에서 다운받을 수 있습니다.
다운받았거나 훈련한 모델의 checkpoint는 OpenPCDet-KaAI/tools에 넣습니다.

3/17 : 현재는 pv_rcnn.yaml만 제공합니다
