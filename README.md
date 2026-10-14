# MCG-DETR

Research implementation for **MCG-DETR: Local Feature Adaptation and Gated Multiscale Fusion for X-ray Prohibited Item Detection**, based on RT-DETR.

## Implemented components

- **MCD-Adapter**: material--contour decoupling adapter for local feature adaptation.
- **CSGMSF**: channel--spatial gated multiscale fusion in the hybrid encoder.
- **WIoU**: a bounding-box regression loss used by the RT-DETR criterion.

Main implementations: `src/zoo/rtdetr/hybrid_encoder.py` and `src/zoo/rtdetr/rtdetr_criterion.py`.

## Set up

Create a Python environment with a compatible PyTorch and CUDA configuration, then install the listed dependencies:

```bash
pip install -r requirements.txt
```

## Data

Download OPIXray and CLCXray from their respective official sources (not included in this repository). The dataset YAML files expect local COCO-format images and annotations under `dataset/`:

```text
dataset/
  OPIXray/
    train/train_image/
    test/test_image/
    annotations/opixray_train.json
    annotations/opixray_test.json
  CLCXray/
    train2017/
    val2017/
    annotations/instances_train2017.json
    annotations/instances_val2017.json
```

Adjust `configs/dataset/coco_detection_opixray.yml` and `configs/dataset/coco_detection_clcxray.yml` if the dataset locations or splits differ. The separate `PIXray_coco` YAML is a legacy/base configuration.

## Training

For OPIXray:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_opixray.yml
```

For CLCXray:

```bash
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_clcxray.yml
```

Configurations may refer to external pretrained weights; network connectivity may be needed, and paths/hyperparameters should be checked against your experiment settings before reproducing reported results.

## Notes

- Dataset contents, pretrained weights, trained checkpoints, and experiment outputs are not included.
- A filename mismatch in `src/zoo/rtdetr/__init__.py` was corrected in this packaged copy so it imports the included `hybrid_encoder.py`.
- The code is provided as received with packaging cleanups; model training and numerical reproduction have **not** been run or verified here.
- Review upstream RT-DETR licensing and attribution requirements before making this repository public.
