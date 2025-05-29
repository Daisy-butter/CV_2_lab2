# 🧠 Object Detection and Instance Segmentation

**Author:** Ruihan Wu  
**Affiliation:** Fudan University  
📧 [22307140084@m.fudan.edu.cn](mailto:22307140084@m.fudan.edu.cn)

---

## 📌 Overview

This repository implements **object detection** and **instance segmentation** using the [MMDetection](https://github.com/open-mmlab/mmdetection) framework. Models used include **Mask R-CNN** and **Sparse R-CNN**, evaluated on the **Pascal VOC 2007** dataset.

### 📁 Contents

- `mask-rcnn_r50_fpn_ms-poly-2x_voc.py`: Mask R-CNN configuration file
- `sparse-rcnn_r50_fpn_1x_voc.py`: Sparse R-CNN configuration file
- `split.py`: Dataset splitting script
- `visualize_rpn_vs_final.py`: Visualize RPN proposals vs final detections
- `class_names.py`, `coco.py`, `two_stage.py`: Modified MMDetection components for VOC categories

🔗 **Resources (Baidu Cloud):**  
Processed Pascal VOC dataset, training model weights, and visualization results can be downloaded via the following Baidu Cloud link.

[Download Link](https://pan.baidu.com/s/1napdHewzDzCgoMz3b_7kMg?pwd=di9q) | **Password:** `di9q`

---

## ⚙️ Implementation Details

### 1. 📦 Environment

- Python: `3.9.21`
- MMDetection: `3.3.0`
- MMCV: `2.1.0`
- PyTorch: `2.1.0+cu121`
- CUDA: `12.9`

### 2. 📊 Data Processing

- Convert VOC XML to COCO JSON using `tools/dataset_converters/pascal_voc.py`
- Reorganize directory and split using `split_data.py` (train:val:test = 8:1:1)
- Final dataset path: `mmdetection/data/coco`

### 3. 🛠 Custom Modifications

- **`coco.py`**: Update `CocoDataset` class with VOC class names & palette
- **`class_names.py`**: Update `coco_classes()` with VOC class list
- **`two_stage.py`**: Modify to extract stage 1 RPN proposals

![Proposal Boxes](README_images/proposal_box.png)

### 4. ⚙️ Configuration Snippets

#### 🧾 Dataset Loader

```python
DATASET_TYPE = 'CocoDataset'
DATA_ROOT = 'data/coco/'

train_dataloader = dict(
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    dataset = dict(
        type = DATASET_TYPE,
        ann_file = 'annotations/instances_train2017.json',
        data_prefix = dict(img='train2017/'),
        data_root = DATA_ROOT,
        filter_cfg = dict(filter_empty_gt=True, min_size=32)
    ),
    sampler = dict(type='DefaultSampler', shuffle=True)
)
```

#### 🗒️ Logging

```python
default_hooks = dict(
    logger = dict(type='LoggerHook', interval=10),
    checkpoint = dict(type='CheckpointHook', interval=1)
)
log_processor = dict(
    type = 'LogProcessor',
    by_epoch = True,
    window_size = 50
)
```

#### 📈 Visualization

```python
visualizer = dict(
    vis_backends = [dict(type='TensorboardVisBackend')]
)
```

#### 📂 Output Directory

```python
work_dir = 'work_dirs_maskrcnn/sparsercnn'
```

---

## 🏋️‍♀️ Training

Run local training using:

```bash
python tools/train.py <CONFIG> [options]
```

### Arguments

- `<CONFIG>`: Path to config file  
  e.g., `configs/experiments/mask-rcnn_r50_fpn_ms-poly-2x_voc.py`

#### Optional

- `--work-dir`: Custom working directory
- `--resume-from`: Resume from a checkpoint
- `--no-validate`: Skip validation
- `--gpus`: Number of GPUs (default: 1)
- `--seed`: Random seed for reproducibility
- `--deterministic`: Enforce deterministic behavior
- `--launcher`: Launch method (`none`, `pytorch`, `slurm`, `mpi`)

Multi-GPU support: [pytorch-multi-gpu-training](https://github.com/jia-zhuang/pytorch-multi-gpu-training.git)

---

## 🧪 Testing

Evaluate a trained model using:

```bash
python tools/test.py <CONFIG> <CHECKPOINT> [options]
```

### Required

- `<CONFIG>`: Configuration file path  
- `<CHECKPOINT>`: Trained model checkpoint

#### Optional

- `--out`: Output file for results (e.g., `results.pkl`)
- `--eval`: Evaluation metrics (`bbox`, `segm`, etc.)
- `--gpu-ids`: GPU ID(s)
- `--show`: Visualize results
- `--show-dir`: Output directory for visualizations
- `--cfg-options`: Override config options
- `--launcher`, `--local_rank`: Distributed config

---

## 🖼️ Single-image Inference

Perform inference on a single image with:

```bash
python demo/image_demo.py <IMAGE_PATH> <CONFIG> <CHECKPOINT> [options]
```

Options are similar to the test script.

---

## 🙏 Acknowledgements

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [COCO Dataset](https://cocodataset.org/#home)
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
