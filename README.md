# Object Detection and Instance Segmentation

**Ruihan Wu**  
Fudan University  
Email: [22307140084@m.fudan.edu.cn](mailto:22307140084@m.fudan.edu.cn)

---


## Overview 🗂

This repository aims to implement **object detection** and **instance segmentation** tasks using models from the [MMDetection](https://github.com/open-mmlab/mmdetection) framework, specifically focusing on **Mask R-CNN** and **Sparse R-CNN**. 🛠 And the project is designed to utilize the **Pascal VOC 2007** dataset as the benchmark for training and evaluation. 

This repository includes the following files:

- `mask-rcnn_r50_fpn_ms-poly-2x_voc.py`: Configuration file used for the **Mask R-CNN** model.
- `sparse-rcnn_r50_fpn_1x_voc.py`: Configuration file used for the **Sparse R-CNN** model.
- `split.py`: Script for splitting the dataset, designed to convert Pascal VOC data into the COCO dataset format.
- `visualize_rpn_vs_final.py`: A script for visualizing the proposal boxes (from the RPN) and the final results of the **Mask R-CNN** model.
- `class_names.py`, `coco.py`, `two_stage.py`: Adaptations of MMDetection framework files to work with the Pascal VOC dataset and its specific categories.


---

**Resource Access** 🔗

- Processed Pascal VOC dataset, training model weights, and visualization results can be downloaded via the following Baidu Cloud link:
  - **Link**: [https://pan.baidu.com/s/1_TR6pxyXcq4oo3VLtJxxTQ](https://pan.baidu.com/s/1_TR6pxyXcq4oo3VLtJxxTQ)
  - **Password**: `txkp`



Stay tuned for results and findings! 🚀

---

## Implementation ⚙️

## Training 🏋️

## Test 🧪
