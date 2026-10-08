# Multi-Scale Convolution Enhanced YOLO26m for Steel Surface Defect Detection

A deep learning project for automated steel surface defect detection using
YOLO-based object detection, with a **Multi-Scale Convolution (MSC) enhancement
applied to YOLO26m**.

---

## Overview

Automated inspection of steel surfaces is important for identifying defects
during manufacturing. Manual inspection can be time-consuming and may be
inconsistent, making computer vision-based inspection a useful alternative.

This project investigates YOLO-based object detection models for detecting
four types of steel surface defects:

- Pitted Surface
- Crazing
- Scratches
- Patches

The project follows a baseline-to-proposed-model approach:

```text
Severstal Steel Defect Dataset
            │
            ▼
      Data Preprocessing
            │
            ▼
      YOLO Dataset Format
            │
            ├──────────────┐
            ▼              ▼
         YOLOv8m        YOLO11m
         Baseline       Baseline
            │              │
            └──────┬───────┘
                   ▼
                YOLO26m
                Baseline
                   │
                   ▼
          Multi-Scale Convolution
                   │
                   ▼
          YOLO26m + MSC
          Proposed Model
                   │
          ┌────────┴─────────┐
          ▼                  ▼
   Quantitative          Explainability
    Evaluation             Analysis
          │                  │
          │             Grad-CAM++-style
          │                  │
          ▼                  ▼
     P/R/mAP metrics      Heatmaps
                   
                   +
                   
             Qwen2.5-VL
       Auxiliary Explanation
```

---

## Dataset

### Severstal: Steel Defect Detection

The project uses the **Severstal Steel Defect Detection** dataset.

The original dataset contains steel surface images with Run-Length Encoded
(RLE) segmentation annotations.

Original image resolution:

```text
1600 × 256 pixels
```

The RLE segmentation annotations are processed and converted into
YOLO-compatible bounding-box annotations.

### Defect Classes

| YOLO Class ID | Defect |
|---:|---|
| 0 | Pitted Surface |
| 1 | Crazing |
| 2 | Scratches |
| 3 | Patches |

---

## Data Preprocessing

The preprocessing pipeline consists of:

1. RLE mask decoding
2. Four-class mask construction
3. Train-validation splitting
4. Image tiling
5. Connected-component based bounding-box extraction
6. Bounding-box filtering
7. Conversion to YOLO annotation format
8. YOLO dataset generation

### Tiling Configuration

| Parameter | Value |
|---|---:|
| Tile width | 608 pixels |
| Tile positions | 0, 495, 991 |
| Bounding-box padding | 3 pixels |

### Dataset Split

| Dataset | Images |
|---|---:|
| Original dataset | 12,568 |
| Training images | 10,054 |
| Validation images | 2,514 |

The generated YOLO dataset contains:

| Dataset | Tiles |
|---|---:|
| Training | 30,162 |
| Validation | 7,542 |

---

## Models

The project evaluates multiple YOLO architectures:

- YOLOv4
- YOLOv8m
- YOLO11m
- YOLO26m
- YOLO26m + MSC

YOLOv8m, YOLO11m, and YOLO26m are used as the main modern YOLO baselines.

YOLO26m is selected as the baseline for the proposed architectural
modification.

---

## Proposed Method

### YOLO26m with Multi-Scale Convolution

The proposed model is:

> **YOLO26m + Multi-Scale Convolution (MSC)**

One intermediate 3×3 convolutional layer in the YOLO26m backbone is replaced
with a Multi-Scale Convolution module.

The modified layer maintains the original **128-channel output**.

### MSC Architecture

The MSC module consists of two parallel convolution branches:

```text
                     Input
                  128 channels
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
         3 × 3 Conv           5 × 5 Conv
         64 channels           64 channels
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
                  Concatenate
                  128 channels
                       │
                       ▼
                     Output
```

Both branches receive the complete input feature map.

The mathematical formulation is:

```text
F₃ = Conv₃×₃(X)

F₅ = Conv₅×₅(X)

FMSC = Concat(F₃, F₅)
```

### Motivation

Different steel defects can have different spatial characteristics.

The two convolution branches provide complementary receptive fields:

- **3×3 convolution** captures finer local features.
- **5×5 convolution** captures broader spatial features.
- **Concatenation** combines both feature representations.

The modification is intentionally localized to one intermediate layer so that
the effect of multi-scale feature extraction can be evaluated against the
original YOLO26m architecture.

---

## Experimental Configuration

The main YOLO26m experiments use:

| Parameter | Value |
|---|---|
| Image size | 800 × 800 |
| Batch size | 8 |
| Epochs | 25 |
| Optimizer | MuSGD |
| Initial learning rate | 0.01 |
| Final learning-rate factor | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| AMP | Enabled |
| Early stopping patience | 10 |

The same general training configuration is used for the YOLO26m baseline and
the proposed YOLO26m + MSC experiment.

---

## Evaluation Metrics

The models are evaluated using:

- **Precision**
- **Recall**
- **mAP@50**
- **mAP@50-95**

### Precision

Precision measures how many of the predicted detections are correct.

### Recall

Recall measures how many of the actual defects are successfully detected.

### mAP@50

Mean Average Precision using an IoU threshold of 0.50.

### mAP@50-95

Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95 in
steps of 0.05.

This provides a stricter evaluation of localization quality.

---

# Results

## Overall Comparison

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| YOLOv8m | 65.75% | 63.00% | 68.49% | 40.52% |
| YOLO11m | 70.03% | 67.52% | 73.99% | 43.09% |
| YOLO26m | 72.66% | 67.84% | 75.54% | 46.01% |
| **YOLO26m + MSC** | **74.40%** | **71.00%** | **76.90%** | **45.90%** |

### Key Findings

Compared with the YOLO26m baseline, the proposed YOLO26m + MSC model achieved:

| Metric | YOLO26m | YOLO26m + MSC | Change |
|---|---:|---:|---:|
| Precision | 72.66% | **74.40%** | **+1.74 pp** |
| Recall | 67.84% | **71.00%** | **+3.16 pp** |
| mAP@50 | 75.54% | **76.90%** | **+1.36 pp** |
| mAP@50-95 | 46.01% | 45.90% | -0.11 pp |

The proposed model achieved the highest Precision, Recall, and mAP@50 among
the evaluated models.

The mAP@50-95 score remained comparable to the YOLO26m baseline, with a
difference of only 0.11 percentage points.

---

## Class-wise Results

### YOLO26m + MSC

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Pitted Surface | 70.9% | 63.6% | 68.9% | 34.6% |
| Crazing | 78.4% | 75.4% | 81.0% | 47.9% |
| Scratches | 75.2% | 73.3% | 80.6% | 50.7% |
| Patches | 72.9% | 71.7% | 77.1% | 50.3% |
| **Overall** | **74.4%** | **71.0%** | **76.9%** | **45.9%** |

The highest class-wise mAP@50 was obtained for **Crazing (81.0%)**, followed
by **Scratches (80.6%)** and **Patches (77.1%)**.

Pitted Surface achieved a mAP@50 of 68.9%.

---

# Explainable AI

## Grad-CAM++-style Visualization

A practical **Grad-CAM++-style approximation** is implemented to provide
visual insight into the feature regions used by the proposed model.

The XAI pipeline is:

```text
Input Steel Image
       │
       ▼
YOLO26m + MSC
       │
       ▼
Intermediate Feature Layer
       │
       ▼
Forward Activations
       +
Backward Gradients
       │
       ▼
Gradient-based Feature Weighting
       │
       ▼
Heatmap Generation
       │
       ▼
Heatmap Overlay
```

The visualization highlights regions of the image associated with the
model's internal feature representations.

The final visualization does not require detection bounding boxes.

### XAI Target

The primary Grad-CAM target is a high-level intermediate feature extraction
layer close to the detection head.

The MSC layer can also be inspected when specifically analyzing the effect of
the proposed architectural modification.

### XAI Note

The implementation is described as a **practical Grad-CAM++-style
approximation**.

It is used for qualitative interpretation and is not used to calculate
Precision, Recall, mAP@50, or mAP@50-95.

---

# Vision-Language Model

## Qwen2.5-VL-3B-Instruct

The project additionally integrates **Qwen2.5-VL-3B-Instruct** as an
auxiliary Vision-Language Model (VLM).

The purpose of the VLM is to provide a natural-language interpretation of
the visible steel defect.

### VLM Pipeline

```text
Steel Image
     │
     ▼
Vision Encoder
     │
     ▼
Visual Features
     │
     +
     │
Text Prompt
     │
     ▼
Qwen2.5-VL
     │
     ▼
Natural-language Explanation
```

The VLM is prompted to identify the visible defect from the four project
classes:

```text
Pitted Surface
Crazing
Scratch
Patch
No confident defect
```

### Role of the VLM

The VLM is an **auxiliary explanation component**.

It is not used for:

- Training the YOLO detector
- Precision calculation
- Recall calculation
- mAP@50 calculation
- mAP@50-95 calculation

The YOLO26m + MSC model remains the primary quantitative detection model.

---

# YOLOv4

YOLOv4 was investigated as part of the model progression and baseline
study.

The intended configuration followed the original YOLOv4/Darknet approach,
including:

```text
Image size : 640 × 640
Optimizer  : Adam
Initial LR : 0.0001
Maximum epochs : 25
```

The YOLOv4 reproduction encountered implementation and environment
compatibility issues during development, including CUDA/cuDNN configuration
and compilation problems.

Therefore, YOLOv4 is not included in the final quantitative comparison
table.

---

# Checkpoints

Model checkpoints are saved during training.

Two important checkpoints are:

### `best.pt`

Contains the model state associated with the best validation performance
during training.

This checkpoint is used for final validation and inference.

### `last.pt`

Contains the model state from the final training epoch.

It can be used to resume training or inspect the final training state.

Trained weights are not included in this repository.

---

# Repository Structure

```text
steel-surface-defect-detection/
│
├── models/
│   └── msc_module.py
│
├── training/
│   ├── modify_yolo26.py
│   └── train_msc.py
│
├── xai/
│   └── gradcam_pp.py
│
├── vlm/
│   └── qwen_vlm.py
│
├── configs/
│   └── data.yaml
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd steel-surface-defect-detection
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

The main Python dependencies are:

```text
ultralytics==8.4.173
torch==2.5.1
torchvision
opencv-python
numpy
matplotlib
transformers
accelerate
qwen-vl-utils
```

GPU-enabled PyTorch installation may need to be adjusted according to the
CUDA version and GPU available on the target system.

---

# Dataset Configuration

The project uses an Ultralytics YOLO YAML configuration.

Example:

```yaml
path: /path/to/dataset

train: images/train
val: images/val

names:
  0: pitted_surface
  1: crazing
  2: scratches
  3: patches
```

The dataset path must be changed according to the local environment.

The actual dataset and generated images/labels are not included in this
repository.

---

# Training Workflow

The proposed model is generated by modifying the YOLO26m architecture.

```text
YOLO26m Baseline Checkpoint
          │
          ▼
      Load Model
          │
          ▼
Locate Target Intermediate Layer
          │
          ▼
Replace Original 3×3 Conv
          │
          ▼
Insert MSC Module
          │
          ▼
Save Modified Model
          │
          ▼
Train YOLO26m + MSC
          │
          ▼
Save Checkpoints
          │
          ▼
Validate Proposed Model
```

The repository contains the corresponding model modification and training
scripts.

---

# Reproducibility

To reproduce the experiments:

1. Prepare the Severstal Steel Defect Detection dataset.
2. Run the preprocessing pipeline to generate YOLO-format annotations.
3. Configure the dataset YAML file.
4. Train the YOLOv8m baseline.
5. Train the YOLO11m baseline.
6. Train the YOLO26m baseline.
7. Apply the MSC modification to YOLO26m.
8. Train the proposed YOLO26m + MSC model.
9. Validate the trained models.
10. Compare Precision, Recall, mAP@50, and mAP@50-95.
11. Generate Grad-CAM++-style visualizations.
12. Optionally use Qwen2.5-VL for auxiliary natural-language interpretation.

---

# Limitations

- The proposed model is evaluated on the Severstal steel defect dataset.
- The MSC modification is applied to one intermediate YOLO26m convolutional
  layer.
- The proposed model does not improve every metric compared with the
  YOLO26m baseline.
- mAP@50-95 is slightly lower for YOLO26m + MSC than for the YOLO26m
  baseline.
- Pitted Surface performance is lower than some of the other defect classes.
- Grad-CAM++-style heatmaps are qualitative explanations and should not be
  interpreted as segmentation masks.
- Qwen2.5-VL is an auxiliary component and is not part of the quantitative
  detector evaluation.
- YOLOv4 was investigated but could not be completed as a directly
  comparable quantitative baseline because of implementation and environment
  issues.

---

# Project Status

| Component | Status |
|---|---|
| Dataset preprocessing | Completed |
| YOLO dataset generation | Completed |
| YOLOv8m baseline | Completed |
| YOLO11m baseline | Completed |
| YOLO26m baseline | Completed |
| MSC architecture | Completed |
| YOLO26m + MSC training | Completed |
| YOLO26m + MSC validation | Completed |
| Overall model comparison | Completed |
| Class-wise evaluation | Completed |
| Grad-CAM++-style XAI | Completed |
| Qwen2.5-VL integration | Completed |
| Qualitative analysis | Completed |
| YOLOv4 reproduction | Attempted; environment/implementation issues encountered |

---

# Conclusion

This project proposes a **Multi-Scale Convolution enhanced YOLO26m model** for
steel surface defect detection.

The proposed MSC module replaces one intermediate convolution with parallel
3×3 and 5×5 convolution branches, enabling the model to capture feature
information at different spatial scales while preserving the original
channel dimension.

The final proposed model achieved:

```text
Precision : 74.40%
Recall    : 71.00%
mAP@50    : 76.90%
mAP@50-95 : 45.90%
```

Compared with the YOLO26m baseline, the proposed model improved Precision,
Recall, and mAP@50 while maintaining comparable mAP@50-95 performance.

The project also incorporates a practical Grad-CAM++-style explainability
component and Qwen2.5-VL for auxiliary natural-language interpretation.

---

# Technologies

- Python
- PyTorch
- Ultralytics
- YOLOv8
- YOLO11
- YOLO26
- OpenCV
- NumPy
- Matplotlib
- Hugging Face Transformers
- Qwen2.5-VL
- CUDA

---

# License

This project is intended for academic and research purposes.

Please refer to the licenses of the underlying datasets, frameworks, models,
and third-party repositories used in this project.
