# Steel Surface Defect Detection Using YOLO

A deep learning project for detecting surface defects in steel images using
YOLO-based object detection models.

## Project Overview

This project investigates YOLO-based approaches for detecting four types
of steel surface defects:

- Pitted Surface
- Crazing
- Scratches
- Patches

The project follows a baseline-to-proposed-model workflow.

## Dataset

Dataset:
Severstal: Steel Defect Detection

The original dataset contains 1600 × 256 steel surface images with
Run-Length Encoded (RLE) segmentation annotations.

The annotations are converted into YOLO bounding-box format during
preprocessing.

## Preprocessing

The preprocessing pipeline includes:

1. RLE mask decoding
2. Four-class mask construction
3. 80/20 train-validation split
4. Image tiling
5. Connected-component based bounding-box extraction
6. Bounding-box filtering
7. Conversion to YOLO label format
8. YOLO dataset generation

Tiling configuration:

- Tile width: 608
- Tile positions: 0, 495, 991
- Bounding-box padding: 3 pixels

## Models

The project evaluates multiple YOLO models:

- YOLOv4
- YOLOv8m
- YOLO11m
- YOLO26m

The modern YOLO26m model is used as the baseline for the proposed
architecture.

## Proposed Model

The proposed model enhances YOLO26m using a Multi-Scale Convolution
(MSC) block.

The MSC block uses parallel 3×3 and 5×5 convolution branches to capture
features at different spatial scales.

## Evaluation

The models are evaluated using:

- Precision
- Recall
- mAP@50
- mAP@50-95

Additional analysis includes qualitative detection results.

## Explainability

Grad-CAM is planned for visual explanation of model attention.

## Vision-Language Model

A Vision-Language Model (VLM) is planned to provide natural-language
interpretation of detected steel defects.

## Project Status

| Component | Status |
|---|---|
| Dataset preprocessing | Completed |
| YOLOv8m baseline | Completed |
| YOLO11m baseline | Completed |
| YOLO26m baseline | Completed |
| YOLO26m + MSC | Training / evaluation |
| YOLOv4 | In progress |
| Grad-CAM | Planned |
| VLM | Planned |
| Final demo | Planned |
