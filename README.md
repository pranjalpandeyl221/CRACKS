# Text-Guided Drywall Defect Segmentation

A deep learning project for automatic quality inspection of drywall surfaces. This project trains segmentation models that take an image + text prompt (e.g., "segment crack", "segment taping area") and produce binary segmentation masks for defects.

## Overview

This project addresses **Drywall QA** (Quality Assurance) using text-conditioned segmentation. The key insight is that different defects require different annotations - cracks need "segment crack" while taping areas need "segment taping area". By conditioning on text prompts, a single model can learn to segment multiple defect types.

### Supported Defect Types

| Dataset | Prompt | Description |
|---------|--------|------------|
| Cracks | "segment crack" | Wall cracks |
| Drywall | "segment taping area" | Taping/joint areas |
| Drywall | "segment joint/tape" | Drywall seams |
| Drywall | "segment drywall seam" | Seam locations |

---

## Results (Cracks Dataset)

### Test Metrics

| Metric | Value |
|--------|-------|
| **Test IoU (Crack Only)** | 0.5012 |
| **Test mIoU (Crack + BG)** | 0.7330 |
| **Precision** | 0.6230 |
| **Recall** | 0.7195 |
| **F1 Score** | 0.6678 |
| **Dice** | 0.6678 |

### Visual Results

![Results 1](https://github.com/pranjalpandeyl221/CRACKS/raw/main/r1.png)

![Results 2](https://github.com/pranjalpandeyl221/CRACKS/raw/main/r2.png)

![Results 3](https://github.com/pranjalpandeyl221/CRACKS/raw/main/r3.png)

### Failure Cases

![Failure Cases](https://github.com/pranjalpandeyl221/CRACKS/raw/main/failure_case.png)

**Failure Analysis:**
- Small thin cracks often missed
- Low contrast cracks in shadowed areas
- Overlapping annotations in training data

---

## Data Preprocessing

### 1. Cracks Dataset (Dataset 2)

**Source:** `cracks.coco.zip` (COCO format with polygon segmentations)

**Original Data:**
- 5,369 images (640x640)
- 8,511 annotations (polygons)
- 2 categories: crack, NewCracks

**Sample Data:**

| Image | Mask | Prompt |
|-------|------|--------|
| `2000x1500_5_resized_jpg.rf.0zMYivYn1ttmm5nyO0aE.jpg` | `2000x1500_5_resized_jpg.rf.0zMYivYn1ttmm5nyO0aE.png` | "segment crack" |
| `1_0005_2-Vertical-cracks_png_jpg.rf.6Bv2WLv15XAJ4dJZ1lXy.jpg` | `1_0005_2-Vertical-cracks_png_jpg.rf.6Bv2WLv15XAJ4dJZ1lXy.png` | "segment joint" |

![Cracks Visualization](https://github.com/pranjalpandeyl221/CRACKS/raw/main/cracks.png)

**Split Statistics:**

| Split  | Count |
|--------|-------|
| Train  | 3,758 |
| Val    | 805  |
| Test   | 806  |

**Text Prompts:** "segment crack", "segment wall crack"

---

### 2. Drywall Dataset (Dataset 1)

**Source:** `Drywall-Join-Detect.coco.zip` (COCO format with BBOX only)

**Original Data:**
- 1,022 images
- 1,424 annotations (bbox, NO polygons)

**Split Statistics:**

| Split  | Count |
|--------|-------|
| Train  | 715  |
| Val    | 153  |
| Test   | 154  |

**Text Prompts:** "segment taping area", "segment joint/tape", "segment drywall seam"

---

## Models

### 1. CLIPSeg (Baseline)

A lightweight baseline using CLIP's image/text encoders with a trainable decoder.

- **Frozen:** CLIP image encoder + text encoder
- **Trainable:** Decoder + text projection
- **Architecture:** Upsampling decoder with batch norm

### 2. Advanced SAM-FiLM (Main Model)

More powerful architecture with FiLM conditioning for better text-feature fusion.

![Architecture](https://github.com/pranjalpandeyl221/CRACKS/raw/main/architecture.png)

**Components:**

- **Image Encoder:** SAM ViT-B (frozen) → 256-d feature maps
- **Text Encoder:** CLIP (frozen) → 512-d text embeddings  
- **FiLM Generator:** MLP(512→256) → generates gamma/beta for feature modulation
- **Dilated Decoder:** 4 parallel branches (dilation 1,3,6,12) for multi-scale context
- **Edge-Aware Head:** Laplacian edge detection + channel attention for boundary accuracy
- **Output:** Binary segmentation mask (1 channel)

**Training:**
- FiLM Generator: trainable
- Decoder: trainable
- Encoders: frozen

**Model Stats:**
- Parameters: 87.59 M
- FLOPs: 149.93 G
- Inference Time: 64.30 ms

---

## Metrics

All models are evaluated with:

| Metric | Description |
|-------|-------------|
| **IoU** | Foreground Intersection over Union |
| **mIoU** | Mean IoU (foreground + background) |
| **Precision** | Per-class mean precision |
| **Recall** | Per-class mean recall |
| **F1** | Per-class mean F1 score |
| **Dice** | Dice coefficient (foreground) |

---

## Installation

```bash
# Core packages
pip install torch torchvision timm einops scipy opencv-python-headless
pip install matplotlib pillow ftfy regex tqdm

# CLIP
pip install git+https://github.com/openai/CLIP.git

# SAM (for Advanced SAM-FiLM)
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## Training

```bash
# Cracks dataset
jupyter notebook train_cracks_segmentation.ipynb

# Drywall dataset  
jupyter notebook train_drywall_segmentation.ipynb
```

**Training Config:**
- Image size: 224 (CLIP) / 640 (SAM)
- Batch size: 4
- Epochs: 20
- Learning rate: 1e-4
- num_workers: 4

---

## Project Structure

```
.
├── train_cracks_segmentation.ipynb    # CLIPSeg baseline for cracks
├── train_drywall_segmentation.ipynb # Advanced SAM-FiLM for drywall
├── cracks_dataset_v2/              # Processed cracks dataset
├── drywall_dataset_v2/              # Processed drywall dataset
├── PRE_PROCESS.md                   # Data preprocessing details
└── README.md                       # This file
```

---

## License

MIT
