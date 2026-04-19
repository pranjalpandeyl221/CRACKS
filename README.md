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

## Data Preprocessing

### 1. Cracks Dataset (Dataset 2)

**Source:** `cracks.coco.zip` (COCO format with polygon segmentations)

**Original Data:**
- 5,369 images (640x640)
- 8,511 annotations (polygons)
- 2 categories: crack, NewCracks

**Sample from Original Dataset:**

| Image | Mask | Prompt |
|-------|------|--------|
| `00021_jpg.rf.S0P7SnzwPx0s4z4ldWvu.jpg` | `00021_jpg.rf.S0P7SnzwPx0s4z4ldWvu.png` | "segment crack" |
| `00028_jpg.rf.WIWJghqsuUyyDgujVwNb.jpg` | `00028_jpg.rf.WIWJghqsuUyyDgujVwNb.png` | "segment wall crack" |
| `1065-dat_png_jpg.rf.NPbxq9RlzpljbRiU8J4I.jpg` | `1065-dat_png_jpg.rf.NPbxq9RlzpljbRiU8J4I.png` | "segment crack" |

![Cracks Dataset Visualization](https://github.com/pranjalpandeyl221/CRACKS/raw/main/cracks_visualization.png)

**Preprocessing Steps:**

1. **Extract ZIP** → COCO annotations + images
2. **Analyze COCO JSON** to understand structure
3. **Create Train/Val/Test Split** (70/15/15, SEED=42)
4. **Convert Polygons to Masks** using cv2.fillPoly()
5. **Fix Filename Alignment** - mask filenames now match image filenames
6. **Export** to train/val/test folders

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

**Sample from Original Dataset:**

| Image | Mask | Prompt |
|-------|------|--------|
| `2000x1500_46_resized_jpg.rf.PrVgoG5ug1wBk53ehTDi.jpg` | `2000x1500_46_resized_jpg.rf.PrVgoG5ug1wBk53ehTDi.png` | "segment taping area" |
| `IMG_20220627_110122-jpg_1500x2000_jpg.rf.6xOiyE2EhMUtNR6J1A7C.jpg` | `IMG_20220627_110122-jpg_1500x2000_jpg.rf.6xOiyE2EhMUtNR6J1A7C.png` | "segment joint/tape" |
| `IMG_8205_JPG_jpg.rf.tFAjesep4ZdwmACgBRFq.jpg` | `IMG_8205_JPG_jpg.rf.tFAjesep4ZdwmACgBRFq.png` | "segment drywall seam" |

**Preprocessing Steps:**

1. **Extract ZIP** → COCO annotations + images
2. **Convert BBOX to Rectangular Masks** - Since no polygons, convert bounding boxes to rectangle masks
3. **Create Train/Val/Test Split** (70/15/15, SEED=42)
4. **Export** to train/val/test folders

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

- **FiLM Generator:** MLP that maps text → gamma/beta parameters
- **Dilated Decoder:** Multi-scale features with different dilation rates
- **Edge-Aware Head:** Incorporates edge detection for boundary accuracy

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

## Results

Training produces:
- Training curves (loss, mIoU, Dice)
- Test evaluation metrics
- Visualization of predictions
- Failure case analysis

---

## License

MIT
