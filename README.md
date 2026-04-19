# Text-Guided Drywall Defect Segmentation

## Problem

Automatic quality inspection of drywall surfaces using deep learning. The model takes an image + text prompt (e.g., "segment crack", "segment taping area") and produces binary segmentation masks for defects.

Different defects require different annotations:
- Cracks → "segment crack"
- Taping areas → "segment taping area"
- Joints → "segment joint/tape"
- Seams → "segment drywall seam"

---

## Data Preprocessing

### 1. Cracks Dataset (Dataset 2)

**Source:** `cracks.coco.zip` (COCO format with polygon segmentations)

- 5,369 images (640x640)
- 8,511 annotations (polygons)
- 2 categories: crack, NewCracks

**Sample Data:**

| Image | Mask | Prompt |
|-------|------|--------|
| `2000x1500_5_resized_jpg.rf.0zMYivYn1ttmm5nyO0aE.jpg` | `2000x1500_5_resized_jpg.rf.0zMYivYn1ttmm5nyO0aE.png` | "segment crack" |
| `1_0005_2-Vertical-cracks_png_jpg.rf.6Bv2WLv15XAJ4dJZ1lXy.jpg` | `1_0005_2-Vertical-cracks_png_jpg.rf.6Bv2WLv15XAJ4dJZ1lXy.png` | "segment joint" |

![Cracks Visualization](https://github.com/pranjalpandeyl221/CRACKS/raw/main/cracks.png)

**Split:** Train 3,758 | Val 805 | Test 806

**Prompts:** "segment crack", "segment wall crack"

---

### 2. Drywall Dataset (Dataset 1)

**Source:** `Drywall-Join-Detect.coco.zip` (BBOX only - NO polygons)

- 1,022 images
- 1,424 annotations

**Split:** Train 715 | Val 153 | Test 154

**Prompts:** "segment taping area", "segment joint/tape", "segment drywall seam"

---

## Models

### 1. CLIPSeg (Baseline)

Lightweight baseline using CLIP encoders with trainable decoder.

- **Frozen:** CLIP image + text encoder
- **Trainable:** Decoder + text projection

### 2. Advanced SAM-FiLM (Main Model)

FiLM conditioning for text-feature fusion.

![Architecture](https://github.com/pranjalpandeyl221/CRACKS/raw/main/architecture.png)

**Components:**
- Image Encoder: SAM ViT-B (frozen) → 256-d
- Text Encoder: CLIP (frozen) → 512-d
- FiLM Generator: MLP(512→256) → gamma/beta
- Dilated Decoder: 4 branches (dilation 1,3,6,12)
- Edge-Aware Head: Laplacian + channel attention

**Training:** FiLM Generator + Decoder (trainable), Encoders (frozen)

**Model Stats:**
- Parameters: 87.59 M
- FLOPs: 149.93 G
- Inference Time: 64.30 ms

---

## Results

### Dataset 1: Cracks

| Metric | Value |
|--------|-------|
| IoU | 0.5012 |
| mIoU | 0.7330 |
| Precision | 0.6230 |
| Recall | 0.7195 |
| F1 | 0.6678 |
| Dice | 0.6678 |

![Crack Results](https://github.com/pranjalpandeyl221/CRACKS/raw/main/r1.png)
![Crack Results](https://github.com/pranjalpandeyl221/CRACKS/raw/main/r2.png)
![Crack Results](https://github.com/pranjalpandeyl221/CRACKS/raw/main/r3.png)

![Crack Failure](https://github.com/pranjalpandeyl221/CRACKS/raw/main/failure_case.png)

---

### Dataset 2: Drywall

| Metric | Test | Val |
|--------|------|-----|
| Loss | 2.2028 | 2.7489 |
| IoU | 0.3755 | 0.3206 |
| mIoU | 0.5441 | 0.4762 |
| Precision | 0.3885 | 0.3362 |
| Recall | 0.9318 | 0.8947 |
| F1 | 0.5338 | 0.4696 |

![Drywall Visualization](https://github.com/pranjalpandeyl221/CRACKS/raw/main/d1v.png)

![Drywall Results](https://github.com/pranjalpandeyl221/CRACKS/raw/main/dataset1_result.png)

![Drywall Failure](https://github.com/pranjalpandeyl221/CRACKS/raw/main/dataset1_failure_case.png)

![Training Curve](https://github.com/pranjalpandeyl221/CRACKS/raw/main/dataset1_training_curve.png)

---

## Metrics

| Metric | Description |
|-------|-------------|
| IoU | Foreground Intersection over Union |
| mIoU | Mean IoU (foreground + background) |
| Precision | Per-class mean precision |
| Recall | Per-class mean recall |
| F1 | Per-class mean F1 score |
| Dice | Dice coefficient (foreground) |

---

## Installation

```bash
pip install torch torchvision timm einops scipy opencv-python-headless
pip install matplotlib pillow ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## Training

```bash
jupyter notebook train_cracks_segmentation.ipynb
jupyter notebook train_drywall_segmentation.ipynb
```

**Config:** batch=4, epochs=20, lr=1e-4, num_workers=4

---

## Project Structure

```
.
├── train_cracks_segmentation.ipynb
├── train_drywall_segmentation.ipynb
├── cracks_dataset_v2/
├── drywall_dataset_v2/
└── README.md
```

---

## License

MIT
