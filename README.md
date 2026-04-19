# Text-Guided Drywall Defect Segmentation

## Problem

Automatic quality inspection of drywall surfaces using deep learning. Model takes image + text prompt (e.g., "segment crack", "segment taping area") → outputs binary segmentation masks.

Different defects need different prompts:
- Cracks → "segment crack"
- Taping areas → "segment taping area"
- Joints → "segment joint/tape"
- Seams → "segment drywall seam"

---

## Data Preprocessing

### 1. Cracks Dataset (Dataset 2)

**Source:** `cracks.coco.zip` (COCO with polygons)

- 5,369 images (640x640)
- 8,511 annotations

**Sample:** `2000x1500_5_resized_jpg.rf.0zMYivYn1ttmm5nyO0aE.jpg` → "segment crack"

![Cracks Viz](https://github.com/pranjalpandeyl221/CRACKS/raw/main/im_crk/cracks.png)

**Split:** Train 3,758 | Val 805 | Test 806

---

### 2. Drywall Dataset (Dataset 1)

**Source:** `Drywall-Join-Detect.coco.zip` (BBOX only - NO polygons)

- 1,022 images
- 1,424 annotations

![Drywall Viz](https://github.com/pranjalpandeyl221/CRACKS/raw/main/im_crk/dlv.png)

**Split:** Train 715 | Val 153 | Test 154

---

## Models

### 1. CLIPSeg (Baseline)

CLIP encoders + trainable decoder.
JUST TO GET REFERENCE RESULTS..

- **Frozen:** CLIP image + text encoder
- **Trainable:** Decoder + text projection

**Results:**

| IoU | Precision | Recall | F1 | Dice |
|-----|----------|--------|----|----|
| 0.4554 | 0.6846 | 0.5763 | 0.6258 | 0.6258 |

---

### 2. Advanced SAM-FiLM (Main Model)

FiLM conditioning for text-feature fusion.

![Architecture](https://github.com/pranjalpandeyl221/CRACKS/raw/main/im_crk/architecture.png)

**Components:**
- Image Encoder: SAM ViT-B (frozen) → 256-d
- Text Encoder: CLIP (frozen) → 512-d
- FiLM Generator: MLP(512→256) → gamma/beta
- Dilated Decoder: 4 branches (dilation 1,3,6,12)
- Edge-Aware Head: Laplacian + channel attention

**Training:** FiLM Generator + Decoder (trainable), Encoders (frozen)

**Stats:** Params: 87.59 M | FLOPs: 149.93 G | Inference: 64.30 ms

---

## Results

### 1. Dataset 2: Cracks (SAM-FiLM)

| IoU | mIoU | Precision | Recall | F1 | Dice |
|-----|------|----------|--------|----|----|
| 0.5012 | 0.7330 | 0.6230 | 0.7195 | 0.6678 | 0.6678 |

![Results](https://github.com/pranjalpandeyl221/CRACKS/raw/main/im_crk/r1.png)

![Results](https://github.com/pranjalpandeyl221/CRACKS/raw/main/im_crk/r2.png)

![Results](https://github.com/pranjalpandeyl221/CRACKS/raw/main/im_crk/r3.png)

### Failure Cases

![Failure](https://github.com/pranjalpandeyl221/CRACKS/raw/main/im_crk/failure_case.png)

**Shown above:** Cases where model predicted poorly (IoU < 0.3). Main issues: small thin cracks, low contrast areas, overlapping annotations.

---

### 2. Prompt Experiments

**"segment crack":**

| IoU | mIoU | Precision | Recall | F1 | Dice |
|-----|------|----------|--------|----|----|
| 0.4457 | 0.6988 | 0.5096 | 0.7805 | 0.6166 | 0.6166 |

**"segment wall crack":**

| IoU | mIoU | Precision | Recall | F1 | Dice |
|-----|------|----------|--------|----|----|
| 0.4587 | 0.7084 | 0.5584 | 0.7199 | 0.6290 | 0.6290 |

![Prompt Exp](https://github.com/pranjalpandeyl221/CRACKS/blob/main/im_crk/p1_p2.png)

---

### 3. Dataset 1: Drywall

| Metric | Test | Val |
|--------|------|-----|
| Loss | 2.2028 | 2.7489 |
| IoU | 0.3755 | 0.3206 |
| mIoU | 0.5441 | 0.4762 |
| Precision | 0.3885 | 0.3362 |
| Recall | 0.9318 | 0.8947 |
| F1 | 0.5338 | 0.4696 |

![Drywall Results](https://github.com/pranjalpandeyl221/CRACKS/raw/main/im_crk/dataset1_result.png)

![Drywall Failure](https://github.com/pranjalpandeyl221/CRACKS/raw/main/im_crk/dataset1_failure_case.png)

![Training Curve](https://github.com/pranjalpandeyl221/CRACKS/raw/main/im_crk/dataset1_training_curve.png)

---

## Metrics

| Metric | Description |
|-------|-------------|
| IoU | Foreground Intersection over Union |
| mIoU | Mean IoU (foreground + background) |
| Precision | Per-class mean precision |
| Recall | Per-class mean recall |
| F1 | Per-class mean F1 |
| Dice | Dice coefficient |

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

Note books are given accordingly

**Config:** batch=4, epochs=20, lr=1e-4, num_workers=4

---

## License

MIT
