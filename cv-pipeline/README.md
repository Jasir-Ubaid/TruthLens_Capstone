# TruthLens — Computer Vision Pipeline (CV Team)

This module handles video decoding, uniform frame extraction, MTCNN face localization, deepfake binary classification using CNNs (EfficientNet / Xception), frame-level score aggregation, and Grad-CAM visual heatmaps.

---

## 📁 Architecture & File Layout

| File | Purpose |
| :--- | :--- |
| `extractor.py` | Extracts 1 frame every $N$ frames via OpenCV `VideoCapture` (or loads single images). Handles fps, timestamps, and safety caps. |
| `face_detector.py` | Detects facial regions using MTCNN (`facenet-pytorch`) with bounding box padding (margin) and canonical resizing to $224 \times 224$. Includes Haar cascade fallback. |
| `classifier.py` | Pretrained PyTorch CNN (EfficientNet-B0 / Xception) binary classifier (`REAL` vs `FAKE`). Aggregates frame scores into a single video verdict. |
| `gradcam.py` | Generates Class Activation Maps (Grad-CAM) overlaid on the face to highlight suspicious spatial artifacts (blending boundaries, eyes, mouth). |
| `pipeline.py` | High-level orchestrator class `CVPipeline` integrating all 4 sub-modules. |

---

## 🚀 Quickstart for CV Team Member

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Standalone Testing in Python
```python
from pipeline import CVPipeline

pipeline = CVPipeline(
    sample_rate_n=15,               # Sample 1 frame every 15 frames
    model_name="efficientnet_b0",
    weights_path=None               # None uses pretrained transfer-learning head
)

result = pipeline.analyze(
    media_path="tests/sample_video.mp4",
    output_heatmap_dir="output_heatmaps",
    scan_id="test_scan_001"
)

print("Verdict:", result["verdict"])
print("Confidence:", result["confidence"])
print("Heatmap saved at:", result["heatmap_path"])
```

---

## 🛠️ How to Plug In Fine-Tuned Weights

When you train an EfficientNet or Xception model on datasets such as **FaceForensics++**, **Celeb-DF**, or **DFDC**:

1. Save your PyTorch weights dictionary:
   ```python
   torch.save(model.state_dict(), "weights/efficientnet_b0_deepfake.pth")
   ```
2. Pass the path to `CVPipeline`:
   ```python
   pipeline = CVPipeline(
       weights_path="path/to/weights/efficientnet_b0_deepfake.pth"
   )
   ```
3. To customize the target Grad-CAM layer for a custom architecture, adjust `target_layer_name` in `gradcam.py`.

---

## 📊 Frame Aggregation Algorithm

Videos may contain deepfake manipulation on only a subset of frames (e.g., face swapped in a short 2-second segment). Our aggregation logic:
1. Calculates **Top-K Anomaly Score** ($K = 30\%$ most anomalous frames).
2. Calculates **Mean Score** across all sampled frames.
3. Combines them:
   $$\text{Video Fake Score} = 0.60 \times \text{Top-K} + 0.40 \times \text{Mean}$$
4. Categorizes as `FAKE` if score $\ge 0.50$, else `REAL`.
