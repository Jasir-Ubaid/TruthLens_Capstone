# TruthLens — 6-Person Capstone Collaboration Guide

This guide establishes the division of responsibilities, independent testing protocols, and Git workflows for our 6-person student capstone team.

---

## 👥 Team Member Roles & Ownership

| Member | Title | Primary Directory | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Backend & DB Lead** | `/backend` | • FastAPI endpoints & route validation<br/>• Motor async MongoDB database integration<br/>• `BackgroundTasks` queue orchestration<br/>• Docker containerization & static file serving |
| **Member 2** | **Frontend & UX Lead** | `/frontend` | • Streamlit interactive interface<br/>• User profile session state management<br/>• Results dashboard & interactive charts<br/>• REST API client polling logic |
| **Member 3** | **CV Extraction & Detection** | `/cv-pipeline` | • OpenCV video sub-sampling (`extractor.py`)<br/>• MTCNN face localization & margin padding (`face_detector.py`)<br/>• Handling multi-face & low-resolution frames |
| **Member 4** | **CV Modeling & Grad-CAM** | `/cv-pipeline` | • EfficientNet / Xception architecture tuning (`classifier.py`)<br/>• Video-level frame score aggregation algorithm<br/>• Grad-CAM activation heatmap generation (`gradcam.py`) |
| **Member 5** | **NLP Preprocessing & Model** | `/nlp-pipeline` | • Text sanitization & regex token span tracking (`preprocessor.py`)<br/>• DistilBERT / RoBERTa sequence classification (`classifier.py`)<br/>• Dataset fine-tuning (LIAR, FakeNewsNet, WELFake) |
| **Member 6** | **NLP Explainability & QA** | `/nlp-pipeline` | • Multi-head attention weight extraction (`explainer.py`)<br/>• Key phrase extraction & styled HTML highlighter<br/>• End-to-end integration tests & benchmarking |

---

## 🛠️ How Each Sub-Team Works Independently

### Backend Team (Members 1 & 2)
The backend and frontend communicate strictly via documented REST endpoints.
- If you don't have MongoDB running locally, the backend **automatically falls back to an in-memory async store**.
- You can test endpoints via Swagger UI at `http://localhost:8000/docs` without needing the Streamlit frontend running.
- The frontend can test its UI using mock responses if the backend is being refactored.

### Computer Vision Team (Members 3 & 4)
Work exclusively inside the `/cv-pipeline` directory:
- Test your pipeline standalone without launching FastAPI or MongoDB:
  ```python
  from pipeline import CVPipeline

  cv = CVPipeline(sample_rate_n=15)
  result = cv.analyze("sample.mp4", output_heatmap_dir="heatmaps", scan_id="test_01")
  print("Verdict:", result["verdict"], "Confidence:", result["confidence"])
  ```
- **Plugging in trained weights:** Place your `.pth` checkpoint in a `weights/` directory and pass `weights_path="weights/my_model.pth"` in `pipeline.py`.

### NLP Team (Members 5 & 6)
Work exclusively inside the `/nlp-pipeline` directory:
- Test your pipeline standalone:
  ```python
  from pipeline import NLPPipeline

  nlp = NLPPipeline()
  res = nlp.analyze("Secret cure cures cancer in 2 days!")
  print("Verdict:", res["verdict"], "Phrases:", res["highlighted_phrases"])
  ```
- **Plugging in fine-tuned models:** Save your fine-tuned Hugging Face model and tokenizer to a directory (e.g. `checkpoints/my_distilbert/`), and initialize `NLPPipeline(model_name_or_path="checkpoints/my_distilbert/")`.

---

## 🌿 Git Branching Strategy

1. **`main`**: Production-ready, stable releases (both Docker Compose and local test pass).
2. **`dev`**: Active integration branch.
3. **Feature branches**:
   - `feat/backend-storage` (Member 1)
   - `feat/frontend-ux` (Member 2)
   - `feat/cv-mtcnn-tuning` (Member 3)
   - `feat/cv-gradcam-refine` (Member 4)
   - `feat/nlp-distilbert-finetune` (Member 5)
   - `feat/nlp-attention-explainer` (Member 6)

Always verify `pytest` or standalone pipeline runs before creating a Pull Request to `dev`.
