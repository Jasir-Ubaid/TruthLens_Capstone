# TruthLens — System Architecture & Data Flow

This document details the multi-tiered architecture, database models, and execution lifecycle of **TruthLens** — an AI-powered platform for deepfake and misinformation detection.

---

## 🏛️ High-Level System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Layer"]
        UI["Streamlit Frontend (Port 8501)"]
        Browser["End User / Analyst"]
        Browser <--> UI
    end

    subgraph API ["Backend Gateway (FastAPI, Port 8000)"]
        Main["FastAPI Router (main.py)"]
        MediaEP["POST /upload-media"]
        TextEP["POST /analyze-text"]
        ResEP["GET /results/{scan_id}"]
        HistEP["GET /history/{user_id}"]
        BG["FastAPI BackgroundTasks"]
        
        UI -->|REST API Calls| Main
        Main --> MediaEP
        Main --> TextEP
        Main --> ResEP
        Main --> HistEP
        MediaEP --> BG
        TextEP --> BG
    end

    subgraph Pipelines ["AI Engine Workers (Async Execution)"]
        CV["CV Pipeline (cv-pipeline/)<br/>• OpenCV Frame Extraction<br/>• MTCNN Face Localization<br/>• EfficientNet / Xception CNN<br/>• Grad-CAM Heatmap Generation"]
        NLP["NLP Pipeline (nlp-pipeline/)<br/>• Regex/Text Normalization<br/>• DistilBERT / RoBERTa Classifier<br/>• Multi-head Attention Explainer<br/>• HTML Highlight Generation"]
        
        BG -.->|Trigger Media Task| CV
        BG -.->|Trigger Text Task| NLP
    end

    subgraph Storage ["Persistence & Assets Layer"]
        Mongo[("MongoDB (Port 27017)<br/>• users<br/>• scans<br/>• predictions<br/>• explainability_outputs")]
        StaticFS[("Local Disk / Static Assets<br/>• /uploads<br/>• /static/heatmaps")]
        
        BG <-->|Motor Async Driver| Mongo
        CV -->|Save Heatmap JPG| StaticFS
        ResEP <--> Mongo
        HistEP <--> Mongo
        UI -->|Load Heatmap URL| StaticFS
    end
```

---

## 🔄 End-to-End Sequence Workflows

### 1. Media Deepfake Analysis Workflow

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Analyst
    participant Front as Streamlit Frontend
    participant Back as FastAPI Backend
    participant DB as MongoDB (Motor)
    participant Worker as Background Task
    participant CV as CV Pipeline

    User->>Front: Upload video / image & click 'Run Scan'
    Front->>Back: POST /upload-media (multipart/form-data)
    Back->>Back: Validate format & save media to uploads/
    Back->>DB: Insert scan record (status="pending")
    Back->>Worker: Enqueue process_media_scan task
    Back-->>Front: Return 202 Accepted {scan_id, status: "pending"}
    
    par Async Processing
        Worker->>DB: Update scan status to "processing"
        Worker->>CV: Execute FrameExtractor (1 frame every N frames)
        CV->>CV: MTCNN Face Detection & Bounding Box Margin Crop
        CV->>CV: CNN Forward Pass (Frame-level Real/Fake Probabilities)
        CV->>CV: Top-K & Mean Anomaly Aggregation -> Video Verdict
        CV->>CV: Compute Grad-CAM on most suspicious frame
        CV-->>Worker: Return {verdict, confidence, frame_scores, heatmap_path}
        Worker->>DB: Insert into 'predictions' collection
        Worker->>DB: Insert into 'explainability_outputs' collection
        Worker->>DB: Update scan status to "completed"
    and Frontend Polling
        loop Poll until completed
            Front->>Back: GET /results/{scan_id}
            Back->>DB: Query scan status
            Back-->>Front: Return {status: "processing" / "completed"}
        end
    end
    Front->>User: Render Dashboard (Badge, Confidence Gauge, Grad-CAM Overlay, Timeline)
```

---

### 2. Text Misinformation Analysis Workflow

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Analyst
    participant Front as Streamlit Frontend
    participant Back as FastAPI Backend
    participant DB as MongoDB (Motor)
    participant Worker as Background Task
    participant NLP as NLP Pipeline

    User->>Front: Paste news claim & click 'Analyze Misinformation'
    Front->>Back: POST /analyze-text {text, user_id}
    Back->>DB: Insert scan record (status="pending")
    Back->>Worker: Enqueue process_text_scan task
    Back-->>Front: Return 202 Accepted {scan_id, status: "pending"}
    
    par Async Processing
        Worker->>DB: Update scan status to "processing"
        Worker->>NLP: Preprocess & normalize raw text
        NLP->>NLP: Sequence Classification (DistilBERT/RoBERTa)
        NLP->>NLP: Self-Attention / Saliency Extraction
        NLP->>NLP: Generate styled inline HTML & phrase weights
        NLP-->>Worker: Return {verdict, confidence, highlighted_phrases, highlighted_html}
        Worker->>DB: Insert into 'predictions' & 'explainability_outputs'
        Worker->>DB: Update scan status to "completed"
    and Frontend Polling
        loop Poll until completed
            Front->>Back: GET /results/{scan_id}
            Back-->>Front: Return {status: "completed", prediction, explainability}
        end
    end
    Front->>User: Render Dashboard (Verdict, Attention-highlighted Text, Top Phrases)
```

---

## 🗄️ Database Schema (MongoDB via Motor)

### 1. `scans` Collection
Tracks the lifecycle of media or text analysis requests.
```json
{
  "_id": "ObjectId('...')",
  "scan_id": "scan_a1b2c3d4e5",
  "user_id": "alice_analyst",
  "scan_type": "media",              // "media" | "text"
  "media_type": "video",             // "video" | "image" (if scan_type is media)
  "filename": "sample_speech.mp4",
  "saved_filepath": "/app/backend/uploads/scan_a1b2c3d4e5_speech.mp4",
  "text_claim": null,
  "text_preview": null,
  "status": "completed",             // "pending" | "processing" | "completed" | "failed"
  "label": "FAKE",
  "confidence": 0.892,
  "created_at": "2026-09-27T10:00:00.000Z",
  "updated_at": "2026-09-27T10:00:04.250Z",
  "error_message": null
}
```

### 2. `predictions` Collection
Stores the model outputs and frame-level statistics.
```json
{
  "_id": "ObjectId('...')",
  "prediction_id": "pred_scan_a1b2c3d4e5",
  "scan_id": "scan_a1b2c3d4e5",
  "model_name": "TruthLens-EFFICIENTNET_B0-MTCNN",
  "label": "FAKE",
  "confidence": 0.892,
  "frame_scores": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "label": "FAKE",
      "confidence": 0.912,
      "fake_probability": 0.912,
      "bbox": [120, 80, 240, 210]
    },
    {
      "frame_index": 15,
      "timestamp_sec": 0.5,
      "label": "FAKE",
      "confidence": 0.874,
      "fake_probability": 0.874,
      "bbox": [122, 82, 242, 212]
    }
  ],
  "summary": "Analyzed 12 frames: 10 flagged as suspicious, 2 authentic. Overall verdict: FAKE (89.2% confidence).",
  "completed_at": "2026-09-27T10:00:04.250Z"
}
```

### 3. `explainability_outputs` Collection
Contains visual heatmaps and phrase highlights.
```json
{
  "_id": "ObjectId('...')",
  "explainability_id": "exp_scan_a1b2c3d4e5",
  "prediction_id": "pred_scan_a1b2c3d4e5",
  "scan_id": "scan_a1b2c3d4e5",
  "type": "gradcam_heatmap",         // "gradcam_heatmap" | "text_highlight"
  "heatmap_url": "/static/heatmaps/heatmap_scan_a1b2c3d4e5.jpg",
  "heatmap_path": "/app/backend/static/heatmaps/heatmap_scan_a1b2c3d4e5.jpg",
  "highlighted_phrases": [],
  "highlighted_html": null,
  "token_importances": []
}
```

### 4. `users` Collection
Stores profile metadata and scan volume.
```json
{
  "_id": "ObjectId('...')",
  "user_id": "alice_analyst",
  "username": "Alice (Fact Checker)",
  "created_at": "2026-09-27T09:00:00.000Z",
  "last_active": "2026-09-27T10:00:04.250Z",
  "scan_count": 8
}
```
