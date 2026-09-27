# TruthLens — REST API Specification

The TruthLens Backend provides RESTful endpoints for video/image deepfake detection, text misinformation analysis, scan result polling, and user scan histories.

**Base URL:** `http://localhost:8000` (Local) or `http://backend:8000` (Docker Network)  
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc` (ReDoc)

---

## 1. Upload Media for Deepfake Scan

### `POST /upload-media`
Accepts an image or video file, stores the file, registers a pending scan in the database, and schedules asynchronous background inference.

#### Request
- **Content-Type:** `multipart/form-data`
- **Parameters (Form Data):**
  - `file` (File, Required): Binary video (`.mp4`, `.avi`, `.mov`, `.mkv`) or image (`.jpg`, `.jpeg`, `.png`, `.webp`). Maximum size: 100 MB.
  - `user_id` (String, Optional): Identifier of the user. Default: `"anonymous_user"`.
  - `sample_rate` (Integer, Optional): Sampling rate (extract 1 frame every $N$ frames). Default: `15`.

#### Example Request (`curl`)
```bash
curl -X POST "http://localhost:8000/upload-media" \
  -F "file=@tests/sample_video.mp4" \
  -F "user_id=alice_analyst" \
  -F "sample_rate=15"
```

#### Response (`202 Accepted`)
```json
{
  "scan_id": "scan_4f92bc31a8",
  "status": "pending",
  "filename": "sample_video.mp4",
  "media_type": "video",
  "message": "Media uploaded successfully. Async processing queued."
}
```

---

## 2. Analyze Text for Misinformation

### `POST /analyze-text`
Accepts a news claim, headline, or excerpt, registers a pending scan in MongoDB, and schedules asynchronous NLP classification.

#### Request
- **Content-Type:** `application/json`
- **Body:**
```json
{
  "text": "Secret miracle herbal drink cures diabetes in 48 hours without insulin!",
  "user_id": "alice_analyst"
}
```

#### Example Request (`curl`)
```bash
curl -X POST "http://localhost:8000/analyze-text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Secret miracle herbal drink cures diabetes in 48 hours without insulin!",
    "user_id": "alice_analyst"
  }'
```

#### Response (`202 Accepted`)
```json
{
  "scan_id": "scan_9a12c8e310",
  "status": "pending",
  "message": "Text claim accepted. Async processing queued."
}
```

---

## 3. Retrieve Scan Results

### `GET /results/{scan_id}`
Returns the current lifecycle status and, once completed, full model predictions, confidence scores, and explainability artifacts.

#### Parameters (Path)
- `scan_id` (String, Required): Unique scan ID returned by the upload endpoint.

#### Response (`200 OK` — Status: Completed for Media)
```json
{
  "scan_id": "scan_4f92bc31a8",
  "user_id": "alice_analyst",
  "scan_type": "media",
  "status": "completed",
  "filename": "sample_video.mp4",
  "text_preview": null,
  "created_at": "2026-09-27T10:14:00.000Z",
  "updated_at": "2026-09-27T10:14:05.120Z",
  "prediction": {
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
      }
    ],
    "summary": "Analyzed 12 frames: 10 flagged as suspicious, 2 authentic. Overall verdict: FAKE (89.2% confidence).",
    "completed_at": "2026-09-27T10:14:05.120Z"
  },
  "explainability": {
    "type": "gradcam_heatmap",
    "heatmap_url": "/static/heatmaps/heatmap_scan_4f92bc31a8.jpg",
    "highlighted_phrases": null,
    "highlighted_html": null,
    "token_importances": null
  },
  "error_message": null
}
```

#### Response (`200 OK` — Status: Completed for Text)
```json
{
  "scan_id": "scan_9a12c8e310",
  "user_id": "alice_analyst",
  "scan_type": "text",
  "status": "completed",
  "filename": null,
  "text_preview": "Secret miracle herbal drink cures diabetes in 48 hours without insulin!...",
  "created_at": "2026-09-27T10:15:00.000Z",
  "updated_at": "2026-09-27T10:15:01.300Z",
  "prediction": {
    "model_name": "TruthLens-DISTILBERT-MISINFORMATION",
    "label": "MISLEADING",
    "confidence": 0.914,
    "frame_scores": null,
    "summary": "Evaluated claim (13 words). Assigned verdict: MISLEADING (91.4% confidence). Identified 3 key influential phrases.",
    "completed_at": "2026-09-27T10:15:01.300Z"
  },
  "explainability": {
    "type": "text_highlight",
    "heatmap_url": null,
    "highlighted_phrases": [
      {
        "phrase": "miracle herbal drink",
        "importance": 0.92,
        "category": "Sensationalism"
      },
      {
        "phrase": "cures diabetes in 48 hours",
        "importance": 0.89,
        "category": "Sensationalism"
      }
    ],
    "highlighted_html": "<div style=\"...\">Secret <mark style=\"...\">miracle</mark> herbal drink <mark style=\"...\">cures</mark> diabetes...</div>",
    "token_importances": [
      { "token": "miracle", "score": 0.92 },
      { "token": "cures", "score": 0.89 }
    ]
  },
  "error_message": null
}
```

---

## 4. User Scan History

### `GET /history/{user_id}`
Returns all past scans performed by a given `user_id`, sorted with the most recent scans first.

#### Parameters (Path)
- `user_id` (String, Required): Identifier of the user.

#### Example Request (`curl`)
```bash
curl -X GET "http://localhost:8000/history/alice_analyst"
```

#### Response (`200 OK`)
```json
{
  "user_id": "alice_analyst",
  "total_scans": 2,
  "scans": [
    {
      "scan_id": "scan_9a12c8e310",
      "scan_type": "text",
      "status": "completed",
      "label": "MISLEADING",
      "confidence": 0.914,
      "filename": null,
      "text_preview": "Secret miracle herbal drink cures diabetes in 48 hours without insulin!...",
      "created_at": "2026-09-27T10:15:00.000Z"
    },
    {
      "scan_id": "scan_4f92bc31a8",
      "scan_type": "media",
      "status": "completed",
      "label": "FAKE",
      "confidence": 0.892,
      "filename": "sample_video.mp4",
      "text_preview": null,
      "created_at": "2026-09-27T10:14:00.000Z"
    }
  ]
}
```

---

## 5. Health Check & Root

- `GET /` — Returns service metadata, database mode, and available route links.
- `GET /health` — Returns `{"status": "healthy", "database_connected": true}`.
