"""
End-to-End System Integration Test for TruthLens
Validates:
  1. CV Pipeline (Frame extraction, Face detection, Classifier, Grad-CAM generation)
  2. NLP Pipeline (Preprocessing, Misinformation classifier, Attention/lexical explanation)
  3. Database Manager (Async Motor / in-memory collections)
  4. FastAPI Backend Routes & Background Task Execution
"""

import os
import sys
import asyncio
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import backend
from cv_pipeline.pipeline import CVPipeline
from nlp_pipeline.pipeline import NLPPipeline
from backend.database import db_manager
from backend.main import app
from fastapi.testclient import TestClient


def create_dummy_test_image(filepath: str):
    """Creates a synthetic test image with facial geometric features."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img = np.ones((300, 300, 3), dtype=np.uint8) * 230

    # Draw head / face oval
    cv2.ellipse(img, (150, 150), (80, 110), 0, 0, 360, (180, 160, 150), -1)
    # Draw eyes
    cv2.circle(img, (120, 130), 12, (50, 40, 30), -1)
    cv2.circle(img, (180, 130), 12, (50, 40, 30), -1)
    # Draw mouth
    cv2.ellipse(img, (150, 200), (25, 12), 0, 0, 180, (40, 40, 180), -1)

    cv2.imwrite(filepath, img)
    return filepath


def test_cv_pipeline():
    print("\n--- [TEST 1] Testing CV Pipeline ---")
    test_img_path = os.path.join(CURRENT_DIR, "scratch", "test_face.png")
    create_dummy_test_image(test_img_path)

    cv = CVPipeline(sample_rate_n=5)
    output_heatmaps = os.path.join(CURRENT_DIR, "scratch", "heatmaps")
    result = cv.analyze(
        media_path=test_img_path,
        output_heatmap_dir=output_heatmaps,
        scan_id="test_cv_001"
    )

    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Total frames: {result['total_frames_extracted']}")
    print(f"Heatmap file: {result['heatmap_path']}")
    print(f"Summary: {result['summary']}")

    assert result["verdict"] in ("REAL", "FAKE"), "Invalid CV verdict"
    assert 0.0 <= result["confidence"] <= 1.0, "Invalid confidence range"
    assert os.path.exists(result["heatmap_path"]), "Heatmap was not generated"
    print(">>> CV Pipeline Test PASSED! <<<")


def test_nlp_pipeline():
    print("\n--- [TEST 2] Testing NLP Pipeline ---")
    nlp = NLPPipeline()

    # Test sensational claim
    sensational_claim = "Secret miracle cure eliminates high blood pressure instantly in 24 hours! Doctors hate this shocking secret!"
    res_fake = nlp.analyze(sensational_claim)
    print(f"Sensational Claim Verdict: {res_fake['verdict']} (Confidence: {res_fake['confidence']})")
    print(f"Top Phrases: {res_fake['highlighted_phrases']}")
    assert res_fake["verdict"] in ("CREDIBLE", "MISLEADING"), "Invalid NLP verdict"
    assert len(res_fake["highlighted_phrases"]) > 0, "No phrases highlighted"
    assert "<mark" in res_fake["highlighted_html"], "HTML highlighting missing mark tags"

    # Test credible claim
    credible_claim = "According to a study published by researchers at Johns Hopkins University, regular exercise improves cardiovascular health."
    res_real = nlp.analyze(credible_claim)
    print(f"Credible Claim Verdict: {res_real['verdict']} (Confidence: {res_real['confidence']})")
    assert res_real["verdict"] in ("CREDIBLE", "MISLEADING")
    print(">>> NLP Pipeline Test PASSED! <<<")


async def test_database_manager():
    print("\n--- [TEST 3] Testing Database Layer ---")
    await db_manager.connect()
    print(f"Database status: {'MongoDB' if db_manager.is_connected else 'In-Memory Fallback'}")

    # Test insert scan
    test_scan = {
        "scan_id": "test_db_001",
        "user_id": "test_user",
        "scan_type": "text",
        "status": "pending",
        "created_at": "2026-09-27T10:00:00"
    }
    await db_manager.scans.insert_one(test_scan)

    # Test find scan
    found = await db_manager.scans.find_one({"scan_id": "test_db_001"})
    assert found is not None, "Scan not found in DB"
    assert found["user_id"] == "test_user"

    # Test update scan
    await db_manager.scans.update_one(
        {"scan_id": "test_db_001"},
        {"$set": {"status": "completed", "label": "CREDIBLE"}}
    )
    updated = await db_manager.scans.find_one({"scan_id": "test_db_001"})
    assert updated["status"] == "completed"
    assert updated["label"] == "CREDIBLE"

    print(">>> Database Layer Test PASSED! <<<")


def test_fastapi_endpoints():
    print("\n--- [TEST 4] Testing FastAPI Backend Endpoints ---")
    with TestClient(app) as client:
        # Test Root
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert res_root.json()["service"] == "TruthLens Backend API"
        print("GET /: OK")

        # Test Health
        res_health = client.get("/health")
        assert res_health.status_code == 200
        print("GET /health: OK")

        # Test POST /analyze-text
        res_text = client.post(
            "/analyze-text",
            json={"text": "Miracle herbal drink cures diabetes in 48 hours!", "user_id": "test_analyst"}
        )
        assert res_text.status_code in (200, 202)
        text_data = res_text.json()
        scan_id = text_data["scan_id"]
        print(f"POST /analyze-text: Created scan {scan_id}")

        # Test GET /results/{scan_id}
        res_result = client.get(f"/results/{scan_id}")
        assert res_result.status_code == 200
        result_data = res_result.json()
        print(f"GET /results/{scan_id}: Status={result_data['status']}")

        # Test POST /upload-media
        test_img = os.path.join(CURRENT_DIR, "scratch", "test_face.png")
        with open(test_img, "rb") as f:
            res_media = client.post(
                "/upload-media",
                files={"file": ("test_face.png", f, "image/png")},
                data={"user_id": "test_analyst", "sample_rate": 5}
            )
        assert res_media.status_code in (200, 202)
        media_scan_id = res_media.json()["scan_id"]
        print(f"POST /upload-media: Created scan {media_scan_id}")

        # Test GET /history/{user_id}
        res_hist = client.get("/history/test_analyst")
        assert res_hist.status_code == 200
        hist_data = res_hist.json()
        assert hist_data["total_scans"] >= 1
        print(f"GET /history/test_analyst: Found {hist_data['total_scans']} scans")

        print(">>> FastAPI Endpoints Test PASSED! <<<")


if __name__ == "__main__":
    print("==================================================")
    print("   TruthLens Capstone Integration Test Suite      ")
    print("==================================================")
    test_cv_pipeline()
    test_nlp_pipeline()
    asyncio.run(test_database_manager())
    test_fastapi_endpoints()
    print("\n==================================================")
    print("   ALL INTEGRATION TESTS PASSED SUCCESSFULLY!    ")
    print("==================================================")
