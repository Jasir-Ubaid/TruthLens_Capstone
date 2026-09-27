"""
TruthLens Frontend REST API Client
Handles HTTP communication between Streamlit UI and FastAPI backend.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Callable
import requests

logger = logging.getLogger("TruthLens.Frontend.APIClient")

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


class TruthLensAPIClient:
    """
    Communicates with TruthLens FastAPI backend endpoints.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or DEFAULT_BACKEND_URL).rstrip("/")

    def check_health(self) -> Dict[str, Any]:
        """Checks if backend service and database are reachable."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=3.0)
            if resp.status_code == 200:
                return {"online": True, **resp.json()}
            return {"online": False, "status_code": resp.status_code}
        except requests.exceptions.RequestException as e:
            return {"online": False, "error": str(e)}

    def upload_media(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: str = "anonymous_user",
        sample_rate: int = 15
    ) -> Dict[str, Any]:
        """
        Calls POST /upload-media with multipart form file and parameters.
        """
        url = f"{self.base_url}/upload-media"
        files = {"file": (filename, file_bytes)}
        data = {"user_id": user_id, "sample_rate": sample_rate}

        try:
            response = requests.post(url, files=files, data=data, timeout=30.0)
            if response.status_code in (200, 202):
                return response.json()
            else:
                error_msg = response.json().get("detail", response.text)
                raise RuntimeError(f"Server error ({response.status_code}): {error_msg}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to backend at {url}: {e}")

    def analyze_text(
        self,
        text: str,
        user_id: str = "anonymous_user"
    ) -> Dict[str, Any]:
        """
        Calls POST /analyze-text with JSON payload.
        """
        url = f"{self.base_url}/analyze-text"
        payload = {"text": text, "user_id": user_id}

        try:
            response = requests.post(url, json=payload, timeout=15.0)
            if response.status_code in (200, 202):
                return response.json()
            else:
                error_msg = response.json().get("detail", response.text)
                raise RuntimeError(f"Server error ({response.status_code}): {error_msg}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to backend at {url}: {e}")

    def get_results(self, scan_id: str) -> Dict[str, Any]:
        """
        Calls GET /results/{scan_id} to retrieve scan status and metrics.
        """
        url = f"{self.base_url}/results/{scan_id}"
        try:
            response = requests.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                raise KeyError(f"Scan '{scan_id}' not found.")
            else:
                raise RuntimeError(f"Server error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch results: {e}")

    def poll_results(
        self,
        scan_id: str,
        max_wait_sec: int = 60,
        interval_sec: float = 1.0,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Polls GET /results/{scan_id} until the scan status is 'completed' or 'failed'.
        """
        start_time = time.time()
        step = 0

        while time.time() - start_time < max_wait_sec:
            step += 1
            result = self.get_results(scan_id)
            current_status = result.get("status", "pending")

            if progress_callback:
                progress_callback(current_status, step)

            if current_status == "completed":
                return result
            elif current_status == "failed":
                err = result.get("error_message", "Unknown pipeline error.")
                raise RuntimeError(f"Scan processing failed: {err}")

            time.sleep(interval_sec)

        raise TimeoutError(f"Scan '{scan_id}' timed out after {max_wait_sec} seconds.")

    def get_history(self, user_id: str) -> Dict[str, Any]:
        """
        Calls GET /history/{user_id} to retrieve past scans.
        """
        url = f"{self.base_url}/history/{user_id}"
        try:
            response = requests.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return {"user_id": user_id, "total_scans": 0, "scans": []}
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not load scan history: {e}")
            return {"user_id": user_id, "total_scans": 0, "scans": []}
