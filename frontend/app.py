"""
TruthLens Streamlit Frontend Application
Full-stack user interface for deepfake and misinformation detection.
"""

import os
import streamlit as st

from api_client import TruthLensAPIClient
from components.auth import render_auth_sidebar, init_session
from components.upload_widget import render_upload_section
from components.results_view import render_results_dashboard
from components.history_table import render_scan_history

# Setup Streamlit page config
st.set_page_config(
    page_title="TruthLens — AI Deepfake & Misinformation Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Clean headers and metric fonts */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #2d3748;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session
init_session()

# Backend Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
api_client = TruthLensAPIClient(base_url=BACKEND_URL)

# Navigation State
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "🔍 Analyze & Scan"

# -------------------------------------------------------------
# Sidebar Navigation & User Session
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='margin-bottom: 2px;'>🔍 TruthLens</h2>", unsafe_allow_html=True)
    st.caption("AI-Powered Misinformation & Deepfake Defense")
    st.markdown("---")

    # Render User Auth & Profile Switcher
    render_auth_sidebar()

    # System Navigation
    st.markdown("### 🧭 Navigation")
    nav_options = [
        "🔍 Analyze & Scan",
        "📊 Results Dashboard",
        "📜 Scan History",
        "ℹ️ Capstone Architecture"
    ]

    selected_nav = st.radio(
        "Go to:",
        options=nav_options,
        index=nav_options.index(st.session_state["active_tab"]) if st.session_state["active_tab"] in nav_options else 0
    )
    if selected_nav != st.session_state["active_tab"]:
        st.session_state["active_tab"] = selected_nav

    st.markdown("---")

    # Backend Connection Status Check
    health = api_client.check_health()
    if health.get("online"):
        st.success("🟢 Backend Connected (Port 8000)")
    else:
        st.error("🔴 Backend Disconnected")
        st.caption(f"Target: `{BACKEND_URL}`")

    st.sidebar.caption("TruthLens v1.0 • 6-Person Capstone")


# -------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------
def on_scan_complete(result_data: dict):
    st.session_state["current_result"] = result_data
    st.session_state["current_scan_id"] = result_data.get("scan_id")
    st.session_state["active_tab"] = "📊 Results Dashboard"
    st.rerun()


def on_select_history_scan(scan_id: str):
    try:
        with st.spinner(f"Loading scan report '{scan_id}'..."):
            result_data = api_client.get_results(scan_id)
            st.session_state["current_result"] = result_data
            st.session_state["current_scan_id"] = scan_id
            st.session_state["active_tab"] = "📊 Results Dashboard"
            st.rerun()
    except Exception as e:
        st.error(f"Failed to load scan {scan_id}: {e}")


# -------------------------------------------------------------
# Main Content Router
# -------------------------------------------------------------
st.markdown("<h1 class='main-title'>TruthLens Platform</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-title'>Multimodal AI verification engine for synthetic media & online misinformation.</p>",
    unsafe_allow_html=True
)

current_tab = st.session_state["active_tab"]

if current_tab == "🔍 Analyze & Scan":
    render_upload_section(api_client, on_scan_complete)

elif current_tab == "📊 Results Dashboard":
    render_results_dashboard(st.session_state.get("current_result"), BACKEND_URL)

elif current_tab == "📜 Scan History":
    render_scan_history(api_client, on_select_history_scan)

elif current_tab == "ℹ️ Capstone Architecture":
    st.markdown("## 🏛️ TruthLens System Architecture & Capstone Guide")
    st.write(
        "TruthLens is architected as a modular, decoupled full-stack platform designed "
        "to allow a 6-person student engineering team to develop and evaluate independently."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ### 🎥 Computer Vision Pipeline
            - **Frame Extraction**: OpenCV uniform sub-sampling (1 frame every $N$ frames).
            - **Face Detection**: MTCNN (`facenet-pytorch`) with bounding-box margin and canonical resizing.
            - **Classifier**: CNN backbone (EfficientNet / Xception) for Real vs Fake frame scoring.
            - **Aggregation**: Top-$K$ anomaly + mean hybrid pooling for video-level verdict.
            - **Explainability**: Grad-CAM activation heatmaps highlighting manipulated facial seams.
            """
        )
    with col2:
        st.markdown(
            """
            ### 📝 NLP Misinformation Pipeline
            - **Preprocessing**: HTML stripping, whitespace normalization, span-preserving tokenization.
            - **Classifier**: DistilBERT / RoBERTa transformer sequence classification (`CREDIBLE` vs `MISLEADING`).
            - **Explainability**: Multi-head self-attention extraction and lexical saliency weights.
            - **Highlighting**: Styled inline HTML rendering highlighting influential sensational claims.
            """
        )

    st.markdown("---")
    st.markdown(
        """
        ### ⚡ Backend & Infrastructure
        - **FastAPI**: Asynchronous REST endpoints with `BackgroundTasks` execution.
        - **Motor & MongoDB**: Async document database managing `users`, `scans`, `predictions`, and `explainability_outputs`.
        - **Docker Compose**: Single-command orchestration for Frontend, Backend, and MongoDB.
        """
    )
