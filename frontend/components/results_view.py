"""
Results Dashboard Component for TruthLens
Renders prediction labels, confidence metrics, and explainability visualizations
(Grad-CAM heatmaps for media, attention-based highlights for text).
"""

import json
import streamlit as st
import pandas as pd


def render_results_dashboard(scan_data: dict, backend_url: str):
    """
    Renders the deepfake / misinformation analysis results.
    """
    if not scan_data:
        st.info("No active scan results. Submit a file or text claim in the **Analyze & Scan** tab.")
        return

    scan_id = scan_data.get("scan_id", "N/A")
    scan_type = scan_data.get("scan_type", "unknown")
    status = scan_data.get("status", "unknown")
    created_at = scan_data.get("created_at", "")
    pred = scan_data.get("prediction") or {}
    exp = scan_data.get("explainability") or {}

    st.markdown(f"## 📊 Scan Analysis Report (`{scan_id}`)")

    if status == "pending" or status == "processing":
        st.warning(f"Scan `{scan_id}` is currently in status: **{status.upper()}**. Please check back shortly.")
        return
    elif status == "failed":
        st.error(f"Scan `{scan_id}` failed. Error: {scan_data.get('error_message', 'Unknown error')}")
        return

    label = pred.get("label", "UNKNOWN").upper()
    confidence = float(pred.get("confidence", 0.0))
    model_name = pred.get("model_name", "TruthLens-AI")
    summary = pred.get("summary", "")

    # Top summary metrics cards
    col_verdict, col_conf, col_model = st.columns([1.5, 1.2, 1.3])

    with col_verdict:
        # High impact color coding
        if label in ("FAKE", "MISLEADING"):
            st.error(f"### 🚨 VERDICT: {label}")
            st.caption("High probability of synthetic manipulation or falsehood.")
        elif label in ("REAL", "CREDIBLE"):
            st.success(f"### 🛡️ VERDICT: {label}")
            st.caption("Content appears authentic or verified by trustworthy sources.")
        else:
            st.warning(f"### ❓ VERDICT: {label}")

    with col_conf:
        conf_pct = confidence * 100.0
        st.metric(label="Model Confidence", value=f"{conf_pct:.1f}%")
        st.progress(confidence)

    with col_model:
        st.metric(label="Inference Architecture", value=model_name.split("-")[0])
        st.caption(f"Full Model: `{model_name}`")

    if summary:
        st.info(f"**AI Diagnostic Summary:** {summary}")

    st.markdown("---")

    # -------------------------------------------------------------
    # Explainability View: Media (Grad-CAM Heatmap)
    # -------------------------------------------------------------
    if scan_type == "media":
        st.markdown("### 🔬 Visual Explainability (Grad-CAM Heatmap)")
        st.write(
            "Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes the specific facial regions "
            "and spatial artifacts that triggered the classifier's verdict. "
            "**Red/Warm zones** indicate high artificial manipulation energy (e.g. eye boundaries, lips, blended seams)."
        )

        heatmap_rel_url = exp.get("heatmap_url")
        if heatmap_rel_url:
            full_heatmap_url = f"{backend_url.rstrip('/')}{heatmap_rel_url}"

            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(
                    full_heatmap_url,
                    caption="Grad-CAM Anomaly Heatmap (Original Face [Left] | Activation Overlay [Right])",
                    use_container_width=True
                )
            with c2:
                st.markdown("#### 🔍 Heatmap Interpretation")
                st.markdown(
                    """
                    - **Eyes & Gaze**: Deepfakes frequently exhibit asymmetrical blinking and reflections.
                    - **Mouth & Teeth**: GAN/Diffusion synthesis often blurs teeth boundaries and lip synchronization.
                    - **Facial Border**: Boundary seams between the swapped face mask and the original skull.
                    """
                )
                st.markdown(f"**Target Class:** `{label}`")
                st.markdown(f"**Timestamp:** `{created_at[:19].replace('T', ' ')} UTC`")
        else:
            st.warning("No Grad-CAM heatmap was generated for this scan.")

        # Frame-by-frame confidence chart (for video)
        frame_scores = pred.get("frame_scores")
        if frame_scores and len(frame_scores) > 1:
            st.markdown("### ⏱️ Temporal Frame-by-Frame Anomaly Timeline")
            st.write("Track suspicious fluctuations across video playback timestamps:")

            df_frames = pd.DataFrame([
                {
                    "Frame": fs["frame_index"],
                    "Timestamp (s)": fs["timestamp_sec"],
                    "Fake Probability": fs.get("fake_probability", 0.5),
                    "Verdict": fs["label"]
                }
                for fs in frame_scores
            ])

            st.line_chart(df_frames, x="Timestamp (s)", y="Fake Probability", color="#ff4b4b")

            with st.expander("📋 View Detailed Frame Scores Table"):
                st.dataframe(df_frames, use_container_width=True)

    # -------------------------------------------------------------
    # Explainability View: Text (Attention Highlighting)
    # -------------------------------------------------------------
    elif scan_type == "text":
        st.markdown("### 🔍 Linguistic Explainability & Phrase Highlighting")
        st.write(
            "Transformer self-attention and lexical saliency identify specific tokens and phrases "
            "influencing the misinformation classifier."
        )

        highlighted_html = exp.get("highlighted_html")
        if highlighted_html:
            st.markdown("#### 🎯 Highlighted Claim View:")
            st.markdown(highlighted_html, unsafe_allow_html=True)
            st.caption("Colored words reflect key influence on the verdict (Darker hue = Higher influence).")

        # Top Influential Phrases Cards
        phrases = exp.get("highlighted_phrases") or []
        if phrases:
            st.markdown("#### 📌 Key Influential Phrases:")
            p_cols = st.columns(min(len(phrases), 3))
            for i, p in enumerate(phrases[:3]):
                with p_cols[i]:
                    st.metric(
                        label=p.get("category", "Phrase"),
                        value=f"\"{p['phrase']}\"",
                        delta=f"{p['importance'] * 100:.0f}% Weight"
                    )

        # Word Importance Breakdown Table
        tokens = exp.get("token_importances") or []
        if tokens:
            with st.expander("📊 Word-Level Importance Weights"):
                df_tokens = pd.DataFrame(tokens)
                df_tokens.columns = ["Token", "Importance Score (0.0 - 1.0)"]
                st.dataframe(df_tokens, use_container_width=True)

    # -------------------------------------------------------------
    # Export / Report
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📥 Export Scan Report")
    report_json = json.dumps(scan_data, indent=2)
    st.download_button(
        label="Download Full JSON Report",
        data=report_json,
        file_name=f"truthlens_scan_{scan_id}.json",
        mime="application/json"
    )
