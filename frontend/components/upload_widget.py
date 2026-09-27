"""
Upload & Analysis UI Component for TruthLens
Supports media upload (images/videos) and text claims with live status polling.
"""

import streamlit as st
from typing import Callable

SAMPLE_CLAIMS = [
    (
        "🚨 Sensational Medical Rumor",
        "Secret miracle herbal drink cures Type-2 diabetes in 48 hours without insulin! Big pharma and doctors don't want you to know this shocking truth!"
    ),
    (
        "🔬 Verified Scientific Report",
        "According to a peer-reviewed study published by researchers at Cambridge University, clinical trials confirm a 35% reduction in cardiovascular risk with Mediterranean dietary habits."
    ),
    (
        "⚠️ Viral Space Conspiracy",
        "Leaked NASA classified documents prove secret alien bases discovered on the far side of the moon! Mind-blowing footage banned from mainstream media!"
    ),
    (
        "📰 Standard Neutral News",
        "The Federal Reserve announced an interest rate adjustment today following scheduled quarterly economic review meetings in Washington."
    )
]


def render_upload_section(api_client, on_scan_complete: Callable[[dict], None]):
    """
    Renders upload tabs for Media Deepfake Detection and Text Misinformation.
    """
    st.markdown("## 🔍 Submit Media or Text for AI Verification")
    st.write(
        "Upload a video/image to inspect for deepfake facial manipulation, "
        "or paste a text claim to evaluate misinformation and sensationalism."
    )

    tab_media, tab_text = st.tabs(["🎥 Media Deepfake Detection", "📝 Text Misinformation Analysis"])

    # -------------------------------------------------------------
    # TAB 1: Media Analysis (Image / Video)
    # -------------------------------------------------------------
    with tab_media:
        st.markdown("### Upload Video or Image File")
        uploaded_file = st.file_uploader(
            "Select an image or video file:",
            type=["mp4", "avi", "mov", "mkv", "png", "jpg", "jpeg", "webp"],
            help="Supports standard video (MP4, AVI, MOV) and image (PNG, JPG) formats up to 100MB."
        )

        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            sample_rate = st.slider(
                "Frame Sampling Rate (1 frame every N frames):",
                min_value=1,
                max_value=60,
                value=15,
                help="Higher sampling rates process faster; lower rates provide denser frame coverage."
            )
        with col_opt2:
            st.info(f"Target User: **{st.session_state.get('user_id', 'anonymous')}**")

        if uploaded_file is not None:
            # File preview
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
            is_video = filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

            with st.expander("👁️ Media Preview", expanded=True):
                if is_video:
                    st.video(file_bytes)
                else:
                    st.image(file_bytes, caption=filename, use_container_width=True)

            if st.button("🚀 Run Deepfake Scan", type="primary", use_container_width=True):
                progress_container = st.container()
                with progress_container:
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0.1)

                    try:
                        status_placeholder.info("📤 Uploading media to backend and registering scan...")
                        upload_res = api_client.upload_media(
                            file_bytes=file_bytes,
                            filename=filename,
                            user_id=st.session_state.get("user_id", "anonymous_user"),
                            sample_rate=sample_rate
                        )
                        scan_id = upload_res["scan_id"]
                        status_placeholder.info(f"⏳ Scan queued (`{scan_id}`). Extracting frames & detecting faces...")
                        progress_bar.progress(0.35)

                        # Status callback for polling
                        def on_poll_step(status, step):
                            if status == "processing":
                                pct = min(0.40 + step * 0.15, 0.90)
                                progress_bar.progress(pct)
                                status_placeholder.info(f"🔬 Running MTCNN face detection & CNN inference... (Step {step})")

                        # Poll for completion
                        final_result = api_client.poll_results(
                            scan_id=scan_id,
                            max_wait_sec=90,
                            interval_sec=1.5,
                            progress_callback=on_poll_step
                        )

                        progress_bar.progress(1.0)
                        status_placeholder.success("✅ Deepfake scan complete! Loading Results Dashboard...")
                        on_scan_complete(final_result)

                    except Exception as e:
                        progress_bar.empty()
                        status_placeholder.error(f"❌ Scan failed: {e}")

    # -------------------------------------------------------------
    # TAB 2: Text Misinformation Analysis
    # -------------------------------------------------------------
    with tab_text:
        st.markdown("### Paste News Claim or Article Snippet")

        # Preset claim loader
        sample_choice = st.selectbox(
            "Or load a sample claim to test instantly:",
            options=["-- Select a sample scenario --"] + [s[0] for s in SAMPLE_CLAIMS]
        )

        preset_text = ""
        for name, text in SAMPLE_CLAIMS:
            if sample_choice == name:
                preset_text = text
                break

        text_input = st.text_area(
            "Enter claim or statement:",
            value=preset_text,
            height=140,
            placeholder="Type or paste any breaking news headline, social media post, or assertion here..."
        )

        if st.button("🔎 Analyze Misinformation & Explainability", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("Please enter some text to analyze.")
            else:
                status_placeholder = st.empty()
                progress_bar = st.progress(0.15)

                try:
                    status_placeholder.info("📤 Sending claim to NLP pipeline...")
                    res = api_client.analyze_text(
                        text=text_input.strip(),
                        user_id=st.session_state.get("user_id", "anonymous_user")
                    )
                    scan_id = res["scan_id"]
                    progress_bar.progress(0.45)
                    status_placeholder.info(f"⏳ Text scan registered (`{scan_id}`). Running DistilBERT classification...")

                    def on_text_poll(status, step):
                        pct = min(0.50 + step * 0.20, 0.90)
                        progress_bar.progress(pct)
                        status_placeholder.info(f"🧠 Computing attention explainability weights... (Step {step})")

                    final_result = api_client.poll_results(
                        scan_id=scan_id,
                        max_wait_sec=60,
                        interval_sec=1.0,
                        progress_callback=on_text_poll
                    )

                    progress_bar.progress(1.0)
                    status_placeholder.success("✅ Misinformation analysis complete! Loading Results Dashboard...")
                    on_scan_complete(final_result)

                except Exception as e:
                    progress_bar.empty()
                    status_placeholder.error(f"❌ Analysis failed: {e}")
