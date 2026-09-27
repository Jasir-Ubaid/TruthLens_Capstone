"""
Scan History UI Component for TruthLens
Displays past scans for the current user in reverse-chronological order.
"""

import streamlit as st
import pandas as pd
from typing import Callable


def render_scan_history(api_client, on_select_scan: Callable[[str], None]):
    """
    Renders chronological scan history table for active user profile.
    """
    user_id = st.session_state.get("user_id", "anonymous_user")
    st.markdown(f"## 📜 Scan History for `{user_id}`")

    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.write("Chronological log of past deepfake and misinformation scans.")
    with col_btn:
        if st.button("🔄 Refresh History", use_container_width=True):
            st.rerun()

    history_data = api_client.get_history(user_id=user_id)
    scans = history_data.get("scans", [])

    if not scans:
        st.info(f"No past scans found for user `{user_id}`. Run a scan from the **Analyze & Scan** tab!")
        return

    # Filter by scan type
    c_filter, _ = st.columns([2, 2])
    with c_filter:
        type_filter = st.selectbox("Filter by Scan Type:", ["All Scans", "Media (Image/Video)", "Text (Misinformation)"])

    filtered_scans = []
    for s in scans:
        if type_filter == "Media (Image/Video)" and s.get("scan_type") != "media":
            continue
        if type_filter == "Text (Misinformation)" and s.get("scan_type") != "text":
            continue
        filtered_scans.append(s)

    st.caption(f"Showing {len(filtered_scans)} of {len(scans)} total scans.")

    # Render card list / table
    for scan in filtered_scans:
        sid = scan["scan_id"]
        stype = scan.get("scan_type", "unknown").upper()
        status = scan.get("status", "unknown")
        label = scan.get("label") or "PENDING"
        conf = scan.get("confidence")
        conf_str = f"{conf * 100:.1f}%" if conf is not None else "N/A"
        target_name = scan.get("filename") or scan.get("text_preview") or "Untitled"
        created_at = scan.get("created_at", "")[:19].replace("T", " ")

        with st.container():
            col1, col2, col3, col4 = st.columns([1.5, 2.5, 1.5, 1.2])

            with col1:
                if label in ("FAKE", "MISLEADING"):
                    st.error(f"**{label}**")
                elif label in ("REAL", "CREDIBLE"):
                    st.success(f"**{label}**")
                else:
                    st.warning(f"**{label}**")
                st.caption(f"ID: `{sid}`")

            with col2:
                icon = "🎥" if stype == "MEDIA" else "📝"
                st.markdown(f"**{icon} {target_name[:65]}**")
                st.caption(f"Type: `{stype}` | Date: `{created_at}`")

            with col3:
                st.metric(label="Confidence", value=conf_str)

            with col4:
                if st.button("🔍 Inspect", key=f"btn_inspect_{sid}", use_container_width=True):
                    on_select_scan(sid)

            st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #2a2e39;'>", unsafe_allow_html=True)
