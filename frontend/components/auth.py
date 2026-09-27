"""
Basic Username Authentication & Session Manager
Handles simple username-based sessions in Streamlit state.
"""

import streamlit as st

DEFAULT_PROFILES = [
    {"user_id": "alice_analyst", "username": "Alice (Fact Checker)", "role": "Investigator"},
    {"user_id": "bob_journalist", "username": "Bob (Journalist)", "role": "Editor"},
    {"user_id": "charlie_student", "username": "Charlie (Capstone Student)", "role": "Researcher"},
    {"user_id": "demo_guest", "username": "Guest User", "role": "Visitor"}
]


def init_session():
    """Ensures user session variables are populated in Streamlit."""
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "alice_analyst"
    if "username" not in st.session_state:
        st.session_state["username"] = "Alice (Fact Checker)"
    if "current_scan_id" not in st.session_state:
        st.session_state["current_scan_id"] = None
    if "current_result" not in st.session_state:
        st.session_state["current_result"] = None


def render_auth_sidebar():
    """Renders user profile selector and session switcher in Streamlit sidebar."""
    init_session()

    st.sidebar.markdown("### 👤 User Session")
    user_options = {p["username"]: p["user_id"] for p in DEFAULT_PROFILES}

    # Find current index
    current_uid = st.session_state.get("user_id", "alice_analyst")
    default_idx = 0
    for idx, p in enumerate(DEFAULT_PROFILES):
        if p["user_id"] == current_uid:
            default_idx = idx
            break

    selected_name = st.sidebar.selectbox(
        "Active Profile:",
        options=list(user_options.keys()),
        index=default_idx,
        help="Select a simulated user profile for tracking individual scan histories."
    )

    new_user_id = user_options[selected_name]
    if new_user_id != st.session_state.get("user_id"):
        st.session_state["user_id"] = new_user_id
        st.session_state["username"] = selected_name
        st.rerun()

    # Custom username override expander
    with st.sidebar.expander("Switch Custom Username"):
        custom_id = st.text_input("Enter User ID / Handle:", value="", placeholder="e.g. analyst_42")
        if st.button("Set Custom User"):
            if custom_id.strip():
                st.session_state["user_id"] = custom_id.strip()
                st.session_state["username"] = custom_id.strip()
                st.rerun()

    st.sidebar.caption(f"Logged in as: **{st.session_state['user_id']}**")
    st.sidebar.markdown("---")
