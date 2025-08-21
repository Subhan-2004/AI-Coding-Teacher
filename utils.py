import streamlit as st
import json

CHAT_KEY = "chat_history"

def init_chat_history():
    """Initialize chat history in session state."""
    if CHAT_KEY not in st.session_state:
        st.session_state[CHAT_KEY] = []

def export_chat_bytes():
    """Export chat history as downloadable JSON."""
    if CHAT_KEY not in st.session_state:
        return b"[]"
    data = json.dumps(st.session_state[CHAT_KEY], ensure_ascii=False, indent=2)
    return data.encode("utf-8")
