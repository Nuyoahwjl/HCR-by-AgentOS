import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import streamlit as st
from src.report import get_health_check_info
from src.prompt import REPORT_PROMPT
from agentos.utils import call_model
import time

st.set_page_config(
    page_title="Report",
    page_icon="📄",
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.model = st.selectbox(
        "Select Model",
        ("deepseek-chat", "deepseek-reasoner"),
        index=0,
        key="model_selector"
    )
    st.session_state.api_key = st.text_input("API Key", type="password")
    st.markdown("---")

st.title("Health Check Report")
st.markdown("### Enter your 8-digit card number to generate the report")

card_number = st.text_input("Card Number", max_chars=8, help="Enter an 8-digit card number")

if st.button("Generate Report"):
    report = get_health_check_info(int(card_number))
    if card_number.isdigit() and len(card_number) == 8 and report != 0:
        with st.spinner("Generating report...",show_time=True):
            start = time.time()
            try:
                report = get_health_check_info(int(card_number))
                result = call_model(
                    messages = [
                    {"role": "user", "content": REPORT_PROMPT.format(report)}
                    ],
                    api_key=st.session_state.api_key,
                    model=st.session_state.model
                )
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            with st.expander("Report Details", expanded=True):
                    st.markdown("### Report Details")
                    st.write(result)
                    st.download_button(label="Download", data=result, file_name="Report.md", use_container_width=True, icon="📥")
    else:
        st.error("Please enter a valid 8-digit card number.")
