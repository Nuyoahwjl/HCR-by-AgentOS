import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import streamlit as st
from src.report import get_health_check_info
from src.prompt import REPORT_PROMPT
from together import Together
from openai import OpenAI
import time

st.set_page_config(
    page_title="体检报告",
    page_icon="📄",
)

with st.sidebar:
    st.header("⚙️ 设置")
    st.session_state.model = st.selectbox(
        "选择模型",
        # ("deepseek-ai/DeepSeek-V3","Qwen/QwQ-32B","google/gemma-2-27b-it"),
        ("deepseek/deepseek-v3.1","qwen/qwen3-235b-a22b-fp8","zai-org/glm-4.6"),
        index=0,
        key="model_selector"
    )
    st.session_state.api_key = st.text_input("API 密钥", type="password")
    st.markdown("---")

def call_model(messages, api_key: str | None = None, model: str = "deepseek/deepseek-v3.1"):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.ppinfra.com/openai",
    )
    completion = client.chat.completions.create(
    model=model,
    messages=messages,
    )
    return completion.choices[0].message.content 


st.title("体检报告")
st.markdown("### 输入您的 8 位卡号以生成报告")

card_number = st.text_input("卡号", max_chars=8, help="请输入 8 位卡号")

if st.button("生成报告"):
    report = get_health_check_info(int(card_number))
    if card_number.isdigit() and len(card_number) == 8 and report != 0:
        with st.spinner("正在生成报告...", show_time=True):
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
                st.success("报告生成成功！")
            except Exception as e:
                st.error(f"发生错误：{e}")
            with st.expander("报告详情", expanded=True):
                    st.markdown("### 报告详情")
                    st.write(result)
                    st.download_button(label="下载", data=result, file_name="报告.md", use_container_width=True, icon="📥")
    else:
        st.error("请输入有效的 8 位卡号。")