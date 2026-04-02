import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🕹️",
)


# 自定义CSS样式
st.markdown("""
    <style>
    /* General styling */
    body {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    
    /* Select box styling */
    .stSelectbox select {
        border-radius: 15px;
        padding: 8px 16px;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 15px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    
    /* Message containers */
    .message-container {
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1.2rem 0;
        max-width: 70%;
        position: relative;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        clear: both;
    }
    .user-message {
        background: #ffffff;
        border: 2px solid #4A90E2;
        float: left;
        margin-left: 2%;
        border-radius: 0 20px 20px 20px;
    }
    .assistant-message {
        background: #f0f4ff;
        border: 2px solid #6c5ce7;
        float: left;
        margin-left: 2%;
        border-radius: 0 20px 20px 20px;
    }
    
    /* Response time styling */
    .response-time {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.8rem;
        text-align: right;
    }
    
    /* Header styling */
    .medical-header {
        padding: 1.5rem;
        background: #F8F9FA;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
            
    /* 标题居中 */
    h1 {
        text-align: center !important;
        color: #2d3436;
        margin-bottom: 1.5rem !important;
    }
    
    /* 通知文本居中 */
    .centered-notice {
        background: rgba(255, 243, 205, 0.9); 
        border: 2px solid rgba(255, 238, 186, 0.5); 
        max-width: 800px;
        text-align: center;
        border-radius: 12px;
        padding: 1rem;
        color: hsl(45, 100%, 30%); 
        
        /* 深色主题覆盖 */
        @media (prefers-color-scheme: dark) {
            background: rgba(77, 60, 15, 0.95); 
            border-color: hsl(45, 70%, 40%);
            max-width: 800px;
            text-align: center;
            border-radius: 12px;
            padding: 1rem;
            color: hsl(45, 70%, 80%); 
        }
    }
    </style>
""", unsafe_allow_html=True)


# 初始化session状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = "deepseek-chat"


# 初始化session状态
MEDICAL_SYSTEM_PROMPT = """
你是一个专业的医疗助手，请遵循以下准则：
1. 仅提供医学信息和建议，不回答非医疗问题
2. 所有信息需基于最新医学指南和权威来源
3. 遇到紧急情况请建议立即就医
4. 用通俗易懂的方式解释专业术语
5. 对不确定的信息要明确说明
6. 不提供诊断，只给建议性信息
7. 如果用户使用的是英文，请用英文回答
"""


# 侧边栏配置
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.model = st.selectbox(
        "Select Model",
        ("deepseek-chat", "deepseek-reasoner"),
        index=0,
        key="model_selector"
    )
    api_key = st.text_input("API Key", type="password")
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []


# 主界面
st.title("🕹️ AI Medical Assistant")
st.markdown("""
    <div class="centered-notice">
    The information provided by this assistant is for reference only and cannot replace professional medical advice. In case of emergency, please contact healthcare professionals immediately.
    </div>
""", unsafe_allow_html=True)


# 处理流式响应
def generate_response(messages):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    full_response = ""
    message_placeholder = st.empty()
    try:
        for chunk in client.chat.completions.create(
            model=st.session_state.model,
            messages=messages,
            stream=True,
            temperature=0.3,
            max_tokens=1000
        ):
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(f'<div class="message-container assistant-message">{full_response}_</div>', unsafe_allow_html=True)
        message_placeholder.markdown(f'<div class="message-container assistant-message">{full_response}</div>', unsafe_allow_html=True)
        return full_response
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


# 显示历史消息
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "assistant-message"
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="message-container {role_class}">{message["content"]}</div>', unsafe_allow_html=True)


# 用户输入处理
if prompt := st.chat_input("Enter your medical query..."):
    if not api_key:
        st.error("Please enter your API Key")
        st.stop()

    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="message-container user-message">{prompt}</div>', unsafe_allow_html=True)
    
    # 构建带系统提示的完整消息
    chat_history = [{"role": "system", "content": MEDICAL_SYSTEM_PROMPT}]
    chat_history += st.session_state.messages[-6:] # 保留最近3轮对话
    
    # 生成并显示助手回复
    with st.chat_message("assistant"):
        response = generate_response(chat_history)
    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})
    
