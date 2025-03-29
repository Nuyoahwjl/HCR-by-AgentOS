import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import pandas as pd
from src.hcr import Recommendation
import time


def format_user_info(gender, age, height, weight, medical_history, symptoms, id="000000"):
    """格式化用户信息"""
    return {
        "id":id,
        "gender": gender,
        "age": age,
        "height": height,
        "weight": weight,
        "medical_history": medical_history,
        "symptoms": symptoms
    }


# 界面布局
st.set_page_config(
    page_title="Recommend",
    page_icon="🥰",
)


re = Recommendation()


st.markdown("""
<div style="text-align: center;">
    <h2><b>🩺Health Check Recommendation</b></h2>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center;">
    <img src="https://cdn.jsdelivr.net/gh//Nuyoahwjl/draft/typing.svg" 
         style="display: block; margin: auto; width: 100%;">
</div>
""", unsafe_allow_html=True)


id=st.text_input("ID(6 figures)", key="id", help="Please enter your ID")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox(label="Gender", 
                          options=["male", "female", "secret"],
                          format_func = str,
                          help = "if you don't want to tell us, keep secret")
    height = st.slider("Height(cm)", 50, 200, 50)
with col2:
    age = st.number_input("Age", 
                          min_value=0, 
                          max_value=100)
    weight = st.slider("Weight(kg)", 0, 100, 0)
medical_history = st.text_area("Medical History", key="medical", height=100)
symptoms = st.text_area("Symptoms", key="symptoms", height=100)
submitted = st.button("Recommend", icon='✔️', use_container_width=True)


with st.sidebar.expander(label="TEST",expanded=False):
    DeepSeek_API = st.text_input("DeepSeek API Key", type="password")
    if not DeepSeek_API.startswith("sk-"):
        st.warning("Please enter API!", icon="⚠️")


if submitted:
    if height == 50 or age ==0 or weight ==0 or not medical_history.strip() or not symptoms.strip():
        st.error("Please fill in all the information", icon="🚨")
        # st.toast("Please fill in all the information", icon="🚨")
    else:
        user_info = format_user_info(gender, age, height, weight, medical_history, symptoms, id)
        with st.spinner("analyzing...",show_time=True):
            start = time.time()
            result = re.run(user_info)
            with st.sidebar.expander(label="TEST",expanded=True):
                st.success(f"successfully(time:{time.time()-start:.1f}s)")
                st.write(user_info)
            with st.expander("RECOMMENDATIONS", expanded=True):
                st.markdown("## RECOMMENDATIONS")
                st.write(result)
                st.download_button(label="Download", data=result, file_name="Recommendations.md", use_container_width=True, icon="📥")
    







# DeepSeek_API = st.sidebar.text_input("DeepSeek API Key", type="password")
# if not DeepSeek_API.startswith("sk-"):
#     st.warning("Please enter your DeepSeek API key!", icon="⚠️")
# else:
#     re = RecommendationChain(api_key=DeepSeek_API)
#     # re.init_llm(api_key=DeepSeek_API)
#     if submitted:
#         if height == 50 or age ==0 or weight ==0 or not medical_history.strip() or not symptoms.strip():
#             st.error("Please fill in all the information", icon="🚨")
#             # st.toast("Please fill in all the information", icon="🚨")
#         else:
#             user_info = format_user_info(gender, age, height, weight, medical_history, symptoms)
#             with st.spinner("analyzing..."):
#                 start = time.time()
#                 result = re.run_chain(user_info)
#                 # result = "## 你好"
#                 with st.sidebar.expander(label=" ",expanded=True):
#                     st.success(f"successfully(time:{time.time()-start:.1f}s)")
#                     st.write(user_info)
#                 with st.expander("Recommendations", expanded=True):
#                     st.markdown("## RECOMMENDATIONS")
#                     st.write(result)
#                     st.download_button(label="download", data=result, file_name="Recommendations.md")