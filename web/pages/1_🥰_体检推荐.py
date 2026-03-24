import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


import streamlit as st
import pandas as pd
from src.hcr import Recommendation
import time
import random


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
    page_title="体检推荐",
    page_icon="🥰",
)



st.markdown("""
<div style="text-align: center;">
    <h2><b>🩺体检推荐</b></h2>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center;">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&pause=1000&color=FF4B4B&center=true&vCenter=true&width=800&lines=%E4%BD%A0%E5%A5%BD%F0%9F%91%8B%2C+%E6%AC%A2%E8%BF%8E%E6%9D%A5%E5%88%B0HCR%E4%BD%93%E6%A3%80%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F!;%E5%A1%AB%E5%85%A5%E4%BB%A5%E4%B8%8B%E4%B8%AA%E4%BA%BA%E4%BF%A1%E6%81%AF%E4%BB%A5%E8%8E%B7%E5%BE%97%E5%AE%8C%E6%95%B4%E6%8E%A8%E8%8D%90" 
         style="display: block; margin: auto; width: 100%;">
</div>
""", unsafe_allow_html=True)



# TOGETHER_AI_API = st.text_input("API", type="password" ,help="https://api.together.xyz/")
# if not TOGETHER_AI_API.startswith("tgp"):
#     st.warning("请输入API!", icon="⚠️")

PIPO_API = st.text_input("API", type="password" ,help="https://ppio.com/model-api/console")
if not PIPO_API.startswith("sk"):
    st.warning("请输入API!", icon="⚠️")





id=st.text_input("ID(6位数)", key="id", help="请输入你的ID")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox(label="性别", 
                          options=["男", "女", "保密"],
                          format_func = str)
    height = st.slider("身高(cm)", 0, 200, 50)
with col2:
    age = st.number_input("年龄", 
                          min_value=0, 
                          max_value=100)
    weight = st.slider("体重(kg)", 0, 100, 40)
medical_history = st.text_area("既往病史", key="medical", height=100)
symptoms = st.text_area("现有症状", key="symptoms", height=100)
submitted = st.button("生成推荐", icon='✔️', use_container_width=True)



if submitted:
    # if not TOGETHER_AI_API.startswith("tgp"):
    if not PIPO_API.startswith("sk"):
        pass
    else:
        if height == 50 or age == 0 or weight == 40 or not medical_history.strip() or not symptoms.strip():
            st.error("请输入完整的个人信息！", icon="🚨")
        else:
            re= Recommendation(PIPO_API)
            user_info = format_user_info(gender, age, height, weight, medical_history, symptoms, id)
            with st.spinner("分析中...",show_time=True):
                start = time.time()
                result = re.run(user_info)
                with st.sidebar.expander(label="TEST",expanded=True):
                    st.success(f"successfully(time:{time.time()-start:.1f}s)")
                    st.write(user_info)
                with st.expander("RECOMMENDATIONS", expanded=True):
                    st.write(result)
                    st.download_button(label="点击下载", data=result, file_name="体检推荐.md", use_container_width=True, icon="📥")




