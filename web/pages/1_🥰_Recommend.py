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


def format_user_info(gender, age, height, weight, medical_history, symptoms, id="000000", budget=1000):
    """格式化用户信息"""
    return {
        "id":id,
        "gender": gender,
        "age": age,
        "height": height,
        "weight": weight,
        "medical_history": medical_history,
        "symptoms": symptoms,
        "budget": budget
    }


# 界面布局
st.set_page_config(
    page_title="Recommend",
    page_icon="🥰",
)



st.markdown("""
<div style="text-align: center;">
    <h2><b>🩺Health Check Recommendation</b></h2>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center;">
    <img src="https://cdn.jsdelivr.net/gh//Nuyoahwjl/Nuyoahwjl/HCR/typing.svg" 
         style="display: block; margin: auto; width: 100%;">
</div>
""", unsafe_allow_html=True)


# @st.dialog("Input your together.ai API")
# def input():
#     api=st.text_input("API", type="password", key="api_key" ,help="https://api.together.xyz/")
#     submitted = st.button("Confirm", icon='✔️', use_container_width=True)
#     if submitted:
#         st.session_state.TOGETHER_AI_API=api

# with st.sidebar:
#     st.header("⚙️ Setting")
#     st.button("Input API", on_click=input,use_container_width=True)
#     if "TOGETHER_AI_API" not in st.session_state or not st.session_state.TOGETHER_AI_API.startswith("tgp"):
#         st.warning("Please enter API!", icon="⚠️")


TOGETHER_AI_API = st.text_input("TOGETHER_AI_API", type="password" ,help="https://api.together.xyz/")
if not TOGETHER_AI_API.startswith("tgp"):
    st.warning("Please enter API!", icon="⚠️")





id=st.text_input("ID(6 figures)", key="id", help="Please enter your ID")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox(label="Gender", 
                          options=["male", "female", "secret"],
                          format_func = str,
                          help = "if you don't want to tell us, keep secret")
    height = st.slider("Height(cm)", 0, 200, 50)
with col2:
    age = st.number_input("Age", 
                          min_value=0, 
                          max_value=100)
    weight = st.slider("Weight(kg)", 0, 100, 40)
medical_history = st.text_area("Medical History", key="medical", height=100)
symptoms = st.text_area("Symptoms", key="symptoms", height=100)

# 添加费用限制输入框
budget = st.number_input("Maximum Budget (¥)", 
                        min_value=0, 
                        max_value=10000,
                        value=1000,
                        step=100,
                        help="Enter your maximum budget for the health checkup")

# 添加项目费用显示
st.markdown("### 💰 Common Checkup Items and Prices")
prices = {
    "血常规": 50,
    "血压监测": 30,
    "血脂检查": 100,
    "心电图": 80,
    "血糖检测": 60,
    "眼科检查": 120,
    "超声心动图": 200,
    "甲状腺功能": 150,
    "骨密度": 180,
    "肿瘤标志物": 300
}

# 显示项目费用表格
price_df = pd.DataFrame(list(prices.items()), columns=['项目', '费用(¥)'])
st.dataframe(price_df, use_container_width=True)

submitted = st.button("Recommend", icon='✔️', use_container_width=True)



if submitted:
    if not TOGETHER_AI_API.startswith("tgp"):
        pass
    else:
        if height == 50 or age == 0 or weight == 40 or not medical_history.strip() or not symptoms.strip():
            st.error("Please fill in all the information", icon="🚨")
        else:
            re= Recommendation(TOGETHER_AI_API)
            user_info = format_user_info(
                gender=gender,
                age=age,
                height=height,
                weight=weight,
                medical_history=medical_history,
                symptoms=symptoms,
                id=id,
                budget=budget
            )
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




# if submitted:
#     if height == 50 or age ==0 or weight ==0 or not medical_history.strip() or not symptoms.strip():
#         st.error("Please fill in all the information", icon="🚨")
#         # st.toast("Please fill in all the information", icon="🚨")
#     else:
#         user_info = format_user_info(gender, age, height, weight, medical_history, symptoms, id)
#         with st.spinner("analyzing...",show_time=True):
#             start = time.time()
#             result = re.run(user_info)
#             with st.sidebar.expander(label="TEST",expanded=True):
#                 st.success(f"successfully(time:{time.time()-start:.1f}s)")
#                 st.write(user_info)
#             with st.expander("RECOMMENDATIONS", expanded=True):
#                 st.markdown("## RECOMMENDATIONS")
#                 st.write(result)
#                 st.download_button(label="Download", data=result, file_name="Recommendations.md", use_container_width=True, icon="📥")