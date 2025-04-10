import streamlit as st

st.set_page_config(page_title="HCR", page_icon="🩺")

st.markdown(
"""
<div style="display: flex; justify-content: center;">
<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Tarikul-Islam-Anik/Animated-Fluent-Emojis@master/Emojis/Smilies/Face%20with%20Tongue.png" width="10%" />
  <img src="https://cdn.jsdelivr.net/gh/Tarikul-Islam-Anik/Animated-Fluent-Emojis@master/Emojis/Smilies/Face%20with%20Spiral%20Eyes.png" width="10%" />
  <img src="https://cdn.jsdelivr.net/gh/Tarikul-Islam-Anik/Animated-Fluent-Emojis@master/Emojis/Smilies/Relieved%20Face.png" width="10%" />
  <img src="https://cdn.jsdelivr.net/gh/Tarikul-Islam-Anik/Animated-Fluent-Emojis@master/Emojis/Smilies/Astonished%20Face.png" width="10%" />
  <img src="https://cdn.jsdelivr.net/gh/Tarikul-Islam-Anik/Animated-Fluent-Emojis@master/Emojis/Smilies/Beaming%20Face%20with%20Smiling%20Eyes.png" width="10%" />
</p>	
</div>
""", unsafe_allow_html=True)

st.markdown("-------------")

# 项目简介
st.markdown("""
<div style="text-align: center; max-width: 800px; margin: 0 auto;">
    <h3>📖 项目概述</h3>
    <p style="font-size: 16px; line-height: 1.6;">
        本系统是基于大语言模型的<strong>个性化体检推荐平台</strong>，融合：<br>
        🧠 <strong>RAG检索增强生成技术</strong> + 
        ⚡ <strong>DeepSeek V3</strong>大模型 + 
        🔍 <strong>ChromaDB</strong>向量数据库 + 
        🎯 <strong>AgentOS</strong>框架<br>
        通过智能分析实现医疗知识与个人需求的精准匹配
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# 技术栈
st.markdown("""
<div style="text-align: center;">
    <h3>🛠️ 技术架构</h3>
    <table style="
        margin: 0 auto;
        border-collapse: collapse;
        width: 70%;
        font-family: Arial;
    ">
        <tr style="background: #f8f9fa;">
            <th style="padding: 12px; border-bottom: 2px solid #dee2e6;">组件</th>
            <th style="padding: 12px; border-bottom: 2px solid #dee2e6;">技术方案</th>
        </tr>
        <tr><td>大语言模型</td><td>DeepSeek V3</td></tr>
        <tr><td>开发框架</td><td>AgentOS</td></tr>
        <tr><td>向量数据库</td><td>ChromaDB</td></tr>
        <tr><td>前端框架</td><td>Streamlit</td></tr>
        <tr><td>文本嵌入</td><td>BAAI/bge-base-zh</td></tr>
    </table>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# 结尾
st.markdown("""
<div style="text-align: center; margin: 40px 0;">
    <h3 style="color: #009688;">🌟 让我们共同构建更智慧的健康未来！</h3>
    <img src="https://placehold.co/800x200/009688/FFFFFF/png?text=AI+Health+Check+Assistant&font=Lora" 
         style="width: 80%; border-radius: 10px; margin-top: 20px;">
</div>
""", unsafe_allow_html=True)










with st.sidebar:
    st.success("Select one page above")
    # st.markdown("Created by [Chia.le](https://github.com/Nuyoahwjl)")
    # st.markdown("Contact me [📮](chia.le@foxmail.com)")
    
