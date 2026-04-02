import streamlit as st
import os
from pathlib import Path
current_dir = Path(__file__).parent
img_dir = current_dir.parent / "img"
# print(list(img_dir.glob("*.png")))

st.set_page_config(page_title="HCR", page_icon="🩺")

st.markdown("""
<div style="text-align: center;">
    <img src="https://placehold.co/800x200/009688/FFFFFF/png?text=AI+Health+Check+Assistant&font=Lora" 
         style="display: block; margin: auto; width: 100%;">
</div>
""", unsafe_allow_html=True)

st.markdown("-------------")

st.markdown(
"""
## 📖 Project Overview  
This project is an intelligent **Health Check Recommendation System** that suggests personalized medical examination packages using:  
- 🧠 **RAG (Retrieval-Augmented Generation) technology**  
- ⚡ **DeepSeek V3** for natural language processing  
- 🔍 **Chormadb** vector database for efficient similarity search  
- 🎯 **AgentOS** framework for pipeline orchestration  

Designed to bridge medical knowledge with individual needs through AI-powered analysis.  

---

## 🗂️ Project Structure  
```bash
HCR-by-AgentOS/
├── config/
│   └── settings.py                  # Configuration (hybrid, rerank, synonym settings)
├── data/
│   ├── health_check_data.csv        # Health check package database
│   ├── symptoms.pdf                 # Medical symptoms reference
│   ├── medical_synonyms.json        # Medical synonym ontology
│   ├── clinical_guidelines.json     # Clinical guidelines for safety checks
│   └── safety_rules.json            # Safety rules
├── agentos/                         # AgentOS framework (internal)
│   ├── agent/
│   │   └── agent.py                 # Base agent with reason-act loop
│   ├── memory/
│   ├── prompt/
│   ├── rag/
│   │   ├── store.py                 # ChromaDB vector store
│   │   ├── embedding.py             # BAAI/bge-base-zh embeddings
│   │   ├── bm25_retriever.py        # BM25 keyword retriever
│   │   ├── hybrid_retriever.py      # Hybrid BM25 + dense retriever
│   │   ├── rerank.py                # Cross-encoder reranker
│   │   ├── split.py                 # Semantic text splitter
│   │   ├── data.py                  # Document loader
│   │   └── load.py                  # Data ingestion utilities
│   ├── tools/
│   └── utils/
│       └── utils.py                 # DeepSeek API wrapper
├── src/
│   ├── hcr.py                       # Recommendation engine (dual-mode)
│   ├── tools.py                     # Agent tool definitions
│   ├── prompt.py                    # System prompts
│   ├── vectorstore.py               # Vector store builder
│   ├── report.py                    # Report generation
│   ├── ontology.py                  # Medical ontology
│   ├── query_decomposer.py          # Query decomposition
│   └── agents/                      # Multi-agent system
│       ├── base_agent.py            # MedicalAgent base class
│       ├── coordinator.py           # Orchestrator with reflection
│       ├── symptom_analyzer.py      # Symptom analysis agent
│       ├── risk_assessor.py         # Risk assessment agent
│       ├── recommendation_agent.py  # Recommendation agent
│       ├── safety_checker.py        # Safety validation agent
│       ├── message.py               # AgentMessage dataclass
│       ├── context.py               # AgentContext shared state
│       └── citation.py              # Citation tracker
├── eval/
│   ├── evaluator.py                 # Base evaluator
│   ├── retrieval_eval.py            # Retrieval metrics (precision, recall, NDCG)
│   ├── recommendation_eval.py       # Recommendation metrics (coverage, diversity)
│   ├── safety_eval.py               # Safety evaluation
│   ├── run_eval.py                  # Full evaluation runner
│   ├── ablation_study.py            # Ablation study (6 configurations)
│   └── test_cases.json              # Labeled test cases
├── scripts/
│   ├── generate_synthetic_data.py   # Synthetic data generation
│   ├── validate_synthetic_data.py   # Data validation
│   └── test_deepseek_api.py         # DeepSeek API connectivity test
├── web/
│   ├── 🩺HCR-HOME.py                # Streamlit entry point
│   └── pages/
│       ├── 1_🥰_Recommend.py        # Recommendation page
│       ├── 2_🤖_Chatbot.py          # Chatbot page
│       ├── 3_🏥_Hospitals.py        # Nearby hospitals page
│       └── 4_📄_Report.py           # Report generation page
├── test/
│   └── hcr_test.py                  # Manual integration test
├── vectordb/                        # ChromaDB vector store (generated)
│   ├── vector_db_1/
│   └── vector_db_2/
├── requirements.txt
├── .env.example                     # API key template
└── README.md
```

---

## ✨ Key Features  
- **Personalized Recommendations**  
  🔍 Analyzes user profile + medical history → suggests tailored checkup packages  

- **Multi-Source Knowledge**  
  📚 Combines structured data (CSV) + unstructured documents (PDF)  

- **Modular Architecture**  
  📢 Separates data processing, AI logic, and UI layers  
 
- **User-Friendly Interface**  
  💻 Streamlit web app with guided conversation flow 

---

## 🛠️ Tech Stack  

"""
)

st.markdown("""
<div>
<style>
.tech-table {
    width: 100% !important;
    table-layout: fixed;
    border-collapse: collapse;
    margin: auto;
    font-family: Arial, sans-serif;
    background: transparent !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.tech-table th,
.tech-table td {
    width: 50% !important;
    padding: 12px;
    text-align: left;
    border-bottom: 2px solid rgba(222, 226, 230, 0.5); /* 半透明边框 */
    word-break: break-word;
    box-sizing: border-box;
    background: transparent !important;
}
.tech-table th {
    border-bottom: 3px solid rgba(73, 80, 87, 0.8); /* 深色半透明边框 */
    font-weight: 600;
}
@media screen and (max-width: 600px) {
    .tech-table {
        font-size: 14px;
        box-shadow: none; /* 小屏幕移除阴影 */
    }
    .tech-table td, 
    .tech-table th {
        padding: 8px;
    }
}
</style>
<table class="tech-table">
    <colgroup>
        <col style="width: 50%;">
        <col style="width: 50%;">
    </colgroup>
    <tr>
        <th>Component</th>
        <th>Technology</th>
    </tr>
    <tr>
        <td><strong>LLM</strong></td>
        <td>DeepSeek V3 API (`deepseek-chat`)</td>
    </tr>
    <tr>
        <td><strong>Agent Framework</strong></td>
        <td>AgentOS (local)</td>
    </tr>
    <tr>
        <td><strong>Vector Database</strong></td>
        <td>ChromaDB</td>
    </tr>
    <tr>
        <td><strong>Text Embedding</strong></td>
        <td>BAAI/bge-base-zh</td>
    </tr>
    <tr>
        <td><strong>Reranker</strong></td>
        <td>BAAI/bge-reranker-v2-m3</td>
    </tr>
    <tr>
        <td><strong>BM25</strong></td>
        <td>rank-bm25 + jieba</td>
    </tr>
    <tr>
        <td><strong>Web UI</strong></td>
        <td>Streamlit</td>
    </tr>
    <tr>
        <td><strong>Geolocation</strong></td>
        <td>geopy (Nominatim), ip-api.com</td>
    </tr>
    <tr>
        <td><strong>Map</strong></td>
        <td>pydeck (Mapbox)</td>
    </tr>
    <tr>
        <td><strong>Data</strong></td>
        <td>pandas, openpyxl, pypdf</td>
    </tr>
    <tr>
        <td><strong>PDF Generation</strong></td>
        <td>fpdf</td>
    </tr>
</table>
</div>
""", unsafe_allow_html=True)

st.markdown("-------------")

st.warning("Let's build smarter healthcare together! 🌟 ")








# st.markdown("""
# 🛠️ Tech Stack  
# | Component                | Technology           |  
# |--------------------------|----------------------|  
# | **Large Language Model** | DeepSeek API         |  
# | **Framework**            | LangChain            |  
# | **Vector Database**      | FAISS                |  
# | **Frontend**             | Streamlit            |  
# | **Embeddings**           | BAAI/bge-base-zh     |  
# | **Environment**          | Python 3.12.9        |            
# ---          
# """)







with st.sidebar:
    st.success("Select one page above")
    # st.markdown("Created by [Chia.le](https://github.com/Nuyoahwjl)")
    # st.markdown("Contact me [📮](chia.le@foxmail.com)")
    # st.markdown(
    # """
    #   <picture>
    #     <img src="https://raw.githubusercontent.com/Nuyoahwjl/Nuyoahwjl/output/github-contribution-grid-snake.svg"/>
    #   </picture>
    # """, unsafe_allow_html=True
    # )
    
