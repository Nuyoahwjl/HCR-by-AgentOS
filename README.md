### ![Powered by Qwen-max](https://img.shields.io/badge/Powered_by-Qwen-max-0A0A0A?style=for-the-badge&logo=deepseek)


# 🩺Health Check Recommendation🩺
This project is a health check recommendation system built using AgentOS, and Qwen-max LLM. It uses RAG technology to recommend health check packages based on user information.


### 🗂️Folder Structure🗂️
```
HCR-by-AgentOS/
├── config/
│   └── settings.py
├── data/
│   ├──health_check_data.csv
│   └──symptoms.pdf
├── vectordb/
│   └── vector_db_1/
│   └── vector_db_2/
├── src/
│   ├── vectorstore.py
│   ├── hcr.py
│   └── hcr_prompts.py
│   └── tools.py
│   └── utils.py
├── test/
├── web/
│   ├── pages/
│   │   └── 1_🥰_Recommend.py
│   │   └── 2_🤖_Chatbot.py
│   │   └── 3_🏥_Hospitals.py
│   └── 🩺HCR-HOME.py
├── requirements.txt
└── README.md
```


### 🚀How to Run🚀

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Build Vector Store:
```bash
python vectordb/vectorstore.py
```
3. Run the Streamlit app:
```bash
streamlit run web/🩺HCR-HOME.py
```


### 💻Tech Stack💻

| Component          | Technology Selection     |
|--------------------|--------------------------|
| Large Model        | Qwen-max API             |
| Framework          | AgentOS                  |
| Vector Database    | Chromadb                 |
| Frontend           | Streamlit                |
| Text Embedding     | BAAI/bge-base-zh         |
| Cross-Encoder      | ms-marco-MiniLM-L6-v2    |