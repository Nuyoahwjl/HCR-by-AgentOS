### ![Powered by DeepSeek](https://img.shields.io/badge/Powered_by-DeepSeek_V3-0A0A0A?style=for-the-badge&logo=deepseek)


# 🩺 Health Check Recommendation (HCR)

A multi-agent RAG system for personalized health check package recommendation, built on the [AgentOS](https://github.com/QinbinLi/AgentOS) framework and powered by [DeepSeek V3](https://github.com/deepseek-ai/DeepSeek-V3).


## Features

- **Hybrid Retrieval**: BM25 + dense vector retrieval with Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: BAAI/bge-reranker-v2-m3 for precision improvement
- **Medical Synonym Expansion**: Ontology-aware query expansion (30+ synonym groups)
- **Query Decomposition**: Automatic splitting into symptom / history / demographic / risk sub-queries
- **Multi-Agent Architecture**: 4 specialized agents coordinated by an Orchestrator with reflection loop
- **Dual-Mode**: Switch between multi-agent pipeline and legacy single-agent mode
- **Evaluation Framework**: Retrieval, recommendation, and safety metrics with ablation study support
- **Streamlit Web UI**: 4 pages — Recommend, Chatbot, Hospitals, Report


## 🗂️ Folder Structure

```
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


## 🚀 How to Run

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd HCR-by-AgentOS

# Create conda environment (recommended)
conda create -n wjl python=3.10
conda activate wjl

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the template and fill in your DeepSeek API key
cp .env.example .env
# Edit .env: DEEPSEEK_API_KEY=sk-xxxxx
```

### 3. Build Vector Store

Run once after data changes:

```bash
python src/vectorstore.py
```

### 4. Test API Connection

```bash
python scripts/test_deepseek_api.py
```

### 5. Run the Web App

```bash
streamlit run web/🩺HCR-HOME.py
```

### 6. Run Evaluation (optional)

```bash
# Full evaluation
python eval/run_eval.py

# Ablation study (6 configurations, outputs to eval/ablation_log_*.txt)
python eval/ablation_study.py

# Direct CLI test
python src/hcr.py
```


## 🏗️ Architecture

### RAG Pipeline

```
User Query
    │
    ▼
Query Decomposer ──► symptom | history | demographic | risk sub-queries
    │
    ▼
Hybrid Retriever
    ├── BM25 Retriever (jieba tokenization)
    └── Dense Retriever (bge-base-zh embeddings)
    │
    ▼  Reciprocal Rank Fusion
Cross-Encoder Reranker (bge-reranker-v2-m3)
    │
    ▼
Context + Prompt → LLM (DeepSeek V3)
```

### Multi-Agent Pipeline

```
User Profile
    │
    ▼
┌─────────────────── Orchestrator (reflection loop) ───────────────────┐
│                                                                      │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
│  │ Symptom Analyzer│→ │ Risk Assessor│→ │ Recommendation Agent    │  │
│  │  (parsing, RAG) │  │ (risk score) │  │ (packages + rationale)  │  │
│  └─────────────────┘  └──────────────┘  └─────────────┬───────────┘  │
│                                                       │              │
│                                              ┌────────▼─────────┐    │
│                                              │ Safety Checker   │    │
│                                              │ (validate, flag) │    │
│                                              └──────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
Final Recommendation + Citations
```


## 💻 Tech Stack

| Component          | Technology                              |
|--------------------|-----------------------------------------|
| LLM                | DeepSeek V3 API (`deepseek-chat`)       |
| Agent Framework    | AgentOS (local)                         |
| Vector Database    | ChromaDB                                |
| Text Embedding     | BAAI/bge-base-zh                        |
| Reranker           | BAAI/bge-reranker-v2-m3                 |
| BM25               | rank-bm25 + jieba                       |
| Web UI             | Streamlit                               |
| Geolocation        | geopy (Nominatim), ip-api.com           |
| Map                | pydeck (Mapbox)                         |
| Data               | pandas, openpyxl, pypdf                 |
| PDF Generation     | fpdf                                    |


## 📊 Evaluation

The system includes an evaluation framework with three metric categories:

| Category         | Metrics                                  |
|------------------|------------------------------------------|
| Retrieval        | Precision, Recall, NDCG                  |
| Recommendation   | Coverage, Diversity, F1                  |
| Safety           | False positive rate, Pass rate           |

Run the ablation study to compare configurations:

| Config           | BM25 | Dense | Rerank | Synonyms |
|------------------|------|-------|--------|----------|
| Full Pipeline    |  ✓   |  ✓    |   ✓    |    ✓     |
| No Rerank        |  ✓   |  ✓    |   ✗    |    ✓     |
| No BM25          |  ✗   |  ✓    |   ✓    |    ✓     |
| No Dense         |  ✓   |  ✗    |   ✓    |    ✓     |
| No Synonyms      |  ✓   |  ✓    |   ✓    |    ✗     |
| Dense Only       |  ✗   |  ✓    |   ✗    |    ✗     |


## 🔒 Safety

- Safety rules validation against clinical guidelines
- Input sanitization for user profiles
- The Safety Checker agent flags potentially harmful recommendations
- API keys are stored in `.env` (gitignored) — never commit secrets
