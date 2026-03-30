# import os
# from dotenv import load_dotenv

# load_dotenv('.env')

class Config:
    # Paths
    VECTORSTORE1_PATH = "/vectordb/vector_db_1"
    VECTORSTORE2_PATH = "/vectordb/vector_db_2"
    DATA = {
        "csv": "/data/health_check_data.csv",
        "pdf": "/data/symptoms.pdf"
    }

    # Hybrid Retrieval
    HYBRID_ALPHA = 0.6          # Dense weight in weighted fusion; BM25 gets 1-alpha
    HYBRID_FUSION = "rrf"       # "rrf" or "weighted"
    RRF_K = 60                  # RRF constant
    BM25_TOP_K = 20             # BM25 candidates
    DENSE_TOP_K = 20            # Dense vector candidates

    # Reranker
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    RERANK_TOP_K = 5

    # Query Decomposition
    SYNONYM_PATH = "/data/medical_synonyms.json"

    # Safety Rules
    SAFETY_RULES_PATH = "/data/safety_rules.json"

    # Clinical Guidelines (for evaluation)
    GUIDELINES_PATH = "/data/clinical_guidelines.json"