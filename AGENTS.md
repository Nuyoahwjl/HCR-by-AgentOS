# AGENTS.md — HCR-by-AgentOS

Developer guide for the Health Check Recommendation system. Multi-agent RAG architecture using AgentOS framework, DeepSeek V3, ChromaDB, and BAAI/bge-base-zh embeddings. Web UI via Streamlit.

## Project Structure

```
config/              - Settings and configuration (settings.py)
src/                 - Core application logic
  src/agents/        - Multi-agent system (coordinator, 4 specialized agents)
  hcr.py             - Recommendation engine (dual-mode)
  tools.py           - Agent tool definitions
  prompt.py          - System prompts
  vectorstore.py     - Vector store builder
  report.py          - Report generation
  ontology.py        - Medical ontology (synonym expansion)
  query_decomposer.py- Query decomposition
agentos/             - AgentOS framework (internal, not a third-party package)
  agent/             - Base agent with reason-act loop
  rag/               - RAG components (store, embedding, bm25, hybrid, rerank, split)
  utils/             - DeepSeek API wrapper
  memory/            - Agent memory management
  prompt/            - Prompt templates
  tools/             - Framework tool infrastructure
eval/                - Evaluation framework
  evaluator.py       - Base evaluator
  retrieval_eval.py  - Retrieval metrics
  recommendation_eval.py - Recommendation metrics
  safety_eval.py     - Safety evaluation
  run_eval.py        - Full evaluation runner
  ablation_study.py  - Ablation study (6 configs)
  test_cases.json    - Labeled test cases
scripts/             - Utility scripts
  generate_synthetic_data.py - Synthetic data generation
  validate_synthetic_data.py - Data validation
  test_deepseek_api.py       - API connectivity test
web/                 - Streamlit frontend
  pages/             - Recommend, Chatbot, Hospitals, Report
data/                - CSV, PDF, JSON data files
test/                - Manual test scripts (hcr_test.py)
vectordb/            - ChromaDB vector store directories (generated)
```

## Build / Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build vector store (run once after data changes)
python src/vectorstore.py

# Run the Streamlit web app
streamlit run web/🩺HCR-HOME.py

# Run the recommendation engine directly (CLI test)
python src/hcr.py

# Test DeepSeek API connectivity
python scripts/test_deepseek_api.py

# Run the test script
python test/hcr_test.py

# Run evaluation
python eval/run_eval.py

# Run ablation study (outputs to eval/ablation_log_*.txt)
python eval/ablation_study.py

# Generate synthetic data
python scripts/generate_synthetic_data.py
```

## Testing

There is no formal test framework (no pytest/unittest). Tests are manual scripts in `test/`.

- `python test/hcr_test.py` — Runs the Agent with a hardcoded user profile, calls tools, and prints memory/response output.
- `python eval/run_eval.py` — Evaluates retrieval, recommendation, and safety metrics.
- `python scripts/test_deepseek_api.py` — Verifies DeepSeek API connectivity.
- To test a specific module, run it directly: `python src/hcr.py`, `python src/report.py`

No lint (`ruff`, `flake8`), format (`black`), or type-check (`mypy`) commands are configured.

## Environment Setup

1. Create a conda environment:
   ```bash
   conda create -n wjl python=3.10
   conda activate wjl
   ```
2. Create a `.env` file in project root:
   ```
   DEEPSEEK_API_KEY=sk-xxxxx
   ```
3. The `.env` file is gitignored — never commit secrets.
4. Always run commands inside the `wjl` conda environment.

## API Configuration

- LLM calls use DeepSeek official API via `agentos/utils/utils.py`
- API endpoint: `https://api.deepseek.com` (model: `deepseek-chat`)
- Uses `openai.OpenAI` client with custom `base_url`
- IP geolocation uses `ip-api.com` (free, no key)
- Geocoding uses Nominatim via `geopy` (free, no key)

## Code Style

### Imports
- Standard library first, then third-party, then local modules.
- Each file adds the project root to `sys.path` at the top:
  ```python
  import sys, os
  current_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.dirname(current_dir)  # adjust depth as needed
  sys.path.insert(0, project_root)
  ```
- Use `from module import Class/function` for local imports.
- Avoid `import *` except in `src/tools.py` where all tool classes are exported.

### Naming
- **Classes**: PascalCase (e.g., `Recommendation`, `TemporaryMemory`, `ChromaDB`)
- **Agent classes**: PascalCase (e.g., `SymptomAnalyzer`, `RiskAssessor`, `MedicalAgent`)
- **Tool classes**: lowercase_with_underscores (e.g., `search_by_id`, `recommend_by_age`) — callable tools for the agent
- **Functions/methods**: snake_case (e.g., `call_model`, `parse_tool_info`, `add_memory`)
- **Constants/prompts**: UPPER_SNAKE_CASE (e.g., `HCR_PROMPT`, `OUTPUT_PROMPT`)
- **Variables**: snake_case

### Types
- Use `str | None` (Python 3.10+ union syntax) for optional parameters.
- Use `List` from `typing` for list type hints (see `agentos/agent/agent.py`).
- Type hints are used inconsistently — prefer adding them to new code.

### Docstrings
- Tool classes use docstrings in the class and in the `run()` method.
- The `run()` method docstring follows this format:
  ```python
  def run(self, arg1: type):
      """
      function_name:brief description in Chinese
      Args:
          arg1 (type): description
      Returns:
          str: description
      """
  ```
- Module-level comments are in Chinese.

### Error Handling
- Minimal error handling throughout the codebase.
- Functions return result strings or `0` for not-found cases (see `report.py`).
- No try/except patterns established — use judgment for new code.

### General Patterns
- Configuration lives in `config/settings.py` as a `Config` class with class-level attributes.
- Database connections use SQLite (`sqlite3`) with inline SQL.
- Vector DB operations go through `agentos.rag.store.ChromaDB`.
- LLM calls go through `agentos.utils.call_model(messages, api_key)`.
- The Agent uses a reason-act loop: `reason()` parses model output for thought/function/args, `act()` executes the tool.
- Multi-agent coordination goes through `src/agents/coordinator.py` Orchestrator.

## Key Dependencies

| Purpose          | Library                            |
|------------------|------------------------------------|
| LLM API          | `openai` (DeepSeek via base_url)   |
| Vector DB        | `chromadb`                         |
| Embeddings       | `sentence-transformers`, `transformers` |
| BM25             | `rank-bm25`, `jieba`               |
| Reranker         | `FlagEmbedding`                    |
| Web UI           | `streamlit`                        |
| Geolocation      | `geopy`, `pydeck`                  |
| Data             | `pandas`, `openpyxl`, `pypdf`     |
| PDF generation   | `fpdf`                             |

## Adding New Tools

Follow the class pattern in `src/tools.py`:

```python
class my_new_tool:
    def __init__(self):
        pass

    def run(self, arg1: str):
        """
        my_new_tool:brief description
        Args:
            arg1 (str): description
        Returns:
            str: description
        """
        # implementation
        return result
```

Then export it in `src/tools.py`:
```python
from tools import my_new_tool  # add to imports
```

## Notes for Agents

- Do not commit `.env`, `__pycache__/`, or `__init__.py` files (they are gitignored).
- The `agentos/` directory is a local framework — treat it as internal code, not a third-party package.
- When adding new tools, follow the class pattern in `src/tools.py`: class with `__init__` and `run()` method with docstring.
- Streamlit pages live in `web/pages/` and are numbered for sidebar ordering (e.g., `1_🥰_Recommend.py`).
- The system supports dual-mode in `src/hcr.py`: set `use_multi_agent=True` for the multi-agent pipeline, `False` for legacy single-agent.
- Multi-agent conversation history is stored in `coordinator.context.conversation_history`.
