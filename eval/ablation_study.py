"""Ablation study: compare different system configurations.

Configurations:
    A: Baseline (single agent + pure vector search)
    B: + Hybrid RAG (baseline + BM25+dense + reranker)
    C: + Multi-Agent (baseline + multi-agent architecture)
    D: + Query Decomp (B + query decomposition + ontology)
    E: + Reflection (C + self-reflection loop)
    F: Full System (all components)

Usage:
    python eval/ablation_study.py [--test-cases eval/test_cases.json]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from datetime import datetime
from eval.evaluator import Evaluator


ABLATION_CONFIGS = {
    "A_baseline": {
        "description": "Baseline: single agent + pure vector search",
        "use_multi_agent": False,
        "use_hybrid": False,
    },
    "B_hybrid_rag": {
        "description": "Baseline + Hybrid RAG (BM25+dense+reranker)",
        "use_multi_agent": False,
        "use_hybrid": True,
    },
    "C_multi_agent": {
        "description": "Baseline + Multi-Agent architecture",
        "use_multi_agent": True,
        "use_hybrid": False,
    },
    "D_query_decomp": {
        "description": "Hybrid RAG + Query Decomposition + Ontology",
        "use_multi_agent": False,
        "use_hybrid": True,
        "use_query_decomp": True,
    },
    "E_reflection": {
        "description": "Multi-Agent + Self-reflection loop (max 2 rounds)",
        "use_multi_agent": True,
        "use_hybrid": False,
        "max_revision": 2,
    },
    "F_full_system": {
        "description": "Full System: all components combined",
        "use_multi_agent": True,
        "use_hybrid": True,
        "max_revision": 2,
    },
}


class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def run_ablation_study(test_cases_path: str = "eval/test_cases.json"):
    """Run ablation study across all configurations."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Setup log file
    eval_dir = os.path.join(project_root, "eval")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(eval_dir, f"ablation_log_{timestamp}.txt")
    tee = Tee(log_path)
    sys.stdout = tee

    try:
        print(f"Ablation study started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_path}\n")

        # Load API key
        dotenv_path = os.path.join(project_root, ".env")
        if not os.path.exists(dotenv_path):
            dotenv_path = os.path.join(project_root, "src", ".env")
        if os.path.exists(dotenv_path):
            from dotenv import load_dotenv
            load_dotenv(dotenv_path)

        api_key = os.environ.get("DEEPSEEK_API_KEY")

        guidelines_path = os.path.join(project_root, "data", "clinical_guidelines.json")
        evaluator = Evaluator(guidelines_path=guidelines_path)

        # Load test cases
        tc_path = os.path.join(project_root, test_cases_path)
        if os.path.exists(tc_path):
            with open(tc_path, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
        else:
            from eval.run_eval import get_builtin_test_cases
            test_cases = get_builtin_test_cases()

        all_results = {}

        for config_name, config in ABLATION_CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"Config: {config_name}")
            print(f"  {config['description']}")
            print(f"{'='*60}")

            from src.hcr import Recommendation

            rec_system = Recommendation(
                api_key=api_key,
                use_multi_agent=config.get("use_multi_agent", False)
            )

            recommendation_cases = []
            total_latency = 0.0
            num_cases = 0

            for i, tc in enumerate(test_cases[:5]):
                profile = tc["profile"]
                print(f"  [{i+1}/{min(5, len(test_cases))}] {profile.get('id', '?')}...", end=" ")

                start = time.time()
                try:
                    result = rec_system.run(profile)
                    latency = time.time() - start
                    total_latency += latency
                    num_cases += 1

                    recommendation_cases.append({
                        "profile": profile,
                        "recommendation_text": result
                    })
                    print(f"OK ({latency:.1f}s)")
                except Exception as e:
                    print(f"ERROR: {e}")

            # Compute metrics
            metrics = evaluator.evaluate_recommendations(recommendation_cases) if recommendation_cases else {}
            avg_latency = total_latency / num_cases if num_cases > 0 else 0
            metrics["avg_latency"] = round(avg_latency, 2)

            all_results[config_name] = {
                "description": config["description"],
                "metrics": metrics,
                "num_cases_evaluated": num_cases
            }

        # Print comparison table
        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS")
        print("=" * 80)

        header = f"{'Config':<20} {'Accuracy':>10} {'Coverage':>10} {'Diversity':>10} {'Precision':>10} {'Latency':>10}"
        print(header)
        print("-" * 80)

        for config_name, data in all_results.items():
            m = data["metrics"]
            print(f"{config_name:<20} "
                  f"{m.get('Accuracy', 0):>10.3f} "
                  f"{m.get('Coverage', 0):>10.3f} "
                  f"{m.get('Diversity', 0):>10.3f} "
                  f"{m.get('Precision', 0):>10.3f} "
                  f"{m.get('avg_latency', 0):>9.1f}s")

        # Save results
        results_path = os.path.join(project_root, "eval", "results", "ablation_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {results_path}")
        print(f"\nLog saved to {log_path}")

    finally:
        sys.stdout = tee.terminal
        tee.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cases", default="eval/test_cases.json")
    args = parser.parse_args()
    run_ablation_study(args.test_cases)


if __name__ == "__main__":
    main()
