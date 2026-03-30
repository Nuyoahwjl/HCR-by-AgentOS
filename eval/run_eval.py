"""Run evaluation on the HCR system.

Usage:
    python eval/run_eval.py [--config all] [--test-cases eval/test_cases.json]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from eval.evaluator import Evaluator


def load_test_cases(path: str) -> list:
    """Load test cases from JSON file."""
    if not os.path.exists(path):
        print(f"Test cases file not found: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_evaluation(config: str = "multi_agent", test_cases_path: str = "eval/test_cases.json"):
    """Run the full evaluation.

    Args:
        config: Configuration to evaluate ("legacy", "multi_agent").
        test_cases_path: Path to test cases JSON.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    guidelines_path = os.path.join(project_root, "data", "clinical_guidelines.json")
    evaluator = Evaluator(guidelines_path=guidelines_path)

    test_cases = load_test_cases(os.path.join(project_root, test_cases_path))
    if not test_cases:
        print("No test cases found. Using built-in examples.")
        test_cases = get_builtin_test_cases()

    # Import the recommendation system
    from src.hcr import Recommendation

    dotenv_path = os.path.join(project_root, "src", ".env")
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)

    import os as os_mod
    api_key = os_mod.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("WARNING: DEEPSEEK_API_KEY not set. Evaluation will fail on API calls.")

    rec_system = Recommendation(api_key=api_key, use_multi_agent=(config == "multi_agent"))

    # Run evaluation
    recommendation_cases = []
    safety_cases = []
    total_latency = 0.0
    num_cases = 0

    print(f"\nRunning evaluation with config: {config}")
    print(f"Test cases: {len(test_cases)}\n")

    for i, tc in enumerate(test_cases):
        profile = tc["profile"]
        print(f"[{i+1}/{len(test_cases)}] Evaluating: {profile.get('id', 'unknown')}...")

        start_time = time.time()
        try:
            result = rec_system.run(profile)
            latency = time.time() - start_time
            total_latency += latency
            num_cases += 1

            recommendation_cases.append({
                "profile": profile,
                "recommendation_text": result
            })

            safety_cases.append({
                "profile": profile,
                "recommendation_text": result,
                "evidence": tc.get("evidence", []),
                "critical_items": set(tc.get("critical_items", []))
            })

            print(f"  Latency: {latency:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Compute metrics
    results = evaluator.evaluate_full(
        recommendation_cases=recommendation_cases if recommendation_cases else None,
        safety_cases=safety_cases if safety_cases else None,
        latency=total_latency / num_cases if num_cases > 0 else None
    )

    # Print and save
    print("\n" + evaluator.format_results(results))

    results_path = os.path.join(project_root, "eval", "results", f"eval_{config}.json")
    evaluator.save_results(results, results_path)
    print(f"Results saved to {results_path}")

    return results


def get_builtin_test_cases() -> list:
    """Built-in test cases when no JSON file exists."""
    return [
        {
            "profile": {
                "id": "000001", "gender": "male", "age": "50",
                "height": "172", "weight": "80",
                "medical_history": "高血压", "symptoms": "头晕"
            },
            "evidence": ["患者ID:671203,性别:女,年龄(岁):45,既往病史:高血压,体检前的症状:头晕,体检项目:血常规、血压监测、血脂检查"],
            "critical_items": ["血压监测", "心电图"]
        },
        {
            "profile": {
                "id": "000002", "gender": "female", "age": "35",
                "height": "160", "weight": "55",
                "medical_history": "无", "symptoms": "乏力"
            },
            "evidence": [],
            "critical_items": ["血常规"]
        },
        {
            "profile": {
                "id": "000003", "gender": "male", "age": "65",
                "height": "170", "weight": "85",
                "medical_history": "糖尿病、高血脂", "symptoms": "视力模糊"
            },
            "evidence": ["相似病例:糖尿病+高血脂患者需关注血糖、血脂、肾功能、眼科"],
            "critical_items": ["血糖检测", "血脂检查", "肾功能", "眼科检查"]
        },
        {
            "profile": {
                "id": "000004", "gender": "female", "age": "25",
                "height": "165", "weight": "52",
                "medical_history": "无", "symptoms": "无"
            },
            "evidence": [],
            "critical_items": ["血常规"]
        },
        {
            "profile": {
                "id": "000005", "gender": "male", "age": "55",
                "height": "175", "weight": "90",
                "medical_history": "痛风", "symptoms": "关节痛"
            },
            "evidence": ["痛风患者需关注尿酸、肾功能"],
            "critical_items": ["肾功能"]
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="multi_agent",
                        choices=["legacy", "multi_agent", "all"])
    parser.add_argument("--test-cases", default="eval/test_cases.json")
    args = parser.parse_args()

    if args.config == "all":
        for cfg in ["legacy", "multi_agent"]:
            run_evaluation(config=cfg, test_cases_path=args.test_cases)
    else:
        run_evaluation(config=args.config, test_cases_path=args.test_cases)


if __name__ == "__main__":
    main()
