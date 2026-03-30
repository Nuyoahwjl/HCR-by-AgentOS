"""Recommendation quality evaluation metrics.

Metrics: Accuracy vs guidelines, Coverage, Diversity, Precision.
"""
import json
import os
from typing import List, Set, Dict


def load_guidelines(guidelines_path: str) -> dict:
    """Load clinical guidelines from JSON file."""
    if not os.path.exists(guidelines_path):
        return {}
    with open(guidelines_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_recommendations(recommendation_text: str) -> List[str]:
    """Extract individual checkup items from recommendation text.

    Splits on common delimiters and cleans whitespace.
    """
    import re
    # Try to find list items
    items = re.split(r'[、，,\n]', recommendation_text)
    cleaned = []
    for item in items:
        item = item.strip()
        # Remove numbering, bullets
        item = re.sub(r'^[\d\.\-\*\s]+', '', item).strip()
        if item and len(item) > 1:
            cleaned.append(item)
    return cleaned


def get_expected_items(profile: dict, guidelines: dict) -> Set[str]:
    """Get expected checkup items based on clinical guidelines.

    Args:
        profile: Patient profile dict.
        guidelines: Clinical guidelines dict.

    Returns:
        Set of expected checkup item names.
    """
    expected = set()
    try:
        age = int(profile.get("age", 0))
    except (ValueError, TypeError):
        age = 0
    gender = profile.get("gender", "")
    history = profile.get("medical_history", "")
    symptoms = profile.get("symptoms", "")

    # Age-based recommendations
    if age <= 18:
        expected.update(["血常规", "尿常规", "视力检查"])
    elif age <= 40:
        expected.update(["血常规", "血压监测", "血脂检查"])
    elif age <= 60:
        expected.update(["血常规", "血压监测", "血脂检查", "心电图", "血糖检测"])
    else:
        expected.update(["血常规", "血压监测", "血脂检查", "心电图", "血糖检测",
                        "肿瘤标志物", "骨密度"])

    # Gender-based
    if gender == "female":
        expected.update(["妇科检查", "乳腺检查"])
    elif gender == "male" and age > 45:
        expected.add("前列腺检查")

    # Condition-based
    if "高血压" in history:
        expected.update(["血压监测", "心电图", "血脂检查"])
    if "糖尿病" in history:
        expected.update(["血糖检测", "肾功能", "眼科检查"])
    if "高血脂" in history:
        expected.add("血脂检查")

    # Custom guidelines from file
    if "age_rules" in guidelines:
        for rule in guidelines["age_rules"]:
            if rule.get("min_age", 0) <= age <= rule.get("max_age", 200):
                for item in rule.get("items", []):
                    expected.add(item)

    if "condition_rules" in guidelines:
        for rule in guidelines["condition_rules"]:
            condition = rule.get("condition", "")
            if condition in history:
                for item in rule.get("items", []):
                    expected.add(item)

    return expected


def accuracy_vs_guidelines(
    recommended: List[str],
    expected: Set[str]
) -> float:
    """Calculate fraction of expected items that are recommended.

    Args:
        recommended: List of recommended items.
        expected: Set of expected items per guidelines.

    Returns:
        Accuracy score (0-1).
    """
    if not expected:
        return 1.0
    rec_set = set(recommended)
    return len(rec_set & expected) / len(expected)


def coverage(recommended: List[str], expected: Set[str]) -> float:
    """Same as accuracy - how many expected items are covered."""
    return accuracy_vs_guidelines(recommended, expected)


def diversity(recommended: List[str]) -> float:
    """Measure diversity of recommendation categories.

    Higher score means more diverse categories.
    Normalized to 0-1.
    """
    # Define category groups
    categories = {
        "血液": ["血常规", "血脂检查", "血糖检测", "肝功能", "肾功能"],
        "心血管": ["心电图", "血压监测", "心脏彩超", "颈动脉超声"],
        "影像": ["胸片", "腹部B超", "胃肠镜"],
        "专项": ["肿瘤标志物", "骨密度", "甲状腺功能", "眼科检查"],
        "性别": ["妇科检查", "乳腺检查", "前列腺检查"],
    }

    covered_categories = set()
    for item in recommended:
        for cat, members in categories.items():
            if item in members:
                covered_categories.add(cat)
                break

    return len(covered_categories) / len(categories)


def precision(recommended: List[str], expected: Set[str]) -> float:
    """Calculate fraction of recommended items that are appropriate.

    Args:
        recommended: List of recommended items.
        expected: Set of appropriate items per guidelines.

    Returns:
        Precision score (0-1).
    """
    if not recommended:
        return 0.0
    rec_set = set(recommended)
    return len(rec_set & expected) / len(rec_set)


def evaluate_recommendations_batch(
    evaluations: List[dict],
    guidelines: dict = None
) -> dict:
    """Evaluate recommendation quality on a batch of cases.

    Args:
        evaluations: List of dicts with keys:
            - profile: Patient profile dict
            - recommendation_text: Raw recommendation text
        guidelines: Clinical guidelines dict.

    Returns:
        Dict with metric results.
    """
    if guidelines is None:
        guidelines = {}

    accs, covs, divs, precs = [], [], [], []

    for ev in evaluations:
        profile = ev["profile"]
        rec_text = ev["recommendation_text"]

        recommended = parse_recommendations(rec_text)
        expected = get_expected_items(profile, guidelines)

        accs.append(accuracy_vs_guidelines(recommended, expected))
        covs.append(coverage(recommended, expected))
        divs.append(diversity(recommended))
        precs.append(precision(recommended, expected))

    n = len(evaluations) if evaluations else 1
    return {
        "Accuracy": sum(accs) / n if accs else 0.0,
        "Coverage": sum(covs) / n if covs else 0.0,
        "Diversity": sum(divs) / n if divs else 0.0,
        "Precision": sum(precs) / n if precs else 0.0,
    }
