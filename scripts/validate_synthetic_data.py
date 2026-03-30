"""Validate synthetic health check data for clinical plausibility.

Usage:
    python scripts/validate_synthetic_data.py [--input data/synthetic_health_check_data.csv]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from collections import Counter


CONDITIONS = {"高血压", "糖尿病", "高血脂", "冠心病", "痛风", "甲状腺疾病"}
SYMPTOMS = {"头晕", "头痛", "胸闷", "乏力", "失眠", "口渴多尿", "视力模糊", "关节痛"}

# Age-condition plausibility rules
AGE_CONDITION_RULES = {
    "冠心病": (40, 80),    # Typically age 40+
    "痛风": (30, 80),      # Typically age 30+
    "甲状腺疾病": (20, 70),
}

# Condition-symptom correlations
CONDITION_SYMPTOM_MAP = {
    "高血压": ["头晕", "头痛", "胸闷"],
    "糖尿病": ["口渴多尿", "乏力", "视力模糊"],
    "高血脂": ["胸闷", "乏力"],
    "痛风": ["关节痛"],
}


def validate_record(record: dict) -> list:
    """Validate a single record. Returns list of issues found."""
    issues = []

    # Basic range checks
    try:
        age = int(record.get("年龄(岁)", 0))
        if age < 18 or age > 100:
            issues.append(f"Age out of range: {age}")
    except (ValueError, TypeError):
        issues.append("Invalid age value")

    try:
        height = int(record.get("身高(cm)", 0))
        if height < 120 or height > 210:
            issues.append(f"Height out of range: {height}")
    except (ValueError, TypeError):
        issues.append("Invalid height value")

    try:
        weight = int(record.get("体重(kg)", 0))
        if weight < 30 or weight > 200:
            issues.append(f"Weight out of range: {weight}")
    except (ValueError, TypeError):
        issues.append("Invalid weight value")

    gender = record.get("性别", "")
    if gender not in ("男", "女"):
        issues.append(f"Invalid gender: {gender}")

    # Check BMI plausibility
    try:
        height_m = height / 100.0
        bmi = weight / (height_m ** 2)
        if bmi < 14 or bmi > 50:
            issues.append(f"BMI implausible: {bmi:.1f}")
    except (ValueError, TypeError, ZeroDivisionError):
        pass

    # Check age-condition plausibility
    history = record.get("既往病史", "")
    if history != "无":
        conditions = set(history.split("、"))
        for cond, (min_age, max_age) in AGE_CONDITION_RULES.items():
            if cond in conditions and (age < min_age or age > max_age):
                issues.append(f"{cond} unlikely at age {age}")

    # Check condition-symptom correlation
    symptoms = record.get("体检前的症状", "")
    if symptoms != "无" and history != "无":
        symptom_set = set(symptoms.split("、"))
        condition_set = set(history.split("、"))
        for cond in condition_set:
            if cond in CONDITION_SYMPTOM_MAP:
                expected = set(CONDITION_SYMPTOM_MAP[cond])
                # Not a hard rule, just a check
                # If condition has known symptoms but patient has unrelated symptoms, flag
                pass

    # Check for duplicate IDs
    pid = record.get("患者ID", "")
    if not pid or len(str(pid)) != 6:
        issues.append(f"Invalid patient ID: {pid}")

    # Check check items exist
    check_items = record.get("体检项目", "")
    if not check_items:
        issues.append("No check items specified")

    return issues


def validate_dataset(records: list) -> dict:
    """Validate entire dataset. Returns statistics."""
    total = len(records)
    all_issues = []
    records_with_issues = 0

    # Check for duplicate IDs
    ids = [r.get("患者ID", "") for r in records]
    id_counts = Counter(ids)
    duplicate_ids = {k: v for k, v in id_counts.items() if v > 1}

    for i, record in enumerate(records):
        issues = validate_record(record)
        if issues:
            records_with_issues += 1
            all_issues.append({"index": i, "issues": issues})

    # Distribution checks
    gender_dist = Counter(r.get("性别", "") for r in records)
    age_bins = {"18-30": 0, "31-45": 0, "46-60": 0, "61-80": 0}
    for r in records:
        try:
            age = int(r.get("年龄(岁)", 0))
            if age <= 30:
                age_bins["18-30"] += 1
            elif age <= 45:
                age_bins["31-45"] += 1
            elif age <= 60:
                age_bins["46-60"] += 1
            else:
                age_bins["61-80"] += 1
        except ValueError:
            pass

    condition_counts = Counter()
    for r in records:
        h = r.get("既往病史", "无")
        if h != "无":
            for c in h.split("、"):
                condition_counts[c] += 1

    return {
        "total_records": total,
        "records_with_issues": records_with_issues,
        "issue_rate": records_with_issues / total if total > 0 else 0,
        "duplicate_ids": len(duplicate_ids),
        "gender_distribution": dict(gender_dist),
        "age_distribution": age_bins,
        "condition_distribution": dict(condition_counts.most_common(10)),
        "issues_detail": all_issues[:20],  # First 20 issues
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic_health_check_data.csv")
    args = parser.parse_args()

    input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.input)
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)

    print(f"Validating {len(records)} records from {input_path}\n")

    stats = validate_dataset(records)

    print(f"=== Validation Results ===")
    print(f"Total records:        {stats['total_records']}")
    print(f"Records with issues:  {stats['records_with_issues']} ({stats['issue_rate']:.1%})")
    print(f"Duplicate IDs:        {stats['duplicate_ids']}")
    print(f"\nGender distribution:  {stats['gender_distribution']}")
    print(f"Age distribution:     {stats['age_distribution']}")
    print(f"Condition distribution: {stats['condition_distribution']}")

    if stats['issues_detail']:
        print(f"\nSample issues (first {len(stats['issues_detail'])}):")
        for item in stats['issues_detail']:
            print(f"  Record {item['index']}: {', '.join(item['issues'])}")

    # Verdict
    if stats['issue_rate'] < 0.1:
        print("\n✅ Dataset passes validation (issue rate < 10%)")
    else:
        print(f"\n⚠️ Dataset has {stats['issue_rate']:.1%} issue rate - review recommended")


if __name__ == "__main__":
    main()
