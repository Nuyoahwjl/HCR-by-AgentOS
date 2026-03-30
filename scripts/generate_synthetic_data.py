"""Generate synthetic health check patient profiles using DeepSeek LLM.

Usage:
    python scripts/generate_synthetic_data.py [--num 200] [--output data/synthetic_health_check_data.csv]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import csv
from agentos.utils import call_model


GENERATION_PROMPT = """你是一个医学数据生成专家。请生成{batch_size}个中国患者的虚拟体检档案，格式为JSON数组。

每个档案包含以下字段：
- 患者ID：6位数字
- 性别："男"或"女"
- 年龄(岁)：18-80之间的整数
- 身高(cm)：150-190之间的整数
- 体重(kg)：45-120之间的整数
- 既往病史：常见慢性病或"无"（高血压、糖尿病、高血脂、冠心病、痛风、甲状腺疾病、无）
- 体检前的症状：常见症状或"无"（头晕、头痛、胸闷、乏力、失眠、口渴多尿、视力模糊、关节痛、无）
- 体检项目：符合该患者情况的合理体检项目组合（如：血常规、尿常规、心电图、血脂检查等）

要求：
1. 数据分布要真实：约30%健康人群，40%有1种慢性病，20%有2种，10%有3种以上
2. 症状要与病史和年龄相关联
3. 体检项目要符合临床指南
4. 年龄分布：18-30岁(20%), 31-45岁(30%), 46-60岁(30%), 61-80岁(20%)
5. 性别比例大致1:1

请直接输出JSON数组，不要包含其他文字。"""

CONDITIONS = ["高血压", "糖尿病", "高血脂", "冠心病", "痛风", "甲状腺疾病"]
SYMPTOMS = ["头晕", "头痛", "胸闷", "乏力", "失眠", "口渴多尿", "视力模糊", "关节痛"]
CHECK_ITEMS = [
    "血常规", "尿常规", "肝功能", "肾功能", "心电图",
    "血脂检查", "血糖检测", "血压监测", "胸片", "腹部B超",
    "甲状腺功能", "肿瘤标志物", "骨密度", "眼科检查",
    "颈椎腰椎检查", "心脏彩超", "颈动脉超声", "胃肠镜",
    "妇科检查", "乳腺检查", "前列腺检查"
]


def generate_batch(batch_size: int, api_key: str = None) -> list:
    """Generate a batch of synthetic patient profiles via LLM."""
    prompt = GENERATION_PROMPT.format(batch_size=batch_size)
    messages = [
        {"role": "system", "content": "你是一个医学数据生成助手，请严格按照要求输出JSON格式数据。"},
        {"role": "user", "content": prompt}
    ]

    response = call_model(messages, api_key=api_key)

    # Parse JSON from response
    try:
        # Try to find JSON array in the response
        start = response.find('[')
        end = response.rfind(']') + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            return data
    except json.JSONDecodeError:
        pass

    # Try parsing the whole response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print(f"Failed to parse LLM response. First 200 chars: {response[:200]}")
        return []


def generate_rule_based(num: int) -> list:
    """Generate synthetic profiles using rule-based approach (no API needed).

    Fallback when LLM API is unavailable.
    """
    import random
    random.seed(42)

    records = []
    for i in range(num):
        age = random.choices(
            [random.randint(18, 30), random.randint(31, 45),
             random.randint(46, 60), random.randint(61, 80)],
            weights=[0.2, 0.3, 0.3, 0.2]
        )[0]

        gender = random.choice(["男", "女"])
        height = random.randint(155, 185) if gender == "男" else random.randint(150, 175)
        weight = random.randint(55, 100) if gender == "男" else random.randint(45, 85)

        # Assign conditions based on age
        num_conditions = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
        if num_conditions == 0:
            history = "无"
        else:
            # Higher age -> higher chance of conditions
            available = CONDITIONS.copy()
            if age > 50 and "高血压" in available:
                available = ["高血压"] + available  # boost hypertension for older
            selected = random.sample(CONDITIONS, min(num_conditions, len(CONDITIONS)))
            history = "、".join(selected)

        # Symptoms (correlated with conditions)
        if history == "无":
            symptom_chance = 0.3
        else:
            symptom_chance = 0.7
        if random.random() < symptom_chance:
            num_symptoms = random.randint(1, 3)
            symptoms = "、".join(random.sample(SYMPTOMS, num_symptoms))
        else:
            symptoms = "无"

        # Check items (based on age, gender, conditions)
        checks = set(random.sample(CHECK_ITEMS[:6], random.randint(3, 5)))
        if age > 40:
            checks.update(random.sample(CHECK_ITEMS[6:10], 2))
        if age > 60:
            checks.add("肿瘤标志物")
            checks.add("骨密度")
        if "高血压" in history:
            checks.add("血压监测")
            checks.add("心电图")
        if "糖尿病" in history:
            checks.add("血糖检测")
        if gender == "女":
            checks.add("妇科检查")
            checks.add("乳腺检查")
        if gender == "男" and age > 45:
            checks.add("前列腺检查")

        patient_id = f"{random.randint(100000, 999999)}"
        records.append({
            "患者ID": patient_id,
            "性别": gender,
            "年龄(岁)": age,
            "身高(cm)": height,
            "体重(kg)": weight,
            "既往病史": history,
            "体检前的症状": symptoms,
            "体检项目": "、".join(sorted(checks))
        })

    return records


def save_csv(records: list, output_path: str):
    """Save records to CSV file."""
    if not records:
        print("No records to save.")
        return

    fieldnames = ["患者ID", "性别", "年龄(岁)", "身高(cm)", "体重(kg)",
                  "既往病史", "体检前的症状", "体检项目"]

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic health check data")
    parser.add_argument("--num", type=int, default=200, help="Number of records to generate")
    parser.add_argument("--output", type=str, default="data/synthetic_health_check_data.csv")
    parser.add_argument("--method", choices=["llm", "rule"], default="rule",
                        help="Generation method: 'llm' uses API, 'rule' is offline")
    args = parser.parse_args()

    if args.method == "llm":
        print(f"Generating {args.num} records via LLM...")
        batch_size = 20
        all_records = []
        for i in range(0, args.num, batch_size):
            current_batch = min(batch_size, args.num - len(all_records))
            print(f"  Batch {i // batch_size + 1}: generating {current_batch} records...")
            records = generate_batch(current_batch)
            all_records.extend(records)
            if len(all_records) >= args.num:
                break
        records = all_records[:args.num]
    else:
        print(f"Generating {args.num} records via rule-based method...")
        records = generate_rule_based(args.num)

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_csv(records, output_path)


if __name__ == "__main__":
    main()
