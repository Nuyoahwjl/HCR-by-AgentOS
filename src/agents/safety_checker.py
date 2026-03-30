import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.agents.base_agent import MedicalAgent
from src.agents.message import AgentMessage
from src.agents.context import AgentContext
from typing import List

SAFETY_CHECKER_PROMPT = """你是一个医学安全审核助手。你的任务是审核体检推荐是否安全、合理。

请检查以下方面：
1. **禁忌症检测**：推荐项是否存在与患者病史/症状冲突的禁忌症
2. **年龄适配性**：检查项目是否适合患者的年龄段
3. **冗余检测**：是否存在重复或功能重叠的检查项
4. **关键遗漏**：对于患者的风险状况，是否遗漏了关键检查项
5. **剂量/频率合理性**：检查频率是否过高或过低

输出格式：
- **审核结果**：[通过/需要修改]
- **发现的问题**：列出具体问题（如有）
- **修改建议**：具体的修改意见
- **严重程度**：[critical/warning/info] 每个问题标注严重程度

注意：只有 critical 级别的问题才需要打回修改，warning 和 info 级别仅作为提醒。"""


# Safety rules for common contraindications
SAFETY_RULES = [
    {
        "condition": lambda info: int(info.get("age", 0)) < 40 and "肠镜" in info.get("_recommendations", ""),
        "severity": "warning",
        "message": "40岁以下无家族史不建议常规肠镜检查"
    },
    {
        "condition": lambda info: "肾" in info.get("medical_history", "") and "造影" in info.get("_recommendations", ""),
        "severity": "critical",
        "message": "肾病患者应避免造影剂相关检查"
    },
    {
        "condition": lambda info: int(info.get("age", 0)) < 18 and "肿瘤标志物" in info.get("_recommendations", ""),
        "severity": "warning",
        "message": "18岁以下不建议常规肿瘤标志物筛查"
    },
]


class SafetyChecker(MedicalAgent):
    """Agent specialized in validating recommendations for safety.

    Checks for contraindications, age-inappropriate tests, redundancy,
    and critical omissions. Outputs structured critique.
    """

    def __init__(self, api_key=None):
        super().__init__(
            name="SafetyChecker",
            system_prompt=SAFETY_CHECKER_PROMPT,
            api_key=api_key
        )

    def _check_rules(self, context: AgentContext) -> List[dict]:
        """Apply rule-based safety checks.

        Args:
            context: AgentContext with recommendations.

        Returns:
            List of rule violations with severity and message.
        """
        violations = []
        if not context.recommendations:
            return violations

        info = dict(context.user_info)
        info["_recommendations"] = context.recommendations

        for rule in SAFETY_RULES:
            try:
                if rule["condition"](info):
                    violations.append({
                        "severity": rule["severity"],
                        "message": rule["message"]
                    })
            except (ValueError, TypeError, KeyError):
                continue

        return violations

    def run(self, context: AgentContext) -> AgentMessage:
        """Validate recommendations for safety issues.

        Args:
            context: AgentContext with recommendations set.

        Returns:
            AgentMessage with safety validation results.
            If critical issues found, message_type="critique".
        """
        # Rule-based checks first
        rule_violations = self._check_rules(context)
        rule_text = ""
        if rule_violations:
            rule_text = "\n规则检测结果：\n"
            for v in rule_violations:
                rule_text += f"  [{v['severity'].upper()}] {v['message']}\n"

        user_info = context.user_info
        symptoms = user_info.get("symptoms", "无")
        history = user_info.get("medical_history", "无")
        age = user_info.get("age", "未知")
        gender = user_info.get("gender", "未知")
        gender_cn = "男" if gender == "male" else ("女" if gender == "female" else gender)

        recommendations = context.recommendations or "无推荐内容"

        user_message = f"""请审核以下体检推荐的安全性：

患者信息：
- 性别：{gender_cn}
- 年龄：{age}岁
- 既往病史：{history}
- 当前症状：{symptoms}

当前推荐：
{recommendations}
{rule_text}
请进行安全审核，标注是否需要修改。"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self._call_llm(messages)

        # Determine if this is a critique (has critical issues)
        has_critical = any(v["severity"] == "critical" for v in rule_violations)
        is_critique = has_critical or "需要修改" in response

        msg = AgentMessage(
            sender=self.name,
            receiver="RecommendationAgent" if is_critique else "Coordinator",
            content=response,
            evidence=[v["message"] for v in rule_violations],
            confidence=0.9,
            message_type="critique" if is_critique else "response"
        )
        context.add_message(msg)
        context.safety_report = response

        return msg
