import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.agents.base_agent import MedicalAgent
from src.agents.message import AgentMessage
from src.agents.context import AgentContext
from typing import List

SYMPTOM_ANALYZER_PROMPT = """你是一个专业的症状分析助手。你的任务是根据患者的症状和病史，进行结构化分析。

请按以下格式输出分析结果：

1. **症状分类**：将症状分为 [急性/慢性] [严重程度: 轻度/中度/重度]
2. **关联疾病**：列出症状可能关联的疾病/系统
3. **风险指标**：识别需要重点关注的风险信号
4. **建议检查方向**：基于症状分析推荐的检查类别

请基于医学常识进行分析，注意：
- 仅做症状分析，不做诊断
- 标注你的分析依据
- 对不确定的部分要说明"""


class SymptomAnalyzer(MedicalAgent):
    """Agent specialized in parsing and categorizing symptoms.

    Takes user symptoms and medical history, produces a structured
    symptom assessment with risk indicators.
    """

    def __init__(self, api_key=None):
        super().__init__(
            name="SymptomAnalyzer",
            system_prompt=SYMPTOM_ANALYZER_PROMPT,
            api_key=api_key
        )

    def run(self, context: AgentContext) -> AgentMessage:
        """Analyze symptoms from the user profile.

        Args:
            context: AgentContext with user_info set.

        Returns:
            AgentMessage with structured symptom assessment.
        """
        user_info = context.user_info
        symptoms = user_info.get("symptoms", "无")
        history = user_info.get("medical_history", "无")
        age = user_info.get("age", "未知")
        gender = user_info.get("gender", "未知")
        gender_cn = "男" if gender == "male" else ("女" if gender == "female" else gender)

        user_message = f"""请分析以下患者的症状：

患者信息：
- 性别：{gender_cn}
- 年龄：{age}岁
- 既往病史：{history}
- 当前症状：{symptoms}

请进行结构化症状分析。"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self._call_llm(messages)

        # Collect evidence from retrieval results
        evidence = []
        if "symptom" in context.retrieval_results:
            evidence.extend(context.retrieval_results["symptom"][:2])
        if "symptom_expanded" in context.retrieval_results:
            evidence.extend(context.retrieval_results["symptom_expanded"][:1])

        msg = AgentMessage(
            sender=self.name,
            receiver="Coordinator",
            content=response,
            evidence=evidence,
            confidence=0.8,
            message_type="response"
        )
        context.add_message(msg)
        context.symptom_assessment = response

        return msg
