import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.agents.base_agent import MedicalAgent
from src.agents.message import AgentMessage
from src.agents.context import AgentContext

RISK_ASSESSOR_PROMPT = """你是一个专业的健康风险评估助手。你的任务是根据患者的个人资料和相似病例数据，评估其健康风险。

请按以下格式输出评估结果：

1. **风险等级**：[低风险/中风险/高风险]
2. **主要风险因子**：列出主要的风险因素（年龄、病史、症状等）
3. **相似病例参考**：引用检索到的相似病例数据
4. **重点关注系统**：心血管系统/代谢系统/消化系统等

请基于数据进行评估，注意：
- 引用具体的相似病例数据作为依据
- 综合考虑年龄、性别、病史、症状的交互影响
- 标注评估的置信度"""


class RiskAssessor(MedicalAgent):
    """Agent specialized in health risk assessment.

    Evaluates health risks based on patient profile and similar cases
    retrieved from the database.
    """

    def __init__(self, api_key=None):
        super().__init__(
            name="RiskAssessor",
            system_prompt=RISK_ASSESSOR_PROMPT,
            api_key=api_key
        )

    def run(self, context: AgentContext) -> AgentMessage:
        """Assess health risk based on user profile and retrieval results.

        Args:
            context: AgentContext with user_info, retrieval_results, and symptom_assessment.

        Returns:
            AgentMessage with risk assessment.
        """
        user_info = context.user_info
        symptoms = user_info.get("symptoms", "无")
        history = user_info.get("medical_history", "无")
        age = user_info.get("age", "未知")
        gender = user_info.get("gender", "未知")
        gender_cn = "男" if gender == "male" else ("女" if gender == "female" else gender)

        # Gather similar case data from retrieval
        similar_cases = ""
        for facet in ["full_profile", "history", "demographic", "risk_combination"]:
            if facet in context.retrieval_results:
                results = context.retrieval_results[facet]
                if results:
                    similar_cases += f"\n[{facet}类检索结果]:\n"
                    for r in results[:3]:
                        similar_cases += f"  - {r}\n"

        # Include symptom analysis if available
        symptom_section = ""
        if context.symptom_assessment:
            symptom_section = f"\n症状分析结果:\n{context.symptom_assessment}\n"

        user_message = f"""请评估以下患者的健康风险：

患者信息：
- 性别：{gender_cn}
- 年龄：{age}岁
- 身高：{user_info.get('height', '未知')}cm
- 体重：{user_info.get('weight', '未知')}kg
- 既往病史：{history}
- 当前症状：{symptoms}
{symptom_section}
相似病例检索数据：{similar_cases if similar_cases else "无相似病例数据"}

请进行健康风险评估。"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self._call_llm(messages)

        # Collect evidence
        evidence = []
        for facet in context.retrieval_results:
            evidence.extend(context.retrieval_results[facet][:2])

        msg = AgentMessage(
            sender=self.name,
            receiver="Coordinator",
            content=response,
            evidence=evidence[:5],
            confidence=0.75,
            message_type="response"
        )
        context.add_message(msg)
        context.risk_assessment = response

        return msg
