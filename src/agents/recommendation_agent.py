import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.agents.base_agent import MedicalAgent
from src.agents.message import AgentMessage
from src.agents.context import AgentContext
from src.agents.citation import CitationTracker

RECOMMENDATION_PROMPT = """你是一个专业的体检规划师。请根据患者的症状分析和风险评估结果，生成个性化的体检推荐。

输出格式要求：
1. **推荐项目**：按优先级列出4-6个检查项目，每个项目附带简要理由
2. **推荐理由**：结合患者具体情况说明
3. **注意事项**：检查前准备事项
4. **依据引用**：每个推荐项必须标注其依据来源（相似病例/临床指南/风险评估）

请务必：
- 每个推荐项引用具体的证据来源
- 按优先级排序（最紧急/最重要的在前）
- 避免推荐重复或不必要的检查
- 考虑患者的年龄、性别适配性"""


class RecommendationAgent(MedicalAgent):
    """Agent specialized in generating prioritized health check recommendations.

    Synthesizes symptom analysis, risk assessment, and retrieval results
    to produce evidence-cited recommendations.
    """

    def __init__(self, api_key=None):
        super().__init__(
            name="RecommendationAgent",
            system_prompt=RECOMMENDATION_PROMPT,
            api_key=api_key
        )

    def run(self, context: AgentContext) -> AgentMessage:
        """Generate health check recommendations.

        Args:
            context: AgentContext with all prior agent outputs.

        Returns:
            AgentMessage with recommendations and citations.
        """
        user_info = context.user_info
        symptoms = user_info.get("symptoms", "无")
        history = user_info.get("medical_history", "无")
        age = user_info.get("age", "未知")
        gender = user_info.get("gender", "未知")
        gender_cn = "男" if gender == "male" else ("女" if gender == "female" else gender)

        # Build evidence section
        evidence_text = ""
        for facet, results in context.retrieval_results.items():
            if results:
                evidence_text += f"\n[{facet}]:\n"
                for r in results[:3]:
                    evidence_text += f"  - {r}\n"

        # Include prior agent outputs
        symptom_section = ""
        if context.symptom_assessment:
            symptom_section = f"\n症状分析：\n{context.symptom_assessment}\n"

        risk_section = ""
        if context.risk_assessment:
            risk_section = f"\n风险评估：\n{context.risk_assessment}\n"

        # Include safety critique if revision round
        critique_section = ""
        if context.safety_report:
            critique_section = f"\n安全审核意见（请据此修正推荐）：\n{context.safety_report}\n"

        user_message = f"""请为以下患者生成体检推荐：

患者信息：
- 性别：{gender_cn}
- 年龄：{age}岁
- 身高：{user_info.get('height', '未知')}cm
- 体重：{user_info.get('weight', '未知')}kg
- 既往病史：{history}
- 当前症状：{symptoms}
{symptom_section}{risk_section}
相似病例/指南数据：{evidence_text if evidence_text else "无"}
{critique_section}
请生成个性化的体检推荐，每个项目标注依据来源。"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self._call_llm(messages)

        # Build citations
        tracker = CitationTracker()
        for facet, results in context.retrieval_results.items():
            for r in results[:2]:
                tracker.add_citation(
                    recommendation="体检推荐",
                    source=r[:80],
                    confidence=0.7,
                    query_facet=facet
                )

        # Collect all evidence
        evidence = context.get_all_evidence()

        msg = AgentMessage(
            sender=self.name,
            receiver="Coordinator",
            content=response,
            evidence=evidence[:5],
            confidence=0.8,
            message_type="response"
        )
        context.add_message(msg)
        context.recommendations = response

        return msg
