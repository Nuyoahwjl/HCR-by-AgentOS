import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.agents.base_agent import MedicalAgent
from src.agents.context import AgentContext
from src.agents.symptom_analyzer import SymptomAnalyzer
from src.agents.risk_assessor import RiskAssessor
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.safety_checker import SafetyChecker
from src.agents.message import AgentMessage
from src.agents.citation import CitationTracker
from src.ontology import MedicalOntology
from src.query_decomposer import QueryDecomposer

from typing import List, Dict, Any, Optional


COORDINATOR_SYNTHESIS_PROMPT = """你是一个专业的体检规划协调员。你已经收集了以下信息：

1. 症状分析结果
2. 风险评估结果
3. 体检推荐方案
4. 安全审核报告

请综合以上所有信息，输出最终的体检推荐报告。格式要求：
- 使用Markdown格式
- 包含推荐项目列表（带优先级）
- 每个推荐项附带依据来源
- 包含注意事项
- 排版整洁美观，可使用emoji"""


class CoordinatorAgent:
    """Orchestrator that coordinates the multi-agent pipeline.

    Pipeline flow:
    1. Query decomposition + hybrid retrieval (evidence gathering)
    2. SymptomAnalyzer (symptom assessment)
    3. RiskAssessor (risk evaluation)
    4. RecommendationAgent (recommendation generation)
    5. SafetyChecker (safety validation)
    6. If critique -> back to step 4 (max 2 rounds)
    7. Synthesize final output

    This mirrors clinical workflows: triage -> risk stratification ->
    recommendation -> peer review.
    """

    def __init__(self, api_key: Optional[str] = None, max_revision_rounds: int = 2):
        self.api_key = api_key
        self.max_revision_rounds = max_revision_rounds

        # Initialize specialized agents
        self.symptom_analyzer = SymptomAnalyzer(api_key=api_key)
        self.risk_assessor = RiskAssessor(api_key=api_key)
        self.recommendation_agent = RecommendationAgent(api_key=api_key)
        self.safety_checker = SafetyChecker(api_key=api_key)

        # Initialize query infrastructure
        self.ontology = MedicalOntology()
        self.decomposer = QueryDecomposer(ontology=self.ontology)

    def _gather_evidence(
        self,
        user_info: Dict[str, Any],
        retriever,
        num_results: int = 5
    ) -> Dict[str, List[str]]:
        """Decompose queries and gather evidence via hybrid retrieval.

        Args:
            user_info: User profile dict.
            retriever: HybridRetriever instance.
            num_results: Max results per sub-query.

        Returns:
            Dict mapping facet -> list of result strings.
        """
        from agentos.rag.data import BaseData

        sub_queries = self.decomposer.decompose(user_info)
        all_results = {}
        seen = set()

        for sq in sub_queries:
            facet = sq["facet"]
            query = sq["query"]

            try:
                results = retriever.retrieve(query=query, top_k=3)
                result_strings = []
                for r in results:
                    content = r.get_content() if isinstance(r, BaseData) else str(r)
                    if content not in seen:
                        seen.add(content)
                        result_strings.append(content)
                all_results[facet] = result_strings
            except Exception:
                continue

        return all_results

    def _synthesize_output(self, context: AgentContext) -> str:
        """Use LLM to synthesize all agent outputs into final report.

        Args:
            context: AgentContext with all agent outputs.

        Returns:
            Synthesized final output string.
        """
        sections = []
        if context.symptom_assessment:
            sections.append(f"【症状分析】\n{context.symptom_assessment}")
        if context.risk_assessment:
            sections.append(f"【风险评估】\n{context.risk_assessment}")
        if context.recommendations:
            sections.append(f"【推荐方案】\n{context.recommendations}")
        if context.safety_report:
            sections.append(f"【安全审核】\n{context.safety_report}")

        combined = "\n\n".join(sections)

        messages = [
            {"role": "system", "content": COORDINATOR_SYNTHESIS_PROMPT},
            {"role": "user", "content": f"以下是各专业模块的分析结果，请综合输出最终报告：\n\n{combined}"}
        ]

        try:
            from agentos.utils import call_model
            return call_model(messages, self.api_key)
        except Exception:
            # Fallback: just concatenate outputs
            return combined

    def run(
        self,
        user_info: Dict[str, Any],
        retriever=None
    ) -> str:
        """Execute the full multi-agent pipeline.

        Args:
            user_info: Dict with keys: id, gender, age, height, weight,
                       medical_history, symptoms.
            retriever: HybridRetriever instance for evidence gathering.
                       If None, skips retrieval.

        Returns:
            Final synthesized recommendation string.
        """
        context = AgentContext()
        context.set_user_info(user_info)

        # Phase 1: Evidence gathering via query decomposition + hybrid retrieval
        if retriever:
            retrieval_results = self._gather_evidence(user_info, retriever)
            for facet, results in retrieval_results.items():
                context.set_retrieval_results(facet, results)

        # Phase 2: Symptom analysis
        try:
            self.symptom_analyzer.run(context)
        except Exception as e:
            context.symptom_assessment = f"症状分析失败: {str(e)}"

        # Phase 3: Risk assessment
        try:
            self.risk_assessor.run(context)
        except Exception as e:
            context.risk_assessment = f"风险评估失败: {str(e)}"

        # Phase 4 + 5: Recommendation + Safety loop
        for round_num in range(self.max_revision_rounds + 1):
            # Generate recommendations
            try:
                self.recommendation_agent.run(context)
            except Exception as e:
                context.recommendations = f"推荐生成失败: {str(e)}"
                break

            # Safety check
            try:
                safety_msg = self.safety_checker.run(context)
            except Exception as e:
                # Safety check failed, accept current recommendations
                break

            # If no critical issues, we're done
            if safety_msg.message_type != "critique":
                break

            # If this was the last round, accept regardless
            if round_num >= self.max_revision_rounds:
                break

            # Otherwise, loop back for revision
            context.safety_report = safety_msg.content

        # Phase 6: Synthesize final output
        final_output = self._synthesize_output(context)
        context.final_output = final_output

        return final_output
