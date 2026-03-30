from typing import List, Dict, Any, Optional
from src.agents.message import AgentMessage


class AgentContext:
    """Shared context for multi-agent communication.

    Stores the user profile, agent messages, retrieval results,
    and final synthesized output. Acts as a shared memory for the
    multi-agent pipeline.
    """

    def __init__(self):
        self.user_info: Dict[str, Any] = {}
        self.messages: List[AgentMessage] = []
        self.retrieval_results: Dict[str, List[str]] = {}
        self.symptom_assessment: Optional[str] = None
        self.risk_assessment: Optional[str] = None
        self.recommendations: Optional[str] = None
        self.safety_report: Optional[str] = None
        self.final_output: Optional[str] = None
        self.iteration: int = 0

    def set_user_info(self, user_info: Dict[str, Any]):
        """Set the user profile information."""
        self.user_info = user_info

    def add_message(self, message: AgentMessage):
        """Add an inter-agent message to the context."""
        self.messages.append(message)

    def get_messages_from(self, agent_name: str) -> List[AgentMessage]:
        """Get all messages sent by a specific agent."""
        return [m for m in self.messages if m.sender == agent_name]

    def get_messages_to(self, agent_name: str) -> List[AgentMessage]:
        """Get all messages addressed to a specific agent."""
        return [m for m in self.messages if m.receiver == agent_name]

    def get_latest_from(self, agent_name: str) -> Optional[AgentMessage]:
        """Get the latest message from a specific agent."""
        msgs = self.get_messages_from(agent_name)
        return msgs[-1] if msgs else None

    def set_retrieval_results(self, facet: str, results: List[str]):
        """Store retrieval results for a specific query facet."""
        self.retrieval_results[facet] = results

    def get_all_evidence(self) -> List[str]:
        """Collect all evidence from all messages."""
        evidence = []
        for msg in self.messages:
            evidence.extend(msg.evidence)
        return list(set(evidence))  # deduplicate

    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get the current state of the agent pipeline."""
        return {
            "user_info": self.user_info,
            "symptom_assessment": self.symptom_assessment,
            "risk_assessment": self.risk_assessment,
            "recommendations": self.recommendations,
            "safety_report": self.safety_report,
            "iteration": self.iteration,
            "message_count": len(self.messages)
        }

    def reset_for_revision(self):
        """Reset for another revision cycle (preserves user info and messages)."""
        self.recommendations = None
        self.safety_report = None
        self.iteration += 1
