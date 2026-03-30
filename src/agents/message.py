from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class AgentMessage:
    """Structured message for inter-agent communication.

    Attributes:
        sender: Name of the sending agent.
        receiver: Name of the target agent.
        content: The message payload (assessment, recommendation, critique, etc.).
        evidence: List of evidence source strings supporting the content.
        confidence: Confidence score (0-1) for the content.
        message_type: Type of message - "request", "response", "critique", "synthesis".
        timestamp: When the message was created.
        metadata: Additional key-value metadata.
    """
    sender: str
    receiver: str
    content: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    message_type: str = "response"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "message_type": self.message_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def format_evidence(self) -> str:
        """Format evidence list into a readable string."""
        if not self.evidence:
            return "无证据来源"
        return "\n".join(f"- {e}" for e in self.evidence)
