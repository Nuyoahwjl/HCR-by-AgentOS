import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agentos.utils import call_model
from typing import List, Dict, Any, Optional


class MedicalAgent:
    """Base class for specialized medical agents.

    Each agent has a name, system prompt, and can process context
    to produce an AgentMessage output. Agents communicate through
    the shared AgentContext.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        api_key: Optional[str] = None
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.api_key = api_key

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM with a list of messages.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            The LLM response string.
        """
        return call_model(messages, self.api_key)

    def run(self, context) -> Any:
        """Process the agent context and produce output.

        Must be implemented by subclasses.

        Args:
            context: AgentContext with user info and previous agent outputs.

        Returns:
            An AgentMessage or other output.
        """
        raise NotImplementedError("Subclasses must implement run()")
