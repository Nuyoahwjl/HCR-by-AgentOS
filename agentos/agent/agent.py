import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import inspect
import re

from agentos.memory import TemporaryMemory,Message,Role
from agentos.prompt import DEFAULT_PROMPT

from typing import List

from agentos.utils import call_model


def parse_tool_info(tools):
    info = ""
    for index, tool in enumerate(tools):
        info = info + "\nfunction " + str(index+1) + ".\n"
        info = info + inspect.cleandoc(tool.run.__doc__)
    return info


class Agent:
    def __init__(
        self,
        name: str = None,
        model: dict = None,
        tools: List = None,
        api_key: str | None = None,
    ):
        self.name = name
        self.model = model
        self.api_key = api_key

        self.tools = {}
        for tool in tools:
            self.tools[tool.__class__.__name__] = tool

        self.memory = TemporaryMemory()

        tool_info = ""
        if tools is not None:
            tool_info = parse_tool_info(tools)

        self.memory.add_memory(Message(Role.SYSTEM, DEFAULT_PROMPT.format(tool_info)))

    def call_tool(self, tool_name: str, tool_args: List):
        return str(self.tools[tool_name].run(*tool_args))

    def _extract_field(self, text: str, field: str) -> str | None:
        """Extract a field value from text using flexible pattern matching.

        Handles: 'field: value', 'field：value', '**field:** value', 'Field: value'
        """
        pattern = rf'(?:\*\*)?{re.escape(field)}(?:\*\*)?\s*[:：]\s*(.+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('*').strip()
        return None

    def reason(self):
        response = call_model(self.memory.memory, self.api_key)

        self.memory.add_memory(Message(Role.ASSISTANT, response))

        thought = ""
        tool_name = ""
        tool_args = []

        lines = response.splitlines()
        clean_lines = [l.strip() for l in lines if l.strip()]

        if not clean_lines:
            print("ERROR: Empty response from model")
            return "", "finish", []

        # Method 1: Try structured parsing (thought: ..., function: ..., argument: ...)
        for line in clean_lines:
            if not thought:
                val = self._extract_field(line, "thought")
                if val:
                    thought = val
                    continue
            if not tool_name:
                val = self._extract_field(line, "function")
                if val:
                    tool_name = val
                    continue
            # Arguments
            arg_val = self._extract_field(line, "argument\d*")
            if arg_val is None:
                arg_val = self._extract_field(line, "arg\d*")
            if arg_val is None:
                for sep in [':', '：']:
                    if sep in line:
                        parts = line.split(sep, 1)
                        if len(parts) == 2:
                            key = parts[0].strip().strip('*').strip()
                            val = parts[1].strip()
                            if key.lower() not in ('thought', 'function', 'finish') and val:
                                arg_val = val
                                break
            if arg_val:
                tool_args.append(arg_val)

        # Method 2: Fallback - try line-by-line colon split
        if not tool_name:
            for line in clean_lines:
                for sep in [':', '：']:
                    if sep in line:
                        parts = line.split(sep, 1)
                        key = parts[0].strip().strip('*').lower()
                        val = parts[1].strip() if len(parts) > 1 else ""
                        if key in ('function', 'func', 'tool') and val:
                            tool_name = val
                            break
                if tool_name:
                    break

        # Method 3: Last resort - check if response contains a known tool name
        if not tool_name:
            known_tools = set(self.tools.keys()) | {'finish'}
            for t in known_tools:
                if t in response:
                    tool_name = t
                    break

        if not tool_name:
            print(f"ERROR: Could not parse tool from response:\n{response[:300]}")
            tool_name = "finish"

        print(response)
        return thought, tool_name, tool_args

    def act(self, tool_name: str, tool_args: List):
        print(f"call tool:{tool_name}\nargs:{tool_args}")

        tool_call_res = self.call_tool(tool_name, tool_args)
        self.memory.add_memory(
            Message(Role.USER, "The " + tool_name + " function has been executed and the result is below:\n" + tool_call_res)
        )

        print(f"tool_call_res:\n{tool_call_res}")
        return True

    def run(self, task: str):
        self.memory.add_memory(Message(Role.USER, task))
        max_steps = 10

        for step in range(max_steps):
            print("-----------------------------reason-----------------------------")
            thought, tool_name, tool_args = self.reason()
            if tool_name == 'finish':
                print("Task finished.")
                break

            print("------------------------------act-------------------------------")
            self.act(tool_name, tool_args)

        if step >= max_steps - 1:
            print("WARNING: Max steps reached, forcing finish.")
