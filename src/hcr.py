import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


from agentos.agent.agent import Agent
from agentos.memory import TemporaryMemory,Message,Role
from src.tools import *
from src.hcr_prompt import HCR_PROMPT,OUTPUT_PROMPT
from agentos.utils import call_model


class Recommendation:
    def __init__(self):
        self.mediagent = Agent(
            name="mediagent",
            model={},
            tools=[
                search_by_id(),
                search_by_other()
                ]
        )

    def run(self,user_info):
        self.mediagent.run(HCR_PROMPT.format(user_info))
        self.mediagent.memory.add_memory(Message(Role.SYSTEM,OUTPUT_PROMPT))
        response=call_model(self.mediagent.memory.memory)
        self.mediagent.memory.add_memory(Message(Role.ASSISTANT,response))
        return response



re = Recommendation()
mes=re.run("id:426815\ngender:男\nage:50\nheight:172cm\nweight:80kg\nmedical_history:高血压\nsymptom:头晕")

print("\n\n\n\n\n")
print("==============================MRMORY==============================")
for i in re.mediagent.memory.memory:
    print(f"【{i['role']}】")
    print(i['content'])
    print("------------")

print("\n\n\n\n\n")
print("=============================RESPONSE=============================")
print(mes)
