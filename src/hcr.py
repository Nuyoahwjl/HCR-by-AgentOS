import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(current_dir+"/.env")
api_key = os.environ.get("TOGETHER_API_KEY")



from agentos.agent.agent import Agent
from agentos.memory import TemporaryMemory,Message,Role
from src.tools import *
from src.hcr_prompt import HCR_PROMPT,OUTPUT_PROMPT
from agentos.utils import call_model


class Recommendation:
    def __init__(self,api_key: str | None = None):
        self.mediagent = Agent(
            name="mediagent",
            model={},
            tools=[
                search_by_id(),
                search_by_other(),
                recommend_by_age(),
                recommend_by_gender(),
                calculate_bmi()  # 添加新工具
                ],
            api_key=api_key
        )

    def run(self,user_info):
        self.mediagent.run(HCR_PROMPT.format(user_info))
        self.mediagent.memory.add_memory(Message(Role.SYSTEM,OUTPUT_PROMPT))
        response=call_model(self.mediagent.memory.memory,self.mediagent.api_key)
        self.mediagent.memory.add_memory(Message(Role.ASSISTANT,response))

        print("\n\n\n\n\n")
        print("==============================MRMORY==============================")
        for i in self.mediagent.memory.memory:
            print(f"【{i['role']}】")
            print(i['content'])
            print("------------")

        print("\n\n\n\n\n")
        print("=============================RESPONSE=============================")
        print(response)
        
        return response




# re = Recommendation(api_key=api_key)
# info={"id":"426815","gender":"男","age":50,"height":"172cm","weight":"80kg","medical_history":"高血压","symptom":"头晕"}
# re.run(info)

