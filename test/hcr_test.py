import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# print(project_root)

from agentos.agent.agent import Agent
from agentos.memory import TemporaryMemory,Message,Role
from src.tools import *
from src.prompt import HCR_PROMPT,OUTPUT_PROMPT
from agentos.utils import call_model

from dotenv import load_dotenv
env_path = os.path.join(project_root, ".env")
if not os.path.exists(env_path):
    env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)
api_key = os.environ.get("DEEPSEEK_API_KEY")

myagent = Agent(
    name="myagent",
    model={},
    tools=[
        search_by_id(),
        search_by_other(),
        recommend_by_age(),
        recommend_by_gender()
    ],
    api_key=api_key
) 

user_info ="id:426815\ngender:男\nage:50\nheight:172cm\nweight:80kg\nmedical_history:高血压\nsymptom:头晕"

myagent.run(HCR_PROMPT.format(user_info))

myagent.memory.add_memory(Message(Role.SYSTEM,OUTPUT_PROMPT))

print("\n==============================HCR===============================")
response=call_model(myagent.memory.memory,api_key=api_key)
print(response)
myagent.memory.add_memory(Message(Role.ASSISTANT,response))

print("\n=============================MRMORY=============================")
for i in myagent.memory.memory:
    print(f"【{i['role']}】")
    print(i['content'])
    print("------------")

