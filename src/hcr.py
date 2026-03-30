import sys
import os

# 获取当前文件的目录和项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(current_dir + "/.env")
api_key = os.environ.get("DEEPSEEK_API_KEY")

# 导入所需模块
from src.prompt import HCR_PROMPT, OUTPUT_PROMPT
from agentos.utils import call_model
import sqlite3


class Recommendation:
    """Health Check Recommendation system.

    Supports two modes:
    - legacy: Single ReAct agent with tools (original behavior)
    - multi_agent: Multi-agent pipeline with CoordinatorAgent (new)

    Set use_multi_agent=True to use the new multi-agent architecture.
    """

    def __init__(self, api_key: str | None = None, use_multi_agent: bool = True):
        self.api_key = api_key
        self.use_multi_agent = use_multi_agent

        if use_multi_agent:
            # Multi-agent mode: use CoordinatorAgent
            from src.agents.coordinator import CoordinatorAgent
            self.coordinator = CoordinatorAgent(api_key=api_key)
        else:
            # Legacy mode: single ReAct agent with tools
            from agentos.agent.agent import Agent
            from src.tools import search_by_id, search_by_other, recommend_by_age, recommend_by_gender
            self.mediagent = Agent(
                name="mediagent",
                model={},
                tools=[
                    search_by_id(),
                    search_by_other(),
                    recommend_by_age(),
                    recommend_by_gender()
                ],
                api_key=api_key
            )

        self.conn = sqlite3.connect('history.db')
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id TEXT,
                gender TEXT,
                age INTEGER,
                height INTEGER,
                weight INTEGER,
                medical_history TEXT,
                symptoms TEXT,
                recommendation TEXT,
                mode TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def run(self, user_info):
        """Run the recommendation pipeline.

        Args:
            user_info: Dict with keys: id, gender, age, height, weight,
                       medical_history, symptoms.

        Returns:
            Recommendation string.
        """
        if self.use_multi_agent:
            return self._run_multi_agent(user_info)
        else:
            return self._run_legacy(user_info)

    def _run_multi_agent(self, user_info):
        """Run the multi-agent pipeline.

        Uses CoordinatorAgent to orchestrate:
        1. Query decomposition + hybrid retrieval
        2. Symptom analysis
        3. Risk assessment
        4. Recommendation generation
        5. Safety validation (with reflection loop)
        6. Synthesis
        """
        from src.tools import hybrid_v1

        print("=" * 60)
        print("Multi-Agent Pipeline Started")
        print("=" * 60)

        response = self.coordinator.run(
            user_info=user_info,
            retriever=hybrid_v1
        )

        print("\n" + "=" * 60)
        print("RESPONSE")
        print("=" * 60)
        print(response)

        # Save history
        self.save_history(user_info, response, mode="multi_agent")
        return response

    def _run_legacy(self, user_info):
        """Run the legacy single-agent pipeline."""
        from agentos.agent.agent import Agent
        from agentos.memory import Message, Role

        self.mediagent.run(HCR_PROMPT.format(user_info))
        self.mediagent.memory.add_memory(Message(Role.SYSTEM, OUTPUT_PROMPT))
        response = call_model(self.mediagent.memory.memory, self.mediagent.api_key)
        self.mediagent.memory.add_memory(Message(Role.ASSISTANT, response))

        print("\n" + "=" * 60)
        print("MEMORY")
        print("=" * 60)
        for i in self.mediagent.memory.memory:
            print(f"【{i['role']}】")
            print(i['content'])
            print("------------")

        print("\n" + "=" * 60)
        print("RESPONSE")
        print("=" * 60)
        print(response)

        self.save_history(user_info, response, mode="legacy")
        return response

    def save_history(self, user_info, recommendation, mode="multi_agent"):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO history (id, gender, age, height, weight, medical_history, symptoms, recommendation, mode)
            VALUES (?,?,?,?,?,?,?,?,?)
        ''', (
            user_info['id'],
            user_info['gender'],
            user_info['age'],
            user_info['height'],
            user_info['weight'],
            user_info['medical_history'],
            user_info['symptoms'],
            recommendation,
            mode
        ))
        self.conn.commit()

    def get_history(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM history WHERE id =?', (user_id,))
        return cursor.fetchall()


if __name__ == "__main__":
    re = Recommendation(api_key=api_key, use_multi_agent=True)
    info = {
        "id": "426815",
        "gender": "male",
        "age": "50",
        "height": "172",
        "weight": "80",
        "medical_history": "高血压",
        "symptoms": "头晕"
    }
    re.run(info)
