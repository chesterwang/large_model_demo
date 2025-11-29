from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.db.sqlite import AsyncSqliteDb
from agno.os import AgentOS
import os

os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
from llm_config import gemini_model

agent = Agent(
    name="Support Agent",
    db=AsyncSqliteDb(db_file="agno.db"),
    add_history_to_context=True,
    model=gemini_model,
)

agent_os = AgentOS(agents=[agent])
app = agent_os.get_app()  # FastAPI app ready to deploy

if __name__ == "__main__":
    agent_os.serve(app="agno_agentos:app", port=9000,reload=True)  # 启动服务器

# 然后按照 https://docs.agno.com/agent-os/connecting-your-os 进行设置， 并操作 control plane。

