from agno.agent import Agent
from agno.workflow import Workflow
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# 科学上网 proxy
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"


# 从你的配置文件中导入 "默认" 模型
from llm_config import gemini_model

researcher = Agent(name="Researcher", tools=[DuckDuckGoTools()], model=gemini_model)
writer = Agent(name="Writer", instructions="Write engaging content", model=gemini_model)

workflow = Workflow(
    name="Content Workflow",
    description="A workflow for creating content",
    steps=[researcher, writer],
)

workflow.print_response("Create a blog post about AI agents", stream=True)
