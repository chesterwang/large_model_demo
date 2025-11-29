from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

## 代码无法调试通过，报错为 agno.exceptions.ModelProviderError: invalid request: unsupported role ROLE_UNSPECIFIED

# 0. 环境配置

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# 科学上网 proxy
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# 1. 初始化 Agent
# - model: 指定使用的模型 (这里使用 GPT-4o)
# - tools: 赋予 Agent 使用搜索引擎的能力
# - debug_mode: 开启调试模式，打印工具调用、模型输出等详细信息
# - markdown: 输出格式化为 Markdown
web_agent = Agent(
    name="Web Researcher",
    # model=OpenAIChat(id="gpt-4o",api_key=""),
    model=OpenAIChat(id="kimi-k2-0711-preview",
                     base_url=OPENAI_API_BASE, 
                     api_key=OPENAI_API_KEY, 
                     role_map = {
                        "system": "system",
                        "user": "user",
                        "assistant": "assistant",
                        "tool": "tool",
                        "model": "assistant",
                    }),
    tools=[DuckDuckGoTools()],
    # instructions=["Always include sources in your response."],
    debug_mode=True,
    markdown=True,
)

# 2. 运行 Agent
# Agent 会自动判断需要搜索网络来回答这个问题
print("--- Agent Running ---")
prompt = """
Always include sources in your response.

What are the latest key features of the Agno (Phidata) framework?
"""
web_agent.print_response(prompt, stream=True)


