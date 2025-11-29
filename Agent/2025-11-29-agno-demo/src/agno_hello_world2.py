from agno.agent import Agent
from agno.models.siliconflow import Siliconflow
from agno.tools.hackernews import HackerNewsTools

# 0. 环境配置

from dotenv import load_dotenv
import os

load_dotenv()
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


agent = Agent(
    model=Siliconflow(
        id="deepseek-ai/DeepSeek-V3.1-Terminus",
        name="deepseek-ai/DeepSeek-V3.1-Terminus",
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL
    ),
    tools=[HackerNewsTools()],
    markdown=True,
)
agent.print_response("Write a report on trending startups.", stream=True)