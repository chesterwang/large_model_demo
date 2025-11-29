# src/llm_config.py

from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()

# --- 在这里定义你的模型 ---

# 1. 定义一个 "默认" 的 Kimi 模型
# 这个模型将从环境变量中读取配置
kimi_model=OpenAIChat(id="kimi-k2-0711-preview",
                    base_url=os.getenv("OPENAI_API_BASE"), 
                    api_key=os.getenv("OPENAI_API_KEY"), 
                    role_map = {
                        "system": "system",
                        "user": "user",
                        "assistant": "assistant",
                        "tool": "tool",
                        "model": "assistant",}
                        )

# 2. 你也可以定义一个 OpenAI 的模型作为备选
openai_model = OpenAIChat(
    id="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"), # 使用不同的环境变量
)

gemini_model = Gemini(
    id="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"), 
)

# --- 你可以根据需要添加更多模型 ---