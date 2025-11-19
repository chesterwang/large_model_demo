import logging
from langsmith import uuid7

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## 实例化llm
logging.info("start to load llm model")
from langchain_openai import ChatOpenAI
LLM_instance = ChatOpenAI(model="kimi-k2-turbo-preview")
response = LLM_instance.invoke("你好")

import langchain_openai

# 也没有开放embedding接口
#langchain_openai.embeddings

# 没有开放completions接口
# from langchain_openai import OpenAI
# LLM_instance = OpenAI(model="kimi-k2-turbo-preview")
# response = LLM_instance.generate(["你好"])
print(response)
logging.info("load llm model succeed")