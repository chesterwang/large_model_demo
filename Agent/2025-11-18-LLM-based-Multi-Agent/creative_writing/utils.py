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
logging.info("load llm model succeed")
