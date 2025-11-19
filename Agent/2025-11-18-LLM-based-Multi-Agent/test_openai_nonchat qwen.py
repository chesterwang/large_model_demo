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
os.environ["OPENAI_API_BASE"] = os.getenv("BAILIAN_OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"] = os.getenv("BAILIAN_OPENAI_API_KEY")


## 实例化llm
logging.info("start to load llm model")
from langchain_openai import ChatOpenAI
LLM_instance = ChatOpenAI(model="qwen-max")
response = LLM_instance.invoke("你好")

# dashscope并不兼容该接口，接口类型和背后的java工程类型是不一致的。
# from langchain_openai import OpenAI
# LLM_instance:OpenAI = OpenAI(model="qwen-coder-turbo")
# prompt = "<|fim_prefix|>写一个python的快速排序函数，def quick_sort(arr):<|fim_suffix|>"
# response = LLM_instance.generate([prompt])

print(response)
logging.info("load llm model succeed")

# embedding接口 也不兼容 应该是数据类型不兼容。
# from langchain_openai import OpenAIEmbeddings
# embedding_model = OpenAIEmbeddings(model="text-embedding-v4", chunk_size=1024,dimensions=1024)
# response = embedding_model.embed_query("你好，欢迎使用百炼大模型服务平台")
# print(response)
