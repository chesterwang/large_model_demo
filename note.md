
开发原则

1. 模型加载时全部通过文件路径来加载，不通过形如BAAI/bge-m3的repo_id形式加载（huggingface、modelscope等缓存工具）。
2. 模型文件下载方式
    1. 本地 通过huggingface下载， 需要挂代理。
    2. 服务器上通过 modelscope 下载，速度快。
    3. Ollama pull 下载，需要挂代理。
3. 


```Python
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
## Lnagsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## 实例化llm
from langchain_openai import ChatOpenAI
logging.info("start to load llm model")
llm = ChatOpenAI(model="kimi-k2-turbo-preview")
```

```Python
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
```