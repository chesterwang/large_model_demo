from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
groq_api_key=os.environ['GROQ_API_KEY']

# 因为网络问题，暂时不使用
llm=ChatGroq(groq_api_key=groq_api_key, model='openai/gpt-oss-120b')

response = llm.invoke("What is the meaning of life?")
print(response)