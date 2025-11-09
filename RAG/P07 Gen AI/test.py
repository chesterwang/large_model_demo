from dotenv import load_dotenv
load_dotenv()
import os
# 设置http的代理和不代理的地址
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
os.environ['NO_PROXY'] = "http://127.0.0.1:11434" #ollama的本地服务地址

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
)

chat_model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
  SystemMessagePromptTemplate.from_template("You're a helpful assistant."),
  HumanMessagePromptTemplate.from_template("{user_question}"),
])

chain = prompt | chat_model

response = chain.invoke({"user_question": "What is the health insurance coverage?"})

print(response.content)
