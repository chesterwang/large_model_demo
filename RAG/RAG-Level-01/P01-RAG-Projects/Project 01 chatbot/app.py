"""
演示远程调用 chatbot 接口
"""
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
## Lnagsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

##Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
)

## streamlit framework
st.title('Langchain Demo with OPENAI API')
input_text = st.text_input("Search the topic u want")

##openAI LLM
# llm=ChatOpenAI(model="gpt-3.5-turbo")
logging.info("start to load llm model")
llm = ChatOpenAI(model="kimi-k2-turbo-preview")
logging.info("llm model load succeed")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
