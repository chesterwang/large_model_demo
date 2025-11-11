"""
演示本地调用 Ollama的chatbot 接口
"""
import os
from dotenv import load_dotenv

load_dotenv()
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

import streamlit as st

## Lnagsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are ahelpful assistant. Please response tp the user queries"),
        ("user", "Question:{question}")
    ]
)

## streamlit framework
st.title('Langchain Demo with OLLama Local')
input_text = st.text_input("Search the topic u want")

## OLLama LLAma-3 model
logging.info("start to load llm model")
llm = OllamaLLM(model="deepseek-r1:1.5b")
logging.info("llm model load succeed")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
