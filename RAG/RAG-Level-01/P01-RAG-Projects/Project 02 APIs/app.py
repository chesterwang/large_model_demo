from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

## FAST API APP INITIALIZATION

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

## making routs

# add_routes(
#     app,
#     ChatOpenAI(model="kimi-k2-turbo-preview"),
#     path="/openai"
# )

model = ChatOpenAI(model="kimi-k2-turbo-preview")

## ollama llama3
llm = OllamaLLM(model="deepseek-r1:1.5b")

prompt1 = ChatPromptTemplate.from_template("Write an assay for me about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write an poem for me about {topic} with 100 words")

add_routes(
    app,
    prompt1 | model,
    path="/assay"
)

add_routes(
    app,
    prompt2 | llm,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9000)
