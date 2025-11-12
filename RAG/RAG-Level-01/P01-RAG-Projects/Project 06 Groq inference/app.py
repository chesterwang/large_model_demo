import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv()
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置http的代理和不代理的地址
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
os.environ['NO_PROXY'] = "127.0.0.1,localhost" #ollama的本地服务地址

# load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

if "vectors" not in st.session_state:
    logger.info("vector store started")
    st.session_state.embeddings=OllamaEmbeddings(model="bge-m3:latest")
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()
    
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    logger.info("vector store finished")
    
st.title("Chat-Groq Agent")

# 因为网络地区问题，groq服务虽然可以连接，但是其服务器因为ip归属拒绝提供服务，所以要设置代理
llm=ChatGroq(groq_api_key=groq_api_key, model='openai/gpt-oss-120b')
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="qwen3:0.6b")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate respobse based o the question
<context>
{context}
<context>
Question:{input} 
"""
)

document_chain= create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain= create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input you prompt here")


if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :", time.process_time()-start)
    st.write(response['answer'])
    
    
    #with a streamlit expander
    with st.expander("Documnet Similarity Search"):
        #find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------------------")