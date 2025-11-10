import streamlit as st
import os
import time

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

import os

# 设置http的代理和不代理的地址
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
os.environ['NO_PROXY'] = "http://127.0.0.1:11434"  # ollama的本地服务地址

## load the GROQ API Key and OPENAI APY key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os

logger.info(f"cwd {os.getcwd()}")

st.title("(RAG) App: LLama-3 ChatBot-🤖")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="openai/gpt-oss-120b")
# from langchain_community.chat_models import ChatOllama
# llm = ChatOllama(model="qwen3:0.6b")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based o the provided context only.
    Please provide the most accurate response based on the question
    <cotext>
    {context}
    <context>
    Questions:{input}
    """
)


def vector_embeddins():
    if "vectors" not in st.session_state:
        logger.info(f"start embedding")
        # st.session_state.embeddings = OllamaEmbeddings(model="bge-m3:latest")

        from langchain_huggingface import HuggingFaceEmbeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",  # sentence-transformers/all-MiniLM-16-v2
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # st.session_state.embeddings=OpenAIEmbeddings()
        doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'papers')
        logger.info(f"loader path {doc_path}")
        st.session_state.loader = PyPDFDirectoryLoader(doc_path)  ## Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  ##Document Loading
        logger.info(f"docs length {len(st.session_state.docs)}")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                                        chunk_overlap=200)  ## chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs)  ## splitting
        logger.info(f"final_documents length {len(st.session_state.final_documents)}")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
                                                        st.session_state.embeddings)  ## vector store OpenAi embeddings


prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embeddings"):
    logger.info("start refresh embedding")
    vector_embeddins()
    logger.info("finish refresh embedding")
    st.write("Vector Store DB is Ready")

## creating prompt
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # with a streamlit expander
    with st.expander("Document Similarity Search"):
        # find the relavent chunk
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------------------------------")
