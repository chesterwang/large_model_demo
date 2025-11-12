
本项目为若干RAG实践项目。

**TOC**

- [环境构建](#环境构建)
- [Project 01 chatbot （LLM模型调用方式）](#project-01-chatbot-llm模型调用方式)
- [Project 02 APIs（prompt模板和chat接口）](#project-02-apisprompt模板和chat接口)
- [Project 03 RAG pipeline （RAG处理流水线）](#project-03-rag-pipeline-rag处理流水线)
- [Project 04 Retriever and Chain （RAG检索处理链条）](#project-04-retriever-and-chain-rag检索处理链条)
- [Project 05 Advanced RAG Q\&A Project （多类型文档QA RAG）](#project-05-advanced-rag-qa-project-多类型文档qa-rag)
- [Project 06 Groq inference （Groq平台接口调用）](#project-06-groq-inference-groq平台接口调用)
- [Project 07 Gen AI （检索搭配文本生成式接口）](#project-07-gen-ai-检索搭配文本生成式接口)
- [Project 08 Powerful Doc Q\&A Chatbot （文档QA）](#project-08-powerful-doc-qa-chatbot-文档qa)
- [Project 11 ImageEnhancer （文生图）](#project-11-imageenhancer-文生图)


## 环境构建

使用uv来构建Python解释器环境。

```Python
uv sync
```

## Project 01 chatbot （LLM模型调用方式）
1. app.py
    1. langchain + openai sdk + kimi k2 的OpenAI兼容api
    2. 调用chain为 `ChatPromptTemplate | Kimi K2 LLM API | StrOutputParser`
2. local_llama.py
    1. langchain + Ollama sdk + Ollama 本地模型服务 + LangSmith tracing服务
    2. 调用chain为 `ChatPromptTemplate | Ollama local LLM API (deepseek-r1:1.5b) | StrOutputParser`

```bash
streamlit run Project\ 01\ chatbot/app.py
```

![alt text](<Project 01 chatbot/image.png>)

```bash
# 安装ollama
# 下载模型
ollama pull deepseek-r1:1.5b

streamlit run Project\ 01\ chatbot/local_llama.py
```

![alt text](<Project 01 chatbot/image-1.png>)

## Project 02 APIs（prompt模板和chat接口）

1. app.py
    1. fastapi 包装两个模型
        1. openai sdk + kimi k2 的OpenAI兼容api
        2. Ollama local LLM API (deepseek-r1:1.5b)
    2. prompt调用chain + fastapi + uvcorn 提供服务

```bash
# 运行服务
python Project\ 02\ APIs/app.py
# 提供界面
streamlit run Project\ 02\ APIs/client.py
```

![image.png](<Project 02 APIs/image.png>)

## Project 03 RAG pipeline （RAG处理流水线）

1. 文档加载器 + splitter + embedding 本地服务 + 向量数据库
    1. langchain 文本加载器(textloader) + 网页加载器(webbaseloader) + pdf文件加载器(pypdfloader)
    2. RecursiveCharacterTextSplitter
    3. HF embedding 本地服务接口（HuggingFaceBgeEmbeddings "BAAI/bge-large-zh-v1.5"）
    4. 向量数据库ChromaDB
    5. 向量数据库FAISS
    6. 最终通过向量数据库接口做相似性性检索

![image.png](<Project 03 RAG pipeline/image.png>)

## Project 04 Retriever and Chain （RAG检索处理链条）

1. 文档加载器 + splitter + embedding 本地服务 + 向量数据库
    1. langchain 文本加载器(textloader) + 网页加载器(webbaseloader) + pdf文件加载器(pypdfloader)
    2. RecursiveCharacterTextSplitter
    3. HF embedding 本地服务接口（HuggingFaceBgeEmbeddings "BAAI/bge-large-zh-v1.5"）
    4. 向量数据库FAISS
2. 构建chain
    1. 向量数据库FAISS
    2. Ollama local LLM API (deepseek-r1:1.5b)
    3. ChatPromptTemplate (包含 context和input 槽)
    4. **最终构建出 chain （向量数据库 | prompt | LLM）**


![alt text](<./Project 04 Retriever and Chain/image.png>)

## Project 05 Advanced RAG Q&A Project （多类型文档QA RAG）

1. 三种tool + tool中间调用过程prompt + 带tools的LLM chat接口 
    1. 三种tool
        1. wikipedia 查询接口
        2. WebBaseLoader 加载网页 + 嵌入向量
            1. HF embedding 本地服务接口（HuggingFaceBgeEmbeddings "BAAI/bge-large-zh-v1.5"）
            2. 相似文档查询 作为 tool
        3. Arxiv 论文查询接口 作为 Tool
    2. 使用 prompt（openai-functions-agent）
        1. 用来格式化 消息历史、query、tool的中间调用信息、
    3. Ollama local LLM API (qwen3:0.6b)
        1. qwen3 支持 tools calling
2. 如下截图 可以实现根据 不同的query 智能调用相关联的tool。


```Python
#prompt(openai-functions-agent)的格式
[SystemMessagePromptTemplate(template='You are a helpful assistant')),
 MessagesPlaceholder(variable_name='chat_history'),
 HumanMessagePromptTemplate(template='{input}')),
 MessagesPlaceholder(variable_name='agent_scratchpad')
 ]
```
```Python
#底层实现机制本质上就是个chain

    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"],
        ),
    )
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
response=agent_executor.invoke({"input":"What's the paper 1605.08386 about?"})
```

![alt text](<Project 05 Advanced RAG Q&A Project/image3.png>)


![alt text](<Project 05 Advanced RAG Q&A Project/image.png>)


## Project 06 Groq inference （Groq平台接口调用）

1. chain （向量数据库 + Groq 接口）
    1. 向量数据库（网页文档经过bge-m3模型embedding）
    2. Groq 调用平台（openai/gpt-oss-120b）
    3. 自定义 prompt 来填充 context（检索出的文档） 和 query

```bash
#运行
streamlit run Project\ 06\ Groq\ inference/app.py
```


![alt text](<Project 06 Groq inference/image.png>)


## Project 07 Gen AI （检索搭配文本生成式接口）

1. embedding本地服务 + FAISS 向量数据库 + 自定义prompt + LLM
    1. HF embedding 本地服务接口（HuggingFaceBgeEmbeddings "BAAI/bge-large-zh-v1.5"）
2. 最终针对query做相似性文档检索 + 根据query查询文档并回答问题
3. 使用 text-generation接口，而非 conversational类接口。

## Project 08 Powerful Doc Q&A Chatbot （文档QA）

1. chain （向量数据库 + 自定义prompt + Groq 接口）
    1. 向量数据库（网页文档经过 BAAI/bge-small-zh-v1.5）
    2. Groq 调用平台（openai/gpt-oss-120b）

```bash
streamlit run Project\ 08\ Powerful\ Doc\ Q\&A\ Chatbot/app.py
```

![alt text](<Project 08 Powerful Doc Q&A Chatbot/image.png>)


## Project 11 ImageEnhancer （文生图）

1. modelscope的openai兼容接口 + 生成图片


```bash
streamlit run Project\ 11\ ImageEnhancer/app.py
```


![alt text](<Project 11 ImageEnhancer/image.png>)