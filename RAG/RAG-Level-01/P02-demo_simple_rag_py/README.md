

本项目为RAG原理实践项目。 软件依赖极少，很好的展示RAG原理性。

**整体架构： Ollama bemdding接口 + dict存储向量数据 + Ollama chat接口**

1. embedding模型 'CompendiumLabs/bge-base-en-v1.5-gguf'
2. LLM模型 'bartowski/Llama-3.2-1B-Instruct-GGUF'


```Python
Python demo.py
```
![alt text](image.png)


## 实现细节

1. 在ChatModel的stream模式下，返回是一个词一个词的进行返回的。
2. 在该demo示例中，检索到的文档是全被塞到一个system message下，然后后续的对话但就是根据此来进行交互。
3. **模型文件下载若干问题**
    1. bartowski/Llama-3.2-1B-Instruct-GGUF 这个模型中的instruct的意思
        1. instruct意思就是指模型是经过 在base model上经过指令微调（或者叫监督微调）过程之后产生的模型，可以直接用于聊天、助手类应用
    2. GGUF 文件包含了tokenizer的数据。
    3. ollama 拉取 hf.co 网站的模型失败
        1. `Ollama pull 进度已经100%， 但为什么会报如下错误`
            1. `Error: max retries exceeded: Get "https://huggingface.co/v2/bartowski/Llama-3.2-1B-Instruct-GGUF/blobs/sha256:948af2743fc78a328dcb3b0f5a31b3d75f415840fdb699e8b1235978392ecf85?__sign=eyJhbGciOiJFZERTQSJ9.eyJyZWFkIjp0cnVlLCJwZXJtaXNzaW9ucyI6eyJyZXBvLmNvbnRlbnQucmVhZCI6dHJ1ZX0sImlhdCI6MTc2Mjc1MzA3Niwic3ViIjoiL2JhcnRvd3NraS9MbGFtYS0zLjItMUItSW5zdHJ1Y3QtR0dVRiIsImV4cCI6MTc2Mjc1MzY3NiwiaXNzIjoiaHR0cHM6Ly9odWdnaW5nZmFjZS5jbyJ9.wVhqKCuc0vtHoS8DUStd86RoS1dhE0e-IqqMIpLZ_mEbfz6ahWeaoQiuTpWuaJCels7Q7uIIwEky9DhHsynYCg": dial tcp 54.89.135.129:443: connect: connection refused`
        2. 检查网络是可以连接 hf.co网站的。
        3. 找不到原因。
    4. 手动下载gguf模型文件，然后本地进行ollama手动构建。
        1. 手动下载 https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/tree/main 中的 Llama-3.2-1B-Instruct-Q4_K_M.gguf文件。
        2. 然后根据文档[Importing a Model - Ollama](https://docs.ollama.com/import#importing-a-gguf-based-model-or-adapter)来导入到本地ollama的缓存目录。
        3. 创建一个临时空目录
        4. 创建 ModelFile文件，内容为 `FROM ./Llama-3.2-1B-Instruct-Q4_K_M.gguf`
        5. 执行命令`ollama create bartowski/Llama-3.2-1B-Instruct-GGUF`
        6. 模型就可以被构建缓并复制到 ollama的缓存目录中。
        7. 并同样构建 bge-base-en-v1.5-q4_k_m.gguf 模型
        8. `ollama.embed(model="CompendiumLabs/bge-base-en-v1.5-gguf")`
            1. 这里的模型名按照/来分割，即说明是本地文件的模型。
            2. 如果是正常的:分割的模型名，则ollama会自动从缓存目录中的 library子目录中查找。
    5. ollama 也可以在本地对模型进行量化。 [quantizing-a-model - Ollama](https://docs.ollama.com/import#quantizing-a-model)
    6.  ollama对仓库模型的选择
        1. ollama run model_repo_id，如果repo中有多版本模型，那么ollama会选择Q4版本的。更好的方法是显式指定具体版本 `ollama run hf.co/{username}/{repository}:{quantization}`
            1. 比如 `ollama run hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-IQ3_M.gguf`
        2. 参考资料 [Use Ollama with any GGUF Model on Hugging Face Hub](https://huggingface.co/docs/hub/en/ollama)
4. 参考资料
    1. 关联博文 [Code a simple RAG from scratch](https://huggingface.co/blog/ngxson/make-your-own-rag)
    2. 项目代码 [demo.py · ngxson/demo\_simple\_rag\_py at main](https://huggingface.co/ngxson/demo_simple_rag_py/blob/main/demo.py)
