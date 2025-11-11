
开发原则

1. 模型加载时全部通过文件路径来加载，不通过形如BAAI/bge-m3的repo_id形式加载（huggingface、modelscope等缓存工具）。
2. 模型文件下载方式
    1. 本地 通过huggingface下载， 需要挂代理。
    2. 服务器上通过 modelscope 下载，速度快。
    3. Ollama pull 下载，需要挂代理。
3. 