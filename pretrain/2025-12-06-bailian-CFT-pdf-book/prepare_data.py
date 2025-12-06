import json
import os
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/home/chester/Downloads/UnderstandingDeepLearning_01_10_24_C.pdf")
docs = loader.load()
docs

# 修正：langchain 的 Document 对象使用点语法 (.page_content) 访问内容
pages_content = [
    json.dumps({"text": doc.page_content}, ensure_ascii=False) for doc in docs
]

# 定义输出文件名
output_file = os.path.join(os.path.dirname(__file__),"output", "book_content.jsonl")

# 将 pages_content 数组保存到文本文件中，每行一个 JSON 对象
with open(output_file, "w", encoding="utf-8") as f:
    for line in pages_content:
        line = line.strip()

        cnt: Counter = Counter(line)
        total_cnt = cnt.total()
        max_key, max_cnt = cnt.most_common(1)[0]

        if len(line) > 30 and max_cnt / total_cnt < 0.2:
            f.write(line + "\n")

print(f"数据已成功保存到 {output_file}")


if __name__ == "__main__":
    pass
