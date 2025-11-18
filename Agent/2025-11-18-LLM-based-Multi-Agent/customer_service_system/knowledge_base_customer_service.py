from typing import List, Dict, Any
import json
from datetime import datetime
import utils
import logging
from customer_service import CustomerServiceSystem
from langchain_openai import ChatOpenAI


CHINESE_INSTRUCTION =  "用中文回答。"

# 初始化知识库
KNOWLEDGE_DEFAULT = [
    ("我们最新款智能手机配备 6.7 英寸 OLED 显示屏和 1.08 亿像素摄像头。", "product features"),
    ("标准配送通常需要 3-5 个工作日。加急配送需额外付费。", "shipping"),
    ("设置新笔记本电脑时，请先连接电源适配器，然后按下电源键开机。", "technical support")
]

QUERIES_EXAMPLE = [
    "你们最新款智能手机的主要特点是什么？",
    "通常 送达 需要多长时间？",  
    "我在设置新笔记本电脑时遇到了问题，能帮忙吗？",
    "你们对学生有优惠吗？",
    "对于损坏商品的退款政策是怎样的？"
]

class KnowledgeItem:
    def __init__(self, content: str, category: str, confidence: float = 1.0):
        self.content = content
        self.category = category
        self.confidence = confidence
        self.last_updated = datetime.now()
        self.usage_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "category": self.category,
            "confidence": self.confidence,
            "last_updated": self.last_updated.isoformat(),
            "usage_count": self.usage_count
        }

class DynamicKnowledgeBase:
    def __init__(self, llm:ChatOpenAI):
        self.llm:ChatOpenAI = llm
        self.knowledge_items: List[KnowledgeItem] = []

    def add_item(self, content: str, category: str, confidence: float = 1.0):
        self.knowledge_items.append(KnowledgeItem(content, category, confidence))

    def update_item(self, index: int, content: str = None, category: str = None, confidence: float = None):
        item = self.knowledge_items[index]
        if content:
            item.content = content
        if category:
            item.category = category
        if confidence is not None:
            item.confidence = confidence
        item.last_updated = datetime.now()

    def get_relevant_items(self, query: str) -> List[KnowledgeItem]:
        prompt = f"""
        Given the following customer query:
        "{query}"

        And the following knowledge base items:
        {json.dumps([item.to_dict() for item in self.knowledge_items], indent=2)}

        Return the indices of the most relevant knowledge items for this query.
        Provide the result as a JSON array of integers.
        """
        relevant_indices = json.loads(self.llm.invoke(prompt).content)
        return [self.knowledge_items[i] for i in relevant_indices]

    def learn_from_interaction(self, query: str, response: str):
        prompt = f"""
        Analyze the following customer interaction:

        Query: "{query}"
        Response: "{response}"

        Determine if there's any new information that should be added to the knowledge base.
        If so, provide the new knowledge item in the following JSON format:
        {{
            "content": "The new information to add",
            "category": "The appropriate category for this information",
            "confidence": 0.8  # A confidence score between 0 and 1
        }}

        If no new information should be added, return an empty JSON object {{}}.
        """
        new_item = json.loads(self.llm.invoke(prompt).content)
        if new_item:
            self.add_item(**new_item)

    def update_confidences(self):
        for item in self.knowledge_items:
            age_factor = (datetime.now() - item.last_updated).days / 365.0
            usage_factor = min(item.usage_count / 100, 1.0)
            item.confidence = max(0, min(item.confidence - 0.1 * age_factor + 0.1 * usage_factor, 1.0))


class EnhancedCustomerServiceSystem(CustomerServiceSystem):
    def __init__(self, llm:ChatOpenAI):
        super().__init__(llm)
        self.knowledge_base = DynamicKnowledgeBase(llm)

    def handle_customer_query(self, query: str) -> Dict[str, Any]:
        relevant_items = self.knowledge_base.get_relevant_items(query)
        
        prompt = f"""
        Given the following customer query:
        "{query}"

        And the following relevant knowledge base items:
        {json.dumps([item.to_dict() for item in relevant_items], indent=2)}

        Provide a comprehensive and accurate response to the customer's query.
        Incorporate the relevant information from the knowledge base items.
        """
        response = self.llm.invoke(prompt).content.strip()
        
        self.knowledge_base.learn_from_interaction(query, response)
        self.knowledge_base.update_confidences()
        
        return {
            "status": "handled",
            "response": response,
            "knowledge_items_used": len(relevant_items)
        }

# 使用示例
def cli():

    enhanced_system = EnhancedCustomerServiceSystem(utils.LLM_instance)

    # 初始化知识库
    for item in KNOWLEDGE_DEFAULT:
        enhanced_system.knowledge_base.add_item(**item)

    for query in QUERIES_EXAMPLE:
        print(f"\nCustomer Query: {query}")
        response = enhanced_system.handle_customer_query(query)
        print(f"System Response: {json.dumps(response, indent=2)}")

    print("\nUpdated Knowledge Base:")
    for item in enhanced_system.knowledge_base.knowledge_items:
        print(json.dumps(item.to_dict(), indent=2))


def main():

    import streamlit as st
    from streamlit_smart_text_input import st_smart_text_input
    ## streamlit framework
    st.title('LLM 客服系统（带知识库）')
    # input_text = st.text_input("请输入客户问题：", "")
    query = st_smart_text_input(label="在这里输入客户问题：",options=QUERIES_EXAMPLE)

    customer_service_system = EnhancedCustomerServiceSystem(utils.LLM_instance)

    # 初始化知识库
    for item in KNOWLEDGE_DEFAULT:
        customer_service_system.knowledge_base.add_item(*item)

    if query:
        logging.info(f"\nCustomer Query: {query}")

        response = customer_service_system.handle_customer_query(query + CHINESE_INSTRUCTION)
        st.subheader("Agent Response:")
        if response["status"] == "unhandled":
            st.write(response["message"])
        else:
            st.write(f"下面的回复使用了{response['knowledge_items_used']}条知识库信息：")
            st.write(response['response'])

        st.subheader("现行所有的knowledge_base如下:")
        show_result = [item.to_dict() for item in customer_service_system.knowledge_base.knowledge_items]
            # # print(json.dumps(item.to_dict(), indent=2))
            # show_result += json.dumps(item.to_dict(), indent=2,ensure_ascii=False) + "\n"
        st.write("```json\n" + json.dumps(show_result, indent=4, ensure_ascii=False) + "\n```")

if __name__ == "__main__":
    # 运行增强版客户服务模拟
    # cli()
    main()
