from typing import List, Dict, Any
import json
import utils
import logging
from langchain_openai import ChatOpenAI

CHINESE_INSTRUCTION =  "用中文回答。"

QUERIES_EXAMPLE = [
    "你们最新款智能手机的主要特点是什么？",
    "通常 送达 需要多长时间？",  
    "我在设置新笔记本电脑时遇到了问题，能帮忙吗？",
    "你们对学生有优惠吗？",
    "对于损坏商品的退款政策是怎样的？"
]

class CustomerServiceAgent:
    def __init__(self, llm:ChatOpenAI, name: str, expertise: List[str]):
        self.llm = llm
        self.name = name
        self.expertise = expertise

    def can_handle(self, query: str) -> bool:
        prompt = f"""
        Given the following customer query:
        "{query}"

        And considering this agent's expertise: {', '.join(self.expertise)}

        Determine if this agent can handle the query.
        Return a boolean value (true or false) without explanation.
        """
        response = self.llm.invoke(prompt).content.strip().lower()
        return response == "true"

    def process_query(self, query: str) -> str:
        prompt = f"""
        As a customer service agent with expertise in {', '.join(self.expertise)},
        provide a helpful and friendly response to the following customer query:

        "{query}"

        Ensure your response is:
        1. Accurate and informative
        2. Tailored to the customer's specific question
        3. Empathetic and professional in tone
        4. Concise but comprehensive
        """
        return self.llm.invoke(prompt).content.strip()

class CustomerServiceSystem:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm
        self.agents = [
            CustomerServiceAgent(llm, "Product Specialist", ["product features", "specifications", "comparisons"]),
            CustomerServiceAgent(llm, "Billing Expert", ["pricing", "invoices", "refunds", "subscriptions"]),
            CustomerServiceAgent(llm, "Technical Support", ["troubleshooting", "software issues", "hardware problems"]),
            CustomerServiceAgent(llm, "Shipping and Delivery", ["order tracking", "delivery times", "shipping options"]),
            CustomerServiceAgent(llm, "General Inquiries", ["company information", "policies", "general questions"])
        ]

    def handle_customer_query(self, query: str) -> Dict[str, Any]:
        suitable_agents = [agent for agent in self.agents if agent.can_handle(query)]
        
        if not suitable_agents:
            return {"status": "unhandled", "message": "I'm sorry, but I couldn't find a suitable agent to handle your query. Let me transfer you to a human representative."}

        chosen_agent = suitable_agents[0]  # For simplicity, choose the first suitable agent
        response = chosen_agent.process_query(query)
        
        return {
            "status": "handled",
            "agent_name": chosen_agent.name,
            "response": response
        }

    def analyze_interaction(self, query: str, response: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following customer service interaction:

        Customer Query: "{query}"
        Agent Response: {json.dumps(response, indent=2,ensure_ascii=False)}

        Provide an analysis including:
        1. Appropriateness of the agent selection
        2. Quality and relevance of the response
        3. Customer satisfaction prediction
        4. Areas for improvement
        5. Any missed opportunities in the interaction

        Return your analysis as a JSON object with these sections as keys.
        """
        response:str = self.llm.invoke(prompt + "用中文回答。").content.strip()
        return json.loads(response)


def cli():
    """
    命令行调用
    """
    customer_service_system = CustomerServiceSystem(utils.LLM_instance)

    for query in QUERIES_EXAMPLE:
        print(f"\nCustomer Query: {query}")

        response = customer_service_system.handle_customer_query(query + CHINESE_INSTRUCTION)
        print(f"Agent Response: {json.dumps(response, indent=2,ensure_ascii=False)}")
        
        analysis = customer_service_system.analyze_interaction(query + CHINESE_INSTRUCTION, response)
        print("Interaction Analysis:")
        print(analysis)

# 运行客户服务模拟

def main():
    """
    webUI调用
    """
    import streamlit as st
    from streamlit_smart_text_input import st_smart_text_input
    ## streamlit framework
    st.title('LLM 客服系统')
    # input_text = st.text_input("请输入客户问题：", "")
    query = st_smart_text_input(label="在这里输入客户问题：",options=QUERIES_EXAMPLE)

    customer_service_system = CustomerServiceSystem(utils.LLM_instance)

    if query:
        logging.info(f"\nCustomer Query: {query}")

        response = customer_service_system.handle_customer_query(query + CHINESE_INSTRUCTION)
        st.subheader("Agent Response:")
        if response["status"] == "unhandled":
            st.write(response["message"])
        else:
            st.write(response['response'])
        
        analysis = customer_service_system.analyze_interaction(query + CHINESE_INSTRUCTION, response)
        st.subheader("Interaction Analysis:")
        # st.write(analysis)
        st.write("```json\n" + json.dumps(analysis,indent=4,ensure_ascii=False) + "\n```")


if __name__ == "__main__":
    # cli()
    main()
