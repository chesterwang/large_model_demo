from typing import Dict, Any
import json
import utils
import logging
from langchain_openai import ChatOpenAI
from knowledge_base_customer_service import EnhancedCustomerServiceSystem

CHINESE_INSTRUCTION =  "用中文回答。"

QUERIES_EXAMPLE = [
    "你们最新款智能手机的主要特点是什么？",
    "通常 送达 需要多长时间？",  
    "我在设置新笔记本电脑时遇到了问题，能帮忙吗？",
    "你们对学生有优惠吗？",
    "对于损坏商品的退款政策是怎样的？"
]

class EmotionRecognizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def recognize_emotion(self, text: str) -> Dict[str, float]:
        prompt = f"""
        Analyze the emotional content of the following text:
        "{text}"

        Provide emotion scores for the following emotions:
        - Joy
        - Sadness
        - Anger
        - Fear
        - Surprise

        Return the results as a compact JSON object with emotions as keys and scores (0 to 1) as values.
        """
        response = self.llm.invoke(prompt)
        return json.loads(response.content)

class PersonalizedResponseGenerator:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def generate_response(self, query: str, emotion_scores: Dict[str, float], base_response: str) -> str:
        prompt = f"""
        Given the following:

        Customer Query: "{query}"
        Emotion Scores: {json.dumps(emotion_scores, indent=2)}
        Base Response: "{base_response}"

        Generate a personalized response that:
        1. Addresses the customer's query
        2. Takes into account their emotional state
        3. Aims to improve their emotional state if negative
        4. Maintains a professional and empathetic tone

        Provide only the generated response, without any additional explanation.
        """
        return self.llm.invoke(prompt).content

class EmotionallyIntelligentCustomerServiceSystem(EnhancedCustomerServiceSystem):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.emotion_recognizer = EmotionRecognizer(llm)
        self.personalized_response_generator = PersonalizedResponseGenerator(llm)

    def handle_customer_query(self, query: str) -> Dict[str, Any]:
        emotion_scores = self.emotion_recognizer.recognize_emotion(query)
        base_response = super().handle_customer_query(query)["response"]
        
        personalized_response = self.personalized_response_generator.generate_response(
            query, emotion_scores, base_response
        )
        
        return {
            "status": "handled",
            "response": personalized_response,
            "emotion_scores": emotion_scores
        }

    def analyze_emotional_interaction(self, query: str, response: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following emotionally-aware customer service interaction:

        Customer Query: "{query}"
        Emotion Scores: {json.dumps(response['emotion_scores'], indent=2)}
        System Response: "{response['response']}"

        Provide an analysis including:
        1. Appropriateness of the emotional recognition
        2. Effectiveness of the personalized response in addressing the customer's emotional state
        3. Potential impact on customer satisfaction
        4. Suggestions for improving emotional intelligence in future interactions

        Return your analysis as a JSON object with these sections as keys.
        """
        return json.loads(self.llm.invoke(prompt).content)


# 使用示例
def cli():

    emotional_system = EmotionallyIntelligentCustomerServiceSystem(utils.LLM_instance)

    queries = [
        "I'm really frustrated that my order hasn't arrived yet. It's been two weeks!",
        "I'm so excited about your new product launch! When can I pre-order?",
        "I'm worried that I might have been charged twice for my last purchase. Can you check?",
        "I'm disappointed with the quality of the product I received. It's not what I expected at all.",
        "I'm grateful for the excellent service your team provided last week. Thank you!"
    ]

    for query in queries:
        print(f"\nCustomer Query: {query}")
        response = emotional_system.handle_customer_query(query)
        print(f"System Response: {json.dumps(response, indent=2, ensure_ascii=False)}")
        
        analysis = emotional_system.analyze_emotional_interaction(query, response)
        print("Interaction Analysis:")
        print(json.dumps(analysis, indent=2,ensure_ascii=False))


def main():
    """
    webUI调用
    """
    import streamlit as st
    from streamlit_smart_text_input import st_smart_text_input
    ## streamlit framework
    st.title('LLM 客服系统（带客户情绪智能识别功能）')
    # input_text = st.text_input("请输入客户问题：", "")
    query = st_smart_text_input(label="在这里输入客户问题：",options=QUERIES_EXAMPLE)

    customer_service_system = EmotionallyIntelligentCustomerServiceSystem(utils.LLM_instance)

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
    # 运行情感智能客户服务模拟
    # cli()
    main()