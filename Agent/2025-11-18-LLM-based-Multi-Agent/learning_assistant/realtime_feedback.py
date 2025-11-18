from typing import List, Dict, Any
import json
import utils
import logging
from langchain_openai import ChatOpenAI

class RealTimeFeedbackSystem:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def provide_instant_feedback(self, user_response: str, correct_answer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Provide instant feedback for the following user response:
        User Response: "{user_response}"
        Correct Answer: "{correct_answer}"
        Context: {json.dumps(context, indent=2)}

        Generate feedback that:
        1. Assesses the correctness of the response
        2. Provides encouraging and constructive comments
        3. Offers hints or explanations if the answer is incorrect
        4. Suggests next steps or additional resources

        Return your feedback as a JSON object with the following fields:
        - correctness_score (0-100)
        - feedback_message
        - hints (if applicable)
        - explanation (if applicable)
        - next_steps
        """
        response = self.llm.invoke(prompt)
        return json.loads(response.content.replace('\n','').replace('```json','').replace('```','').strip())

class ContinuousAssessmentSystem:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def assess_learning(self, user_id: str, learning_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""
        Perform a continuous assessment based on the following learning history:
        {json.dumps(learning_history, indent=2)}

        Provide an assessment that includes:
        1. Overall progress in key topic areas
        2. Identification of any learning gaps or misconceptions
        3. Trends in performance over time
        4. Recommendations for areas to focus on
        5. Suggested adjustments to the learning path

        Return your assessment as a JSON object with these sections as keys.
        """
        response = self.llm.invoke(prompt)
        return json.loads(response.content.replace('\n','').replace('```json','').replace('```','').strip())

from teaching_agent import MultiStrategyLearningAssistant

class AdvancedPersonalizedLearningAssistant(MultiStrategyLearningAssistant):
    def __init__(self, llm:ChatOpenAI):
        super().__init__(llm)
        self.real_time_feedback = RealTimeFeedbackSystem(llm)
        self.continuous_assessment = ContinuousAssessmentSystem(llm)

    def submit_answer(self, user_id: str, question: str, user_answer: str, correct_answer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        feedback = self.real_time_feedback.provide_instant_feedback(user_answer, correct_answer, context)
        self.progress_tracker.record_learning_activity(user_id, {
            "type": "question_answer",
            "question": question,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "feedback": feedback
        })
        return feedback

    def get_learning_assessment(self, user_id: str) -> Dict[str, Any]:
        learning_history = self.progress_tracker.learning_records.get(user_id, [])
        return self.continuous_assessment.assess_learning(user_id, learning_history)

    def adjust_learning_path(self, user_id: str, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = f"""
        Based on the following learning assessment:
        {json.dumps(assessment, indent=2)}

        Adjust the learning path for the user. Provide:
        1. Topics to prioritize
        2. Topics to review
        3. Recommended learning activities
        4. Suggested pace of learning

        Return the adjusted learning path as a JSON array of objects, each representing a recommended learning step.
        """
        response = self.llm.invoke(prompt)
        return json.loads(response.content.replace('\n','').replace('```json','').replace('```','').strip())

# 使用示例
def run_advanced_personalized_learning_assistant_simulation():
    advanced_assistant = AdvancedPersonalizedLearningAssistant(utils.LLM_instance)

    user_id = "student_789"
    learning_goals = ["Master Python for data science", "Understand advanced statistical methods"]

    # 开始学习会话
    session = advanced_assistant.start_learning_session(user_id, learning_goals)
    print("Initial Learning Session:")
    print(json.dumps(session, indent=2))

    # 模拟一系列问答交互
    questions = [
        {
            "question": "What is the primary purpose of the pandas library in Python?",
            "correct_answer": "Data manipulation and analysis",
            "context": {"topic": "Python for Data Science", "difficulty": "Beginner"}
        },
        {
            "question": "In statistics, what does p-value represent?",
            "correct_answer": "The probability of obtaining test results at least as extreme as the observed results, assuming the null hypothesis is true",
            "context": {"topic": "Statistical Methods", "difficulty": "Intermediate"}
        },
        {
            "question": "What is the difference between supervised and unsupervised learning in machine learning?",
            "correct_answer": "Supervised learning uses labeled data, while unsupervised learning does not use labeled data",
            "context": {"topic": "Machine Learning Basics", "difficulty": "Intermediate"}
        }
    ]

    for q in questions:
        print(f"\nQuestion: {q['question']}")
        user_answer = input("Your answer: ")  # In a real system, this would come from the user interface
        feedback = advanced_assistant.submit_answer(user_id, q['question'], user_answer, q['correct_answer'], q['context'])
        print("Feedback:")
        print(json.dumps(feedback, indent=2))

    # 获取学习评估
    assessment = advanced_assistant.get_learning_assessment(user_id)
    print("\nLearning Assessment:")
    print(json.dumps(assessment, indent=2))

    # 调整学习路径
    adjusted_path = advanced_assistant.adjust_learning_path(user_id, assessment)
    print("\nAdjusted Learning Path:")
    print(json.dumps(adjusted_path, indent=2))

    # 获取下一个建议的活动
    next_activity = advanced_assistant.get_next_activity(user_id)
    print("\nNext Recommended Activity:")
    print(json.dumps(next_activity, indent=2))


if __name__ == "__main__":
    # 运行高级个性化学习助手模拟
    run_advanced_personalized_learning_assistant_simulation()