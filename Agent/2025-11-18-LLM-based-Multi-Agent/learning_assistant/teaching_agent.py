from typing import List, Dict, Any
import json
import utils
import logging
from langchain_openai import ChatOpenAI

class TeachingStrategyAgent:
    def __init__(self, llm:ChatOpenAI, strategy_name: str, strategy_description: str):
        self.llm = llm
        self.strategy_name = strategy_name
        self.strategy_description = strategy_description

    def generate_learning_activity(self, topic: str, user_progress: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Generate a learning activity using the {self.strategy_name} teaching strategy.
        Strategy description: {self.strategy_description}

        Topic: {topic}
        User Progress: {json.dumps(user_progress, indent=2)}

        Create an engaging learning activity that:
        1. Aligns with the teaching strategy
        2. Is appropriate for the user's current level
        3. Addresses the specified topic
        4. Includes clear learning objectives
        5. Provides step-by-step instructions or guidelines

        Return the activity as a JSON object with the following fields:
        - activity_type
        - title
        - description
        - learning_objectives
        - steps_or_guidelines
        - materials_needed (if any)
        - estimated_duration
        - difficulty_level
        """
        response = self.llm.invoke(prompt)
        return json.loads(response.content.replace('\n','').replace('```json','').replace('```','').strip())

    def evaluate_activity_effectiveness(self, activity: Dict[str, Any], user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Evaluate the effectiveness of the following learning activity:
        {json.dumps(activity, indent=2)}

        Based on the user feedback:
        {json.dumps(user_feedback, indent=2)}

        Provide an evaluation including:
        1. How well the activity aligned with the {self.strategy_name} strategy
        2. The activity's effectiveness in achieving its learning objectives
        3. User engagement and satisfaction level
        4. Areas for improvement
        5. Recommendations for future activities

        Return your evaluation as a JSON object with these sections as keys.
        """
        response = self.llm.invoke(prompt)
        return json.loads(response.content.replace('\n','').replace('```json','').replace('```','').strip())

from learning_assistant import LearningProgressTracker, AdaptiveLearningPathGenerator, PersonalizedLearningAssistant

class MultiStrategyLearningAssistant(PersonalizedLearningAssistant):
    def __init__(self, llm:ChatOpenAI):
        super().__init__(llm)
        self.teaching_strategies = [
            TeachingStrategyAgent(llm, "General", "A balanced approach combining various teaching methods"),
            TeachingStrategyAgent(llm, "Visual Learning", "Emphasizes visual aids, diagrams, and graphical representations"),
            TeachingStrategyAgent(llm, "Hands-on Learning", "Focuses on practical exercises and real-world applications"),
            TeachingStrategyAgent(llm, "Collaborative Learning", "Encourages group work and peer-to-peer learning"),
            TeachingStrategyAgent(llm, "Gamification", "Incorporates game-like elements to increase engagement"),
            TeachingStrategyAgent(llm, "Spaced Repetition", "Utilizes repeated review of material at increasing intervals")
        ]

    def get_next_activity(self, user_id: str) -> Dict[str, Any]:
        progress = self.progress_tracker.get_learning_progress(user_id)
        topic = self._determine_next_topic(progress)
        strategy = self._select_best_strategy(progress)
        return strategy.generate_learning_activity(topic, progress)

    def _determine_next_topic(self, progress: Dict[str, Any]) -> str:
        prompt = f"""
        Based on the following learning progress:
        {json.dumps(progress, indent=2)}

        Determine the most appropriate next topic for the user to study.
        Return only the name of the topic as a string.
        """
        return self.llm.invoke(prompt).content.strip()

    def _select_best_strategy(self, progress: Dict[str, Any]) -> TeachingStrategyAgent:
        prompt = f"""
        Based on the following learning progress:
        {json.dumps(progress, indent=2)}

        Select the most appropriate teaching strategy from the following options:
        {', '.join([strategy.strategy_name for strategy in self.teaching_strategies])}

        Return only the name of the selected strategy.
        """
        selected_strategy_name = self.llm.invoke(prompt).content.strip()
        return next(strategy for strategy in self.teaching_strategies if strategy.strategy_name == selected_strategy_name)

    def provide_learning_feedback(self, user_id: str, activity: Dict[str, Any], user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        valid_strategy = [s for s in self.teaching_strategies if s.strategy_name == activity.get('activity_type', '')]
        if not valid_strategy:
            logging.warning(f"No valid teaching strategy found for activity type: {activity.get('activity_type', '')}. Using General strategy.")
            valid_strategy = self.teaching_strategies[:1]
        strategy = valid_strategy[0]
        evaluation = strategy.evaluate_activity_effectiveness(activity, user_feedback)
        self.progress_tracker.record_learning_activity(user_id, {**activity, "feedback": user_feedback, "evaluation": evaluation})
        return evaluation

# 使用示例
def run_multi_strategy_learning_assistant_simulation():
    multi_strategy_assistant = MultiStrategyLearningAssistant(utils.LLM_instance)

    user_id = "student_456"
    learning_goals = ["Learn data analysis with Python", "Understand statistical concepts"]

    # 开始学习会话
    session = multi_strategy_assistant.start_learning_session(user_id, learning_goals)
    print("Initial Learning Session:")
    print(json.dumps(session, indent=2,ensure_ascii=False))

    # 获取并完成几个学习活动
    for _ in range(3):
        activity = multi_strategy_assistant.get_next_activity(user_id)
        print(f"\nNext Activity:")
        print(json.dumps(activity, indent=2,ensure_ascii=False))

        # 模拟用户反馈
        user_feedback = {
            "completion_status": "completed",
            "enjoyment_level": 8,
            "difficulty_level": 6,
            "time_spent": 45,
            "comments": "Enjoyed the interactive elements, but found some concepts challenging."
        }

        feedback = multi_strategy_assistant.provide_learning_feedback(user_id, activity, user_feedback)
        print("\nActivity Feedback and Evaluation:")
        print(json.dumps(feedback, indent=2,ensure_ascii=False))

    # 获取学习建议
    recommendations = multi_strategy_assistant.provide_learning_recommendations(user_id)
    print("\nPersonalized Learning Recommendations:")
    print(json.dumps(recommendations, indent=2,ensure_ascii=False))


if __name__ == "__main__":
    # 运行多策略个性化学习助手模拟
    run_multi_strategy_learning_assistant_simulation()