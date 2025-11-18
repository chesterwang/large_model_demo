from typing import List, Dict, Any
import json
from datetime import datetime
import utils
import logging
from langchain_openai import ChatOpenAI

class LearningProgressTracker:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm
        self.learning_records = {}

    def record_learning_activity(self, user_id: str, activity: Dict[str, Any]):
        if user_id not in self.learning_records:
            self.learning_records[user_id] = []
        self.learning_records[user_id].append({
            "timestamp": datetime.now().isoformat(),
            **activity
        })

    def get_learning_progress(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.learning_records:
            return {"error": "No learning records found for this user"}

        prompt = f"""
        Analyze the following learning records for a user:
        {json.dumps(self.learning_records[user_id], indent=2)}

        Provide a learning progress report including:
        1. Topics covered and mastery level for each
        2. Areas of strength
        3. Areas needing improvement
        4. Learning pace and consistency
        5. Overall progress assessment

        Return your analysis as a JSON object with these sections as keys.
        """
        return json.loads(self.llm.invoke(prompt).content.replace('\n','').replace('```json','').replace('```','').strip())

class AdaptiveLearningPathGenerator:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def generate_path(self, learning_progress: Dict[str, Any], learning_goals: List[str]) -> List[Dict[str, Any]]:
        prompt = f"""
        Given the following learning progress:
        {json.dumps(learning_progress, indent=2)}

        And the learning goals:
        {json.dumps(learning_goals, indent=2)}

        Generate an adaptive learning path that:
        1. Addresses areas needing improvement
        2. Builds upon existing strengths
        3. Aligns with the specified learning goals
        4. Provides a balanced and engaging learning experience

        Return the learning path as a JSON array of objects, each representing a learning activity or module, including:
        - Topic
        - Difficulty level
        - Estimated time to complete
        - Prerequisites (if any)
        - Learning objectives
        - Recommended resources or materials

        Ensure the path is personalized based on the user's progress and goals.
        """
        response = self.llm.invoke(prompt)
        return json.loads(response.content.replace('\n','').replace('```json','').replace('```','').strip())

class PersonalizedLearningAssistant:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm
        self.progress_tracker = LearningProgressTracker(llm)
        self.path_generator = AdaptiveLearningPathGenerator(llm)

    def start_learning_session(self, user_id: str, learning_goals: List[str]) -> Dict[str, Any]:
        progress = self.progress_tracker.get_learning_progress(user_id)
        learning_path = self.path_generator.generate_path(progress, learning_goals)
        return {
            "user_id": user_id,
            "learning_goals": learning_goals,
            "current_progress": progress,
            "recommended_path": learning_path
        }

    def complete_learning_activity(self, user_id: str, activity: Dict[str, Any]) -> Dict[str, Any]:
        self.progress_tracker.record_learning_activity(user_id, activity)
        updated_progress = self.progress_tracker.get_learning_progress(user_id)
        return {
            "user_id": user_id,
            "completed_activity": activity,
            "updated_progress": updated_progress
        }

    def get_next_activity(self, user_id: str) -> Dict[str, Any]:
        progress = self.progress_tracker.get_learning_progress(user_id)
        current_path = self.path_generator.generate_path(progress, [])  # Assuming goals are stored elsewhere
        return current_path[0] if current_path else {"message": "Learning path completed"}

    def provide_learning_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        progress = self.progress_tracker.get_learning_progress(user_id)
        prompt = f"""
        Based on the following learning progress:
        {json.dumps(progress, indent=2)}

        Provide personalized learning recommendations, including:
        1. Suggested topics to focus on next
        2. Study techniques that might be effective for this learner
        3. Additional resources or materials that could enhance learning
        4. Potential challenges the learner might face and how to overcome them

        Return your recommendations as a JSON array of objects, each with a 'type' and 'description' field.
        """
        return json.loads(self.llm.invoke(prompt).content.replace('\n','').replace('```json','').replace('```','').strip())

# 使用示例
def run_personalized_learning_assistant_simulation():
    learning_assistant = PersonalizedLearningAssistant(utils.LLM_instance)

    user_id = "student_123"
    learning_goals = ["Master Python programming", "Understand machine learning basics"]

    # 开始学习会话
    session = learning_assistant.start_learning_session(user_id, learning_goals)
    print("Initial Learning Session:")
    print(json.dumps(session, indent=2))

    # 模拟完成几个学习活动
    activities = [
        {"topic": "Python basics", "performance": "good", "time_spent": 120},
        {"topic": "Data structures in Python", "performance": "excellent", "time_spent": 90},
        {"topic": "Introduction to machine learning", "performance": "average", "time_spent": 150}
    ]

    for activity in activities:
        result = learning_assistant.complete_learning_activity(user_id, activity)
        print(f"\nCompleted Activity: {activity['topic']}")
        print(json.dumps(result, indent=2))

    # 获取下一个建议的活动
    next_activity = learning_assistant.get_next_activity(user_id)
    print("\nNext Recommended Activity:")
    print(json.dumps(next_activity, indent=2))

    # 获取学习建议
    recommendations = learning_assistant.provide_learning_recommendations(user_id)
    print("\nPersonalized Learning Recommendations:")
    print(json.dumps(recommendations, indent=2))


if __name__ == "__main__":
    # 运行个性化学习助手模拟
    run_personalized_learning_assistant_simulation()