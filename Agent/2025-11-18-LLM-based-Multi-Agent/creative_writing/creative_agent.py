from typing import List, Dict, Any
import json
import utils
import logging
from langchain_openai import ChatOpenAI


CHINESE_INSTRUCTION =  "用中文生成。"

class CreativeAgent:
    def __init__(self, llm:ChatOpenAI, role: str, skills: List[str]):
        self.llm = llm
        self.role = role
        self.skills = skills

    def generate_ideas(self, prompt: str) -> List[str]:
        llm_prompt = f"""
        As a {self.role} with skills in {', '.join(self.skills)},
        generate 5 creative ideas based on the following prompt（{CHINESE_INSTRUCTION}）:
        "{prompt}"

        Return the ideas as a JSON array of strings.
        """
        return json.loads(self.llm.invoke(llm_prompt).content.strip())

    def elaborate_idea(self, idea: str) -> str:
        llm_prompt = f"""
        As a {self.role} with skills in {', '.join(self.skills)},
        elaborate on the following idea:
        "{idea}"

        Provide a detailed explanation and potential implementation steps（{CHINESE_INSTRUCTION}）.
        """
        return self.llm.invoke(llm_prompt).content.strip()

    def critique_idea(self, idea: str) -> Dict[str, Any]:
        llm_prompt = f"""
        As a {self.role} with skills in {', '.join(self.skills)},
        critique the following idea（{CHINESE_INSTRUCTION}）:
        "{idea}"

        Provide a critique including:
        1. Strengths
        2. Weaknesses
        3. Potential improvements
        4. Overall feasibility (score from 1 to 10)

        Return your critique as a JSON object with these sections as keys.
        """
        return json.loads(self.llm.invoke(llm_prompt).content.strip())


ROLE_AND_SKILLS = [
    ("Writer", ["storytelling", "character development", "dialogue"]),
    ("Editor", ["grammar", "style", "structure"]),
    ("Visual Artist", ["illustration", "color theory", "composition"]),
    ("Marketing Specialist", ["branding", "audience engagement", "trend analysis"])
]


class CollaborativeCreationSystem:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm
        self.agents = [ CreativeAgent(llm, role, skills) for role, skills in ROLE_AND_SKILLS]

    def brainstorm(self, prompt: str) -> List[Dict[str, Any]]:
        all_ideas = []
        for agent in self.agents:
            ideas = agent.generate_ideas(prompt)
            all_ideas.extend([{"idea": idea, "role": agent.role} for idea in ideas])
        return all_ideas

    def develop_concept(self, idea: str) -> Dict[str, Any]:
        elaborations = {}
        critiques = {}
        for agent in self.agents:
            elaborations[agent.role] = agent.elaborate_idea(idea)
            critiques[agent.role] = agent.critique_idea(idea)
        
        return {
            "original_idea": idea,
            "elaborations": elaborations,
            "critiques": critiques
        }

    def synthesize_feedback(self, concept: Dict[str, Any]) -> str:
        prompt = f"""
        Synthesize the following concept development feedback:
        {json.dumps(concept, indent=2)}

        Provide a summary that includes:
        1. Key points from each role's elaboration
        2. Common themes in the critiques
        3. Overall feasibility assessment
        4. Suggested next steps for further development

        Return your synthesis as a cohesive paragraph（{CHINESE_INSTRUCTION}）.
        """
        return self.llm.invoke(prompt).content.strip()

# 使用示例
def main():

    collab_system = CollaborativeCreationSystem(utils.LLM_instance)

    # 头脑风暴阶段
    prompt = "Create a children's book series about a time-traveling scientist cat"
    ideas = collab_system.brainstorm(prompt)
    print("Brainstorming Results:")
    for idea in ideas:
        print(f"{idea['role']}: {idea['idea']}")

    # 选择一个想法进行深入开发
    selected_idea = ideas[0]['idea']  # 为简单起见，选择第一个想法
    print(f"\nDeveloping Concept: {selected_idea}")
    concept = collab_system.develop_concept(selected_idea)

    # 综合反馈
    synthesis = collab_system.synthesize_feedback(concept)
    print("\nFeedback Synthesis:")
    print(synthesis)


if __name__ == "__main__":
    # run_collaborative_creation_simulation()
    main()
