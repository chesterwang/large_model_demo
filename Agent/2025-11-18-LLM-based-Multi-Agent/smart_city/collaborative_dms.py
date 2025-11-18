from typing import List, Dict, Any
import json
import utils
import logging
import time
from langchain_openai import ChatOpenAI

class CityDepartment:
    def __init__(self, llm:ChatOpenAI, name: str, responsibilities: List[str]):
        self.llm = llm
        self.name = name
        self.responsibilities = responsibilities

    def generate_action_plan(self, city_status: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        As the {self.name} department with responsibilities in {', '.join(self.responsibilities)},
        generate an action plan based on the following city status:
        {json.dumps(city_status)}

        Provide an action plan that:
        1. Addresses issues relevant to your department
        2. Proposes specific actions to take
        3. Estimates resource requirements
        4. Identifies potential challenges
        5. Suggests collaboration points with other departments

        Return your action plan as a JSON object with these sections as keys.
        """
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            response = self.llm.invoke(prompt)
            text = response.content.replace('\n','').replace('```json','').replace('```','').strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logging.warning(f"{self.name} department: JSON decode failed (attempt {attempt}/{max_attempts})")
                if attempt == max_attempts:
                    logging.error(f"{self.name} department: failed to decode JSON after {max_attempts} attempts")
                    return {"error": "json_decode_error", "raw": text}
                time.sleep(1)

class CollaborativeDecisionMaker:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm
        self.departments = [
            CityDepartment(llm, "Transportation", ["traffic management", "public transport"]),
            CityDepartment(llm, "Energy", ["power distribution", "renewable energy"]),
            CityDepartment(llm, "Environment", ["air quality", "waste management"]),
            CityDepartment(llm, "Public Safety", ["emergency services", "crime prevention"]),
            CityDepartment(llm, "Urban Planning", ["infrastructure development", "zoning"])
        ]

    def make_collaborative_decision(self, city_status: Dict[str, Any]) -> Dict[str, Any]:
        department_plans = [dept.generate_action_plan(city_status) for dept in self.departments]

        prompt = f"""
        Analyze the following department action plans:
        {json.dumps(department_plans)}

        Create a collaborative decision that:
        1. Integrates actions from all departments
        2. Resolves any conflicts between department plans
        3. Identifies synergies and opportunities for collaboration
        4. Prioritizes actions based on urgency and impact
        5. Provides a timeline for implementation
        6. Suggests a resource allocation strategy

        Return your collaborative decision as a JSON object with these sections as keys.
        """
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            response = self.llm.invoke(prompt)
            text = response.content.replace('\n','').replace('```json','').replace('```','').strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logging.warning(f"CollaborativeDecisionMaker: JSON decode failed (attempt {attempt}/{max_attempts})")
                if attempt == max_attempts:
                    logging.error(f"CollaborativeDecisionMaker: failed to decode JSON after {max_attempts} attempts")
                    return {"error": "json_decode_error", "raw": text}
                time.sleep(1)


from smart_city_dms import SmartCityDataManagementSystem

class SmartCityCollaborativeManagementSystem(SmartCityDataManagementSystem):
    def __init__(self, llm:ChatOpenAI):
        super().__init__(llm)
        self.collaborative_decision_maker = CollaborativeDecisionMaker(llm)

    def generate_city_management_plan(self) -> Dict[str, Any]:
        city_status_report = self.get_city_status_report()
        collaborative_decision = self.collaborative_decision_maker.make_collaborative_decision(city_status_report['integrated_data'])

        return {
            "city_status": city_status_report,
            "collaborative_decision": collaborative_decision
        }

# 使用示例
def run_smart_city_collaborative_management_simulation():
    smart_city_system = SmartCityCollaborativeManagementSystem(utils.LLM_instance)

    city_management_plan = smart_city_system.generate_city_management_plan()

    print("Smart City Management Plan:")
    print(json.dumps(city_management_plan, indent=2,ensure_ascii=False))


if __name__ == "__main__":
    # 运行智能城市协作管理模拟
    run_smart_city_collaborative_management_simulation()