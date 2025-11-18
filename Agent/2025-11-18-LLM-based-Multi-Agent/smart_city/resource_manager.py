from typing import List, Dict, Any
import json
from datetime import datetime
import utils
import logging
from langchain_openai import ChatOpenAI

class EmergencyEvent:
    def __init__(self, event_type: str, location: str, severity: int, description: str):
        self.event_type = event_type
        self.location = location
        self.severity = severity
        self.description = description
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "location": self.location,
            "severity": self.severity,
            "description": self.description,
            "timestamp": self.timestamp
        }

class EmergencyResponsePlanner:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def generate_response_plan(self, event: EmergencyEvent, city_status: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Generate an emergency response plan for the following event:
        {json.dumps(event.to_dict())}

        Consider the current city status:
        {json.dumps(city_status)}

        Provide a response plan that includes:
        1. Immediate actions to take
        2. Resources to be mobilized
        3. Departments to be involved
        4. Communication strategy
        5. Evacuation plans (if necessary)
        6. Estimated timeline for resolution
        7. Potential risks and mitigation strategies

        Return your response plan as a JSON object with these sections as keys.
        """

        import time
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            response = self.llm.invoke(prompt)
            text = response.content.replace('\n','').replace('```json','').replace('```','').strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logging.warning(f"EmergencyResponsePlanner: JSON decode failed (attempt {attempt}/{max_attempts})")
                if attempt == max_attempts:
                    logging.error(f"EmergencyResponsePlanner: failed to decode JSON after {max_attempts} attempts")
                    return {"error": "json_decode_error", "raw": text}
                time.sleep(1)

class ResourceManager:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm
        self.available_resources = {
            "emergency_vehicles": 50,"police_officers": 200,
            "firefighters": 150,
            "paramedics": 100,
            "hospital_beds": 500,
            "shelters": 10,
            "water_pumps": 20,
            "generators": 30,
            "sandbags": 10000
        }

    def allocate_resources(self, response_plan: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Allocate resources for the following emergency response plan:
        {json.dumps(response_plan, indent=2)}

        Available resources:
        {json.dumps(self.available_resources, indent=2)}

        Provide a resource allocation plan that:
        1. Assigns specific resources to each action in the response plan
        2. Ensures efficient use of available resources
        3. Identifies any resource shortages
        4. Suggests alternatives for any insufficient resources
        5. Prioritizes resource allocation based on criticality

        Return your allocation plan as a JSON object for the emergency response plan, mapping each action to its allocated resources, only listing requested resource.
        """
        response = self.llm.invoke(prompt)
        allocation_plan = json.loads(response.content.replace('\n','').replace('```json','').replace('```','').strip())

        # Update available resources
        for action, resources in allocation_plan.items():
            for resource, amount in resources.items():
                if resource in self.available_resources:
                    self.available_resources[resource] -= amount

        return allocation_plan

from collaborative_dms import SmartCityCollaborativeManagementSystem

class SmartCityEmergencyManagementSystem(SmartCityCollaborativeManagementSystem):
    def __init__(self, llm:ChatOpenAI):
        super().__init__(llm)
        self.emergency_response_planner = EmergencyResponsePlanner(llm)
        self.resource_manager = ResourceManager(llm)

    def handle_emergency(self, event: EmergencyEvent) -> Dict[str, Any]:
        city_status = self.get_city_status_report()['integrated_data']
        response_plan = self.emergency_response_planner.generate_response_plan(event, city_status)
        resource_allocation = self.resource_manager.allocate_resources(response_plan)

        return {
            "event": event.to_dict(),
            "response_plan": response_plan,
            "resource_allocation": resource_allocation
        }

    def simulate_emergency_scenario(self) -> Dict[str, Any]:
        # Simulate a major emergency event
        event = EmergencyEvent(
            event_type="Flash Flood",
            location="Downtown Area",
            severity=9,
            description="Sudden heavy rainfall causing rapid flooding in the city center. Multiple streets submerged, vehicles stranded, and buildings at risk."
        )

        emergency_response = self.handle_emergency(event)

        # Generate post-emergency city status
        post_emergency_status = self.get_city_status_report()

        # Develop recovery plan
        recovery_plan = self.generate_recovery_plan(emergency_response, post_emergency_status)

        return {
            "emergency_response": emergency_response,
            "post_emergency_status": post_emergency_status,
            "recovery_plan": recovery_plan
        }

    def generate_recovery_plan(self, emergency_response: Dict[str, Any], post_emergency_status: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Generate a recovery plan based on the following emergency response and post-emergency city status:

        Emergency Response:
        {json.dumps(emergency_response, indent=2)}

        Post-Emergency City Status:
        {json.dumps(post_emergency_status, indent=2)}

        Provide a recovery plan that includes:
        1. Immediate restoration actions
        2. Medium-term recovery strategies
        3. Long-term resilience improvements
        4. Resource requirements for recovery
        5. Timeline for different recovery phases
        6. Departments involved and their responsibilities
        7. Community engagement and support initiatives

        Return your recovery plan as a JSON object with these sections as keys.
        """
        response = self.llm.invoke(prompt)
        return json.loads(response.content.replace('\n','').replace('```json','').replace('```','').strip())

# 使用示例
def run_smart_city_emergency_management_simulation():
    smart_city_system = SmartCityEmergencyManagementSystem(utils.LLM_instance)

    emergency_scenario_results = smart_city_system.simulate_emergency_scenario()

    print("Smart City Emergency Management Simulation Results:")
    print(json.dumps(emergency_scenario_results, indent=2,ensure_ascii=False))

if __name__ == "__main__":
    # 运行智能城市应急管理模拟
    run_smart_city_emergency_management_simulation()