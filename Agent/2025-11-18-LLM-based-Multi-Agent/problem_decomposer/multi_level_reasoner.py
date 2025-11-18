from typing import List, Dict, Any
import json
import utils
import logging
from langchain_openai import ChatOpenAI

class MultiLevelReasoner:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def perform_multi_level_reasoning(self, problem_statement: str, integrated_result: Dict[str, Any], validation_report: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Perform multi-level reasoning on the following complex problem:
        "{problem_statement}"

        Based on the integrated result and validation report:

        Integrated Result:
        {json.dumps(integrated_result, indent=2)}

        Validation Report:
        {json.dumps(validation_report, indent=2)}

        Provide a multi-level reasoning analysis including:
        1. First-order implications: Direct consequences of the findings
        2. Second-order implications: Potential ripple effects or indirect consequences
        3. System-level impacts: How the solution might affect the broader system or context
        4. Temporal considerations: Short-term vs long-term effects
        5. Stakeholder analysis: How different groups might be affected by the solution
        6. Uncertainty assessment: Identification of key uncertainties and their potential impacts
        7. Trade-off analysis: Evaluation of potential trade-offs between different aspects of the solution

        Return your analysis as a JSON object with these sections as keys.
        """
        return json.loads(self.llm.invoke(prompt).content.replace('\n','').strip)

class DecisionMaker:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def make_decision(self, problem_statement: str, multi_level_reasoning: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Make a decision on the following complex problem:
        "{problem_statement}"

        Based on the multi-level reasoning analysis:
        {json.dumps(multi_level_reasoning, indent=2)}

        Provide a decision including:
        1. Recommended course of action
        2. Justification for the decision
        3. Anticipated outcomes
        4. Potential risks and mitigation strategies
        5. Implementation considerations
        6. Metrics for measuring success
        7. Contingency plans

        Return your decision as a JSON object with these sections as keys.
        """
        return json.loads(self.llm.invoke(prompt).content.replace('\n','').strip())

class AdvancedComplexProblemSolvingSystem(EnhancedComplexProblemSolvingSystem):
    def __init__(self, llm:ChatOpenAI):
        super().__init__(llm)
        self.multi_level_reasoner = MultiLevelReasoner(llm)
        self.decision_maker = DecisionMaker(llm)

    def solve_problem(self, problem_statement: str) -> Dict[str, Any]:
        # 使用增强版系统解决问题
        initial_solution = super().solve_problem(problem_statement)

        # 执行多层次推理
        multi_level_reasoning = self.multi_level_reasoner.perform_multi_level_reasoning(
            problem_statement,
            initial_solution['integrated_result'],
            initial_solution['validation_report']
        )

        # 做出决策
        decision = self.decision_maker.make_decision(problem_statement, multi_level_reasoning)

        # 生成最终报告
        final_report = self._generate_final_report(problem_statement, initial_solution, multi_level_reasoning, decision)

        return {
            **initial_solution,
            "multi_level_reasoning": multi_level_reasoning,
            "decision": decision,
            "final_report": final_report
        }

    def _generate_final_report(self, problem_statement: str, initial_solution: Dict[str, Any], multi_level_reasoning: Dict[str, Any], decision: Dict[str, Any]) -> str:
        prompt = f"""
        Generate a comprehensive final report for the following complex problem:
        "{problem_statement}"

        Incorporate the following components:
        1. Initial solution: {json.dumps(initial_solution['final_solution'], indent=2)}
        2. Multi-level reasoning: {json.dumps(multi_level_reasoning, indent=2)}
        3. Decision: {json.dumps(decision, indent=2)}

        The report should include:
        1. Executive summary
        2. Problem statement and context
        3. Methodology
        4. Key findings and insights
        5. Multi-level analysis
        6. Decision rationale
        7. Implementation plan
        8. Risk assessment and mitigation strategies
        9. Success metrics and evaluation plan
        10. Conclusions and recommendations for future work

        Provide a detailed, well-structured report that a decision-maker can act upon.
        """
        return self.llm.invoke(prompt).content.strip()

# 使用示例
def run_advanced_complex_problem_solving_simulation():
    advanced_problem_solving_system = AdvancedComplexProblemSolvingSystem(some_llm)

    problem_statement = """
    Design a comprehensive strategy to transform a mid-sized city (population 500,000) into a smart,
    sustainable, and resilient urban center over the next 20 years, addressing challenges in energy,
    transportation, waste management, and social equity while fostering economic growth and innovation.
    """

    solution = advanced_problem_solving_system.solve_problem(problem_statement)

    print("Advanced Complex Problem Solving Results:")
    print(f"Problem Statement: {solution['problem_statement']}")
    print("\nMulti-level Reasoning:")
    print(json.dumps(solution['multi_level_reasoning'], indent=2))
    print("\nDecision:")
    print(json.dumps(solution['decision'], indent=2))
    print("\nFinal Report:")
    print(solution['final_report'])


if __name__ == "__main__":
    # 运行高级复杂问题求解模拟
    run_advanced_complex_problem_solving_simulation()