from typing import List, Dict, Any
import json
import utils
import logging
from langchain_openai import ChatOpenAI

from problem_decomposer import ComplexProblemSolvingSystem
CHINESE_INSTRUCTION =  "用中文回答。"

class ResultIntegrator:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def integrate_results(self, sub_problem_solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""
        Integrate the following sub-problem solutions（{CHINESE_INSTRUCTION}）:
        {json.dumps(sub_problem_solutions, indent=2)}

        Provide an integrated result including:
        1. Summary of key findings from each sub-problem
        2. Identification of any conflicting results or inconsistencies
        3. Synthesis of common themes or patterns
        4. Overall confidence level in the integrated result (0-100)
        5. Areas where further investigation or clarification is needed

        Return your integrated result as a compact JSON object with these sections as keys.
        """
        return json.loads(self.llm.invoke(prompt).content.strip())

class ResultValidator:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def validate_results(self, integrated_result: Dict[str, Any], original_sub_problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""
        Validate the following integrated result against the original sub-problems:

        Integrated Result:
        {json.dumps(integrated_result, indent=2)}

        Original Sub-problems:
        {json.dumps(original_sub_problems, indent=2)}

        Provide a validation report including:
        1. Consistency check: Do the integrated results align with the individual sub-problem solutions?
        2. Completeness check: Are all aspects of the original sub-problems addressed in the integrated result?
        3. Logical coherence: Is the integrated result logically sound and free of contradictions?
        4. Identification of any gaps or oversights
        5. Overall validity score(0-100)
        6. Recommendations for improving the validity of the results

        Return your validation report as a JSON object with these sections as keys，仅仅对其中value使用中文.
        """
        return json.loads(self.llm.invoke(prompt).content.strip())


class EnhancedComplexProblemSolvingSystem(ComplexProblemSolvingSystem):
    def __init__(self, llm:ChatOpenAI):
        super().__init__(llm)
        self.result_integrator = ResultIntegrator(llm)
        self.result_validator = ResultValidator(llm)

    def solve_problem(self, problem_statement: str) -> Dict[str, Any]:
        # 分解问题
        sub_problems = self.problem_decomposer.decompose_problem(problem_statement)

        # 分配和解决子问题
        solutions = []
        for sub_problem in sub_problems:
            expert = self._find_best_expert(sub_problem)
            solution = expert.solve_sub_problem(sub_problem)
            solutions.append({"sub_problem": sub_problem, "solution": solution, "expert": expert.expertise})

        # 整合结果
        integrated_result = self.result_integrator.integrate_results(solutions)

        # 验证结果
        validation_report = self.result_validator.validate_results(integrated_result, sub_problems)

        # 如果验证分数低，可以考虑重新求解或调整
        if validation_report["overall_validity_score"] < 70:
            # 这里可以实现重新求解的逻辑，例如重新分配子问题或调整求解方法
            pass

        # 生成最终的综合解决方案
        final_solution = self._generate_final_solution(problem_statement, integrated_result, validation_report)

        return {
            "problem_statement": problem_statement,
            "sub_problems": sub_problems,
            "solutions": solutions,
            "integrated_result": integrated_result,
            "validation_report": validation_report,
            "final_solution": final_solution
        }

    def _generate_final_solution(self, problem_statement: str, integrated_result: Dict[str, Any], validation_report: Dict[str, Any]) -> str:
        prompt = f"""
        Generate a final comprehensive solution for the following problem:
        "{problem_statement}"

        Based on the integrated result and validation report（{CHINESE_INSTRUCTION}）:

        Integrated Result:
        {json.dumps(integrated_result, indent=2)}

        Validation Report:
        {json.dumps(validation_report, indent=2)}

        Provide a final solution that:
        1. Addresses the original problem comprehensively
        2. Incorporates the key findings from the integrated result
        3. Takes into account the validation feedback
        4. Discusses any limitations or uncertainties
        5. Suggests next steps or areas for further research

        Return your final solution as a detailed report.
        """
        return self.llm.invoke(prompt).content.strip()

# 使用示例
def run_enhanced_complex_problem_solving_simulation():
    enhanced_problem_solving_system = EnhancedComplexProblemSolvingSystem(utils.LLM_instance)

    problem_statement = """
    Develop a sustainable and resilient urban transportation system that reduces carbon emissions,
    improves air quality, and enhances mobility for all socioeconomic groups in a rapidly growing city,
    while considering economic feasibility and technological advancements.
    """

    solution = enhanced_problem_solving_system.solve_problem(problem_statement)

    print("Enhanced Complex Problem Solving Results:")
    print(f"Problem Statement: {solution['problem_statement']}")
    print("\nSub-problems and Solutions:")
    for item in solution['solutions']:
        print(f"\nSub-problem: {item['sub_problem']['description']}")
        print(f"Solved by: {item['expert']}")
        print(f"Solution: {json.dumps(item['solution'], indent=2)}")

    print("\nIntegrated Result:")
    print(json.dumps(solution['integrated_result'], indent=2,ensure_ascii=False))

    print("\nValidation Report:")
    print(json.dumps(solution['validation_report'], indent=2,ensure_ascii=False ))

    print("\nFinal Solution:")
    print(solution['final_solution'])

if __name__ == "__main__":
    # 运行增强版复杂问题求解模拟
    run_enhanced_complex_problem_solving_simulation()