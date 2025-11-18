from typing import List, Dict, Any
import json
import utils
import logging
from langchain_openai import ChatOpenAI

CHINESE_INSTRUCTION =  "用中文回答。"

class ProblemDecomposer:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def decompose_problem(self, problem_statement: str) -> List[Dict[str, Any]]:
        prompt = f"""
        Decompose the following complex problem into smaller, manageable sub-problems（{CHINESE_INSTRUCTION}）:
        "{problem_statement}"

        For each sub-problem, provide:
        1. A clear description of the sub-problem
        2. The domain or expertise required to solve it
        3. Estimated complexity (low, medium, high)
        4. Dependencies on other sub-problems, if any

        Return the decomposition as a JSON array of objects, each representing a sub-problem（仅将json中的value使用中文，key使用英文）.
        """
        return json.loads(self.llm.invoke(prompt).content.strip())

class ExpertAgent:
    def __init__(self, llm:ChatOpenAI, expertise: str, skills: List[str]):
        self.llm = llm
        self.expertise = expertise
        self.skills = skills

    def solve_sub_problem(self, sub_problem: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        As an expert in {self.expertise} with skills in {', '.join(self.skills)},
        solve the following sub-problem（{CHINESE_INSTRUCTION}）:

        {json.dumps(sub_problem, indent=2)}

        Provide a solution including:
        1. Your approach to solving the problem
        2. Key findings or results
        3. Any assumptions made
        4. Confidence level in your solution (0-100)
        5. Potential implications or next steps

        Return your solution as a JSON object with these sections as keys.
        """
        return json.loads(self.llm.invoke(prompt).content.strip())

class ComplexProblemSolvingSystem:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm
        self.problem_decomposer = ProblemDecomposer(llm)
        self.expert_agents = [
            ExpertAgent(llm, "数据科学", ["机器学习", "统计分析", "数据可视化"]),
            ExpertAgent(llm, "环境科学", ["气候建模", "生态系统分析", "可持续性"]),
            ExpertAgent(llm, "经济学", ["计量经济学", "政策分析", "市场动态"]),
            ExpertAgent(llm, "工程学", ["系统设计", "优化", "风险评估"]),
            ExpertAgent(llm, "公共卫生", ["流行病学", "医疗保健系统", "公共政策"])
        ]

    def solve_problem(self, problem_statement: str) -> Dict[str, Any]:
        # 分解问题
        sub_problems = self.problem_decomposer.decompose_problem(problem_statement)

        print(f"decompose as follows:\n{json.dumps(sub_problems, indent=2,ensure_ascii=False)}\n")

        # 分配和解决子问题
        solutions = []
        for sub_problem in sub_problems:
            expert = self._find_best_expert(sub_problem)
            solution = expert.solve_sub_problem(sub_problem)
            solutions.append({"sub_problem": sub_problem, "solution": solution, "expert": expert.expertise})

        # 整合解决方案
        integrated_solution = self._integrate_solutions(problem_statement, solutions)

        return {
            "problem_statement": problem_statement,
            "sub_problems": sub_problems,
            "solutions": solutions,
            "integrated_solution": integrated_solution
        }

    def _find_best_expert(self, sub_problem: Dict[str, Any]) -> ExpertAgent:
        # 简单实现：根据领域匹配专家
        required_expertise = sub_problem["domain"]
        for agent in self.expert_agents:
            if agent.expertise.lower() in required_expertise.lower():
                return agent
        return self.expert_agents[0]  # 默认返回第一个专家

    def _integrate_solutions(self, problem_statement: str, solutions: List[Dict[str, Any]]) -> str:
        prompt = f"""
        Integrate the solutions to the following complex problem:
        "{problem_statement}"

        Sub-problem solutions:
        {json.dumps(solutions, indent=2)}

        Provide an integrated solution that:
        1. Synthesizes the insights from all sub-problems
        2. Addresses the original problem comprehensively
        3. Highlights key findings and their implications
        4. Identifies any remaining challenges or areas for further investigation

        Return your integrated solution as a cohesive report.
        """
        return self.llm.invoke(prompt).content.strip()

# 使用示例
def run_complex_problem_solving_simulation():
    problem_solving_system = ComplexProblemSolvingSystem(utils.LLM_instance)

    problem_statement = """
    Develop a comprehensive strategy to mitigate the impact of climate change on urban food security,
    considering environmental, economic, and public health factors in a rapidly growing metropolitan area.
    """

    solution = problem_solving_system.solve_problem(problem_statement)

    print("Complex Problem Solving Results:")
    print(f"Problem Statement: {solution['problem_statement']}")
    print("\nSub-problems and Solutions:")
    for item in solution['solutions']:
        print(f"\nSub-problem: {item['sub_problem']['description']}")
        print(f"Solved by: {item['expert']} 专家")
        print(f"Solution: {json.dumps(item['solution'], indent=2,ensure_ascii=False)}")

    print("\nIntegrated Solution:")
    print(solution['integrated_solution'])


if __name__ == "__main__":
    # 运行复杂问题求解模拟
    run_complex_problem_solving_simulation()
    # main()