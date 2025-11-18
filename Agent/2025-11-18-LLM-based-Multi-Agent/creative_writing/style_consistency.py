from typing import List, Dict, Any
import json
import utils
import logging
from langchain_openai import ChatOpenAI

CHINESE_INSTRUCTION =  "用中文回答。"

class StyleAnalyzer:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def analyze_style(self, text: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the writing style of the following text（{CHINESE_INSTRUCTION}）:
        "{text}"

        Provide an analysis including:
        1. Tone (formal, informal, humorous, serious, etc.)
        2. Vocabulary level (simple, advanced, technical, etc.)
        3. Sentence structure (simple, complex, varied, etc.)
        4. Literary devices used (metaphors, similes, alliteration, etc.)
        5. Overall mood or atmosphere

        Return your analysis as a JSON object with these aspects as keys.
        """
        return json.loads(self.llm.invoke(prompt).content.strip())

class StyleConsistencyEnforcer:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm
        self.style_analyzer = StyleAnalyzer(llm)

    def enforce_consistency(self, original_text: str, new_text: str) -> str:
        original_style = self.style_analyzer.analyze_style(original_text)
        prompt = f"""
        Rewrite the following text to match the style of the original text:

        Original text style:
        {json.dumps(original_style, indent=2)}

        Text to rewrite:
        "{new_text}"

        Ensure the rewritten text maintains the same content and meaning while adopting the style of the original text（{CHINESE_INSTRUCTION}）.
        """
        return self.llm.invoke(prompt).content.strip()

from version_control import CollaborativeWritingSystem

class StyleConsistentCollaborativeWritingSystem(CollaborativeWritingSystem):
    def __init__(self, llm:ChatOpenAI):
        super().__init__(llm)
        self.style_enforcer = StyleConsistencyEnforcer(llm)

    def edit_document(self, version: int, new_content: str, author: str) -> int:
        base_version = self.version_control.get_version(version)
        consistent_content = self.style_enforcer.enforce_consistency(base_version["content"], new_content)
        return super().edit_document(version, consistent_content, author)

    def analyze_style_consistency(self, version1: int, version2: int) -> Dict[str, Any]:
        content1 = self.version_control.get_version(version1)["content"]
        content2 = self.version_control.get_version(version2)["content"]
        style1 = self.style_enforcer.style_analyzer.analyze_style(content1)
        style2 = self.style_enforcer.style_analyzer.analyze_style(content2)

        prompt = f"""
        Compare the writing styles of two versions of a document:

        Version 1 style:
        {json.dumps(style1, indent=2)}

        Version 2 style:
        {json.dumps(style2, indent=2)}

        Provide an analysis of style consistency including({CHINESE_INSTRUCTION}):
        1. Areas of strong consistency
        2. Notable differences in style
        3. Overall consistency score (0-100)
        4. Recommendations for improving consistency

        Return your analysis as a JSON object with these sections as keys.
        """
        return json.loads(self.llm.invoke(prompt).content.strip())

# 使用示例
def run_style_consistent_collaborative_writing_simulation():
    style_consistent_system = StyleConsistentCollaborativeWritingSystem(utils.LLM_instance)

    # 创建初始文档
    initial_content = "In the shadowy recesses of the old mansion, whispers of forgotten secrets lingered like cobwebs in the corners."
    doc_version = style_consistent_system.create_document(initial_content, "Writer")
    print(f"Initial document created. Version: {doc_version}")

    # 编辑文档
    edit1 = "The ancient house held many mysteries, its rooms filled with echoes of the past."
    new_version1 = style_consistent_system.edit_document(doc_version, edit1, "Editor")
    print(f"Document edited by Editor. New version: {new_version1}")

    # 另一个作者编辑
    edit2 = "Secrets and mysteries were hidden in every nook and cranny of the old building."
    new_version2 = style_consistent_system.edit_document(new_version1, edit2, "Writer")
    print(f"Document edited by Writer. New version: {new_version2}")

    # 分析风格一致性
    consistency_analysis = style_consistent_system.analyze_style_consistency(doc_version, new_version2)
    print("\nStyle Consistency Analysis:")
    print(json.dumps(consistency_analysis, indent=2,ensure_ascii=False))


if __name__ == "__main__":
    # 运行风格一致的协作写作模拟
    run_style_consistent_collaborative_writing_simulation()