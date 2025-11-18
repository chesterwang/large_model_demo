from typing import List, Dict, Any
import json
import utils
import logging
from langchain_openai import ChatOpenAI
from difflib import unified_diff
from datetime import datetime

class VersionControlSystem:
    def __init__(self):
        self.versions = []
        self.current_version = 0

    def create_version(self, content: str, author: str) -> int:
        self.versions.append({
            "version": len(self.versions),
            "content": content,
            "author": author,
            "timestamp": datetime.now().isoformat()
        })
        return len(self.versions) - 1

    def get_version(self, version: int) -> Dict[str, Any]:
        return self.versions[version]

    def get_diff(self, version1: int, version2: int) -> List[str]:
        content1 = self.versions[version1]["content"].splitlines()
        content2 = self.versions[version2]["content"].splitlines()
        return list(unified_diff(content1, content2, lineterm=''))

class ConflictResolver:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm

    def resolve_conflict(self, base_version: Dict[str, Any], version1: Dict[str, Any], version2: Dict[str, Any]) -> str:
        prompt = f"""
        Resolve the conflict between two versions of a document:

        Base Version:
        {base_version['content']}

        Version 1 (by {version1['author']}):
        {version1['content']}

        Version 2 (by {version2['author']}):
        {version2['content']}

        Merge these versions into a single coherent document, preserving the best elements of both changes.
        Ensure the merged version is consistent and logical.
        """
        return self.llm.invoke(prompt).content.strip()

from creative_agent import CollaborativeCreationSystem

class CollaborativeWritingSystem(CollaborativeCreationSystem):
    def __init__(self, llm:ChatOpenAI):
        super().__init__(llm)
        self.version_control = VersionControlSystem()
        self.conflict_resolver = ConflictResolver(llm)

    def create_document(self, content: str, author: str) -> int:
        return self.version_control.create_version(content, author)

    def edit_document(self, version: int, new_content: str, author: str) -> int:
        base_version = self.version_control.get_version(version)
        if base_version["author"] != author:
            # 检查是否有冲突
            latest_version = self.version_control.get_version(self.version_control.current_version)
            if latest_version["content"] != base_version["content"]:
                resolved_content = self.conflict_resolver.resolve_conflict(
                    base_version, latest_version, {"content": new_content, "author": author}
                )
                return self.version_control.create_version(resolved_content, f"Merged: {latest_version['author']}, {author}")
        return self.version_control.create_version(new_content, author)

    def get_document_history(self) -> List[Dict[str, Any]]:
        return self.version_control.versions

    def compare_versions(self, version1: int, version2: int) -> List[str]:
        return self.version_control.get_diff(version1, version2)

# 使用示例
def run_collaborative_writing_simulation():
    collab_writing_system = CollaborativeWritingSystem(utils.LLM_instance)

    # 创建初始文档
    initial_content = "Once upon a time, in a world where cats ruled the universe..."
    doc_version = collab_writing_system.create_document(initial_content, "Writer")
    print(f"Initial document created. Version: {doc_version}")

    # 编辑文档
    edit1 = "Once upon a time, in a world where cats ruled the universe with wisdom and grace..."
    new_version1 = collab_writing_system.edit_document(doc_version, edit1, "Editor")
    print(f"Document edited by Editor. New version: {new_version1}")

    # 另一个作者同时编辑
    edit2 = "In a universe governed by feline overlords, there once was a tale..."
    new_version2 = collab_writing_system.edit_document(doc_version, edit2, "Writer")
    print(f"Document edited by Writer. New version: {new_version2}")

    # 查看文档历史
    history = collab_writing_system.get_document_history()
    print("\nDocument History:")
    for version in history:
        print(f"Version {version['version']} by {version['author']} at {version['timestamp']}")

    # 比较版本
    diff = collab_writing_system.compare_versions(new_version1, new_version2)
    print("\nDiff between latest versions:")
    for line in diff:
        print(line)

if __name__ == "__main__":
    run_collaborative_writing_simulation()