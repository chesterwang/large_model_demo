import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict

class DecisionTreeVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_decision_node(self, node_id: str, label: str, parent_id: str = None):
        self.graph.add_node(node_id, label=label)
        if parent_id:
            self.graph.add_edge(parent_id, node_id)

    def visualize(self, filename: str):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=False, node_color='lightblue', node_size=1000, arrows=True)
        
        labels = nx.get_node_attributes(self.graph, 'label')
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.title("Agent Decision Tree")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

class PerformanceDashboard:
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {}

    def update_metrics(self, metric_name: str, value: float):
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        self.metrics_history[metric_name].append(value)

    def visualize(self, filename: str):
        plt.figure(figsize=(12, 6))
        for metric, values in self.metrics_history.items():
            plt.plot(values, label=metric)
        
        plt.title("Agent Performance Over Time")
        plt.xlabel("Time")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

# 使用示例
# 决策树可视化
tree_visualizer = DecisionTreeVisualizer()
tree_visualizer.add_decision_node("A", "Initial State")
tree_visualizer.add_decision_node("B", "Action 1", "A")
tree_visualizer.add_decision_node("C", "Action 2", "A")
tree_visualizer.add_decision_node("D", "Outcome 1", "B")
tree_visualizer.add_decision_node("E", "Outcome 2", "B")
tree_visualizer.add_decision_node("F", "Outcome 3", "C")
tree_visualizer.visualize("decision_tree.png")

# 性能仪表板
dashboard = PerformanceDashboard()
for i in range(10):
    dashboard.update_metrics("Accuracy", 0.8 + 0.02 * i)
    dashboard.update_metrics("Response Time", 1.0 - 0.05 * i)
dashboard.visualize("performance_dashboard.png")

print("Visualizations generated: decision_tree.png and performance_dashboard.png")