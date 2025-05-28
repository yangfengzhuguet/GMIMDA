import dgl
import torch

# 创建一个简单的图
g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]))

# 使用 Graphbolt (如果可用)
try:
    import dgl.graphbolt as gb
    print("Graphbolt is available.")
except ImportError as e:
    print(f"Graphbolt import error: {e}")

print("DGL imported successfully.")
