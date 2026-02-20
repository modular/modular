# DOC: max/develop/index.mdx

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops

# Step 1: Define the graph structure
cpu = CPU()
input_type = TensorType(DType.float32, shape=[5], device=cpu)

with Graph("relu_graph", input_types=[input_type]) as graph:
    x = graph.inputs[0]
    y = ops.relu(x)
    graph.output(y)

# Step 2: Compile the graph
session = InferenceSession(devices=[cpu])
model = session.load(graph)

# Step 3: Execute with data
input_data = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
result = model.execute(input_data)
print(np.from_dlpack(result[0]))
