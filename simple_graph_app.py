import sys

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

threads = int(sys.argv[1]) if len(sys.argv) > 1 else None
session = InferenceSession(num_threads=threads)
print("Session created", flush=True)

with Graph(
    "add_one",
    input_types=[TensorType(DType.float32, [1], device=DeviceRef.CPU())],
) as graph:
    graph.output(graph.inputs[0].tensor + 1.0)

model = session.load(graph)
result = model.execute(Buffer.from_numpy(np.array([1.0], dtype=np.float32)))
print(result[0].to_numpy())
