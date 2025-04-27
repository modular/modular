# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from typing import Any

import numpy as np
from max import engine
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def add_tensors(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    # 1. Build the graph
    input_type = TensorType(
        dtype=DType.float32, shape=(1,), device=DeviceRef.CPU()
    )
    with Graph(
        "simple_add_graph", input_types=(input_type, input_type)
    ) as graph:
        lhs, rhs = graph.inputs
        out = ops.add(lhs, rhs)
        graph.output(out)
        print("final graph:", graph)

    # 2. Create an inference session
    session = engine.InferenceSession()
    model = session.load(graph)

    for tensor in model.input_metadata:
        print(
            f"name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}"
        )

    # 3. Execute the graph
    ret = model.execute(a, b)[0]
    print("result:", ret)
    return ret


if __name__ == "__main__":
    input0 = np.array([1.0], dtype=np.float32)
    input1 = np.array([1.0], dtype=np.float32)
    add_tensors(input0, input1)
