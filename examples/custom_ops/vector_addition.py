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

import os
from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops

if __name__ == "__main__":
    # This is necessary only in specific build environments.
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    path = Path(__file__).parent / "kernels.mojopkg"

    vector_width = 10
    dtype = DType.float32

    # Configure our simple one-operation graph.
    with Graph(
        "vector_addition",
        input_types=[
            TensorType(dtype, shape=[vector_width]),
            TensorType(dtype, shape=[vector_width]),
        ],
        custom_extensions=[path],
    ) as graph:
        # Take in the two inputs to the graph.
        lhs, rhs = graph.inputs
        output = ops.custom(
            name="vector_addition",
            values=[lhs, rhs],
            out_types=[
                TensorType(dtype=lhs.tensor.dtype, shape=lhs.tensor.shape)
            ],
        )[0].tensor
        graph.output(output)

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Set up an inference session for running the graph.
    session = InferenceSession(
        devices=[device],
    )

    # Compile the graph.
    model = session.load(graph)

    # Fill input matrices with random values.
    lhs_values = np.random.uniform(size=(vector_width)).astype(np.float32)
    rhs_values = np.random.uniform(size=(vector_width)).astype(np.float32)

    # Create driver tensors from this, and move them to the accelerator.
    lhs_tensor = Tensor.from_numpy(lhs_values).to(device)
    rhs_tensor = Tensor.from_numpy(rhs_values).to(device)

    # Perform the calculation on the target device.
    result = model.execute(lhs_tensor, rhs_tensor)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    result = result.to(CPU())

    print("Left-hand-side values:")
    print(lhs_values)
    print()

    print("Right-hand-side values:")
    print(rhs_values)
    print()

    print("Graph result:")
    print(result.to_numpy())
    print()

    print("Expected result:")
    print(lhs_values + rhs_values)
