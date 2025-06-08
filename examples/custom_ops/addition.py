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

from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

if __name__ == "__main__":
    mojo_kernels = Path(__file__).parent / "kernels"

    rows = 5
    columns = 10
    dtype = DType.float32
    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()
    # Configure our simple one-operation graph.
    graph = Graph(
        "addition",
        # The custom Mojo operation is referenced by its string name, and we
        # need to provide inputs as a list as well as expected output types.
        forward=lambda x: ops.custom(
            name="add_one",
            device=DeviceRef.from_device(device),
            values=[x],
            out_types=[
                TensorType(
                    dtype=x.dtype,
                    shape=x.tensor.shape,
                    device=DeviceRef.from_device(device),
                )
            ],
        )[0].tensor,
        input_types=[
            TensorType(
                dtype,
                shape=[rows, columns],
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    )

    # Set up an inference session for running the graph.
    session = InferenceSession(
        devices=[device],
    )

    # Compile the graph.
    model = session.load(graph)

    # Fill an input matrix with random values.
    x_values = np.random.uniform(size=(rows, columns)).astype(np.float32)

    # Create a tensor and move it to the device (CPU or GPU).
    x = Tensor.from_numpy(x_values).to(device)

    # Run inference with the input tensor.
    result = model.execute(x)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    result = result.to(CPU())

    print("Graph result:")
    print(result.to_numpy())
    print()

    print("Expected result:")
    print(x_values + 1)
