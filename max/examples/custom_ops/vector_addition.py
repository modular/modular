# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

mojo_kernels = Path(__file__).parent / "kernels"


def build_graph(op_name: str, vector_width: int, device: Device) -> Graph:
    """Builds a one-op graph that adds two vectors with the named custom op."""
    vector_type = TensorType(
        DType.float32,
        shape=[vector_width],
        device=DeviceRef.from_device(device),
    )
    with Graph(
        op_name,
        input_types=[vector_type, vector_type],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        lhs, rhs = graph.inputs
        output = ops.custom(
            name=op_name,
            device=DeviceRef.from_device(device),
            values=[lhs, rhs],
            out_types=[vector_type],
        )[0].tensor
        graph.output(output)
    return graph


if __name__ == "__main__":
    vector_width = 10

    # Run on the CPU, and on a GPU if one is available, so that each
    # platform-specific kernel of `vector_addition_by_platform` runs.
    devices = [CPU()] + ([Accelerator()] if accelerator_count() > 0 else [])

    # Fill input vectors with random values.
    lhs_values = np.random.uniform(size=(vector_width)).astype(np.float32)
    rhs_values = np.random.uniform(size=(vector_width)).astype(np.float32)
    expected = lhs_values + rhs_values

    print("Left-hand-side values:")
    print(lhs_values)
    print()

    print("Right-hand-side values:")
    print(rhs_values)
    print()

    print("Expected result:")
    print(expected)
    print()

    for device in devices:
        # Set up an inference session for running graphs on this device.
        session = InferenceSession(devices=[device])

        # `vector_addition` branches on the device inside one kernel, and
        # `vector_addition_by_platform` registers one kernel per platform.
        for op_name in ("vector_addition", "vector_addition_by_platform"):
            # Compile the graph.
            graph = build_graph(op_name, vector_width, device)
            model = session.init(session.compile(graph))

            # Create driver tensors from the inputs, and move them to the
            # device.
            lhs_tensor = Buffer.from_numpy(lhs_values).to(device)
            rhs_tensor = Buffer.from_numpy(rhs_values).to(device)

            # Perform the calculation on the target device.
            result = model.execute(lhs_tensor, rhs_tensor)[0]

            # Copy values back to the CPU to be read.
            assert isinstance(result, Buffer)
            result_values = result.to(CPU()).to_numpy()
            np.testing.assert_allclose(result_values, expected, rtol=1e-6)

            print(f"{op_name} result on {device.label}:")
            print(result_values)
            print()
