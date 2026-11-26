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

# DOC: develop/build-custom-ops.mdx

from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.functional import custom_op
from max.graph import DeviceRef, Graph, TensorType
from max.tensor import Tensor

mojo_kernels = Path(__file__).parent / "kernels"

# Both inputs and the output are vectors of the same length, n.
(n,) = custom_op.Symbols("n")
vector_addition = custom_op.declare(
    "vector_addition",
    inputs={
        "lhs": custom_op.TemplateType(DType.float32, [n]),
        "rhs": custom_op.TemplateType(DType.float32, [n]),
    },
    outputs=[custom_op.TemplateType(DType.float32, [n])],
    custom_extensions=[mojo_kernels],
)


def check(name: str, result: np.ndarray, expected: np.ndarray) -> None:
    # A declaration that understates the output shape truncates the result
    # rather than raising, so compare the shape as well as the values.
    assert result.shape == expected.shape, (name, result.shape)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    print(f"{name} result:")
    print(result)
    print()


if __name__ == "__main__":
    # Place the op on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    lhs_values = np.random.uniform(size=10).astype(np.float32)
    rhs_values = np.random.uniform(size=10).astype(np.float32)
    expected = lhs_values + rhs_values

    # Eager: call the op on tensors, and it runs now.
    lhs = Tensor.from_dlpack(lhs_values).to(device)
    rhs = Tensor.from_dlpack(rhs_values).to(device)
    result = vector_addition(lhs, rhs)
    assert isinstance(result, Tensor)
    check("Eager", result.to(CPU()).to_numpy(), expected)

    # Graph: call the same op on graph values, and it adds a node to the graph.
    device_ref = DeviceRef.from_device(device)
    vector_type = TensorType(DType.float32, ["n"], device=device_ref)
    graph = Graph(
        "declared_vector_addition",
        forward=vector_addition,
        input_types=[vector_type, vector_type],
    )

    session = InferenceSession(devices=[device])
    model = session.load(graph)
    output = model.execute(
        Buffer.from_numpy(lhs_values).to(device),
        Buffer.from_numpy(rhs_values).to(device),
    )[0]
    assert isinstance(output, Buffer)
    check("Graph", output.to(CPU()).to_numpy(), expected)
