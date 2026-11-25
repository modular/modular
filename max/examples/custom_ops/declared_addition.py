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

# The input and the output are matrices with the same shape.
rows, columns = custom_op.Symbols("rows", "columns")
add_one = custom_op.declare(
    "add_one",
    inputs={"x": custom_op.TemplateType(DType.float32, [rows, columns])},
    outputs=[custom_op.TemplateType(DType.float32, [rows, columns])],
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

    x_values = np.random.uniform(size=(5, 10)).astype(np.float32)
    expected = x_values + 1

    # Eager: call the op on a tensor.
    x = Tensor.from_dlpack(x_values).to(device)
    result = add_one(x)
    assert isinstance(result, Tensor)
    check("Eager", result.to(CPU()).to_numpy(), expected)

    # Graph: call the same op on a graph value.
    device_ref = DeviceRef.from_device(device)
    graph = Graph(
        "declared_addition",
        forward=add_one,
        input_types=[TensorType(DType.float32, ["m", "n"], device=device_ref)],
    )

    session = InferenceSession(devices=[device])
    model = session.load(graph)
    output = model.execute(Buffer.from_numpy(x_values).to(device))[0]
    assert isinstance(output, Buffer)
    check("Graph", output.to(CPU()).to_numpy(), expected)
