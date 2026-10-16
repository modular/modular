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
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops


def add_one(x: TensorValue, device_ref: DeviceRef) -> TensorValue:
    return ops.custom(
        name="add_one_elementwise",
        device=device_ref,
        values=[x],
        out_types=[TensorType(dtype=x.dtype, shape=x.shape, device=device_ref)],
    )[0].tensor


def main() -> None:
    mojo_kernels = Path(__file__).parent / "kernels"

    rows = 5
    inner = 8
    columns = 10
    dtype = DType.float32
    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()
    device_ref = DeviceRef.from_device(device)

    # The custom op and the multiply follow a matmul, so the graph compiler
    # can fuse them into the matmul's epilogue.
    graph = Graph(
        "elementwise_addition",
        forward=lambda a, b: add_one(a @ b, device_ref) * 2,
        input_types=[
            TensorType(dtype, shape=[rows, inner], device=device_ref),
            TensorType(dtype, shape=[inner, columns], device=device_ref),
        ],
        custom_extensions=[mojo_kernels],
    )

    session = InferenceSession(devices=[device])
    compiled = session.compile(graph)
    model = session.init(compiled)

    a_values = np.random.uniform(size=(rows, inner)).astype(np.float32)
    b_values = np.random.uniform(size=(inner, columns)).astype(np.float32)
    a = Buffer.from_numpy(a_values).to(device)
    b = Buffer.from_numpy(b_values).to(device)

    result = model.execute(a, b)[0]
    assert isinstance(result, Buffer)
    result_values = result.to(CPU()).to_numpy()

    expected = (a_values @ b_values + 1) * 2
    print("Graph result:")
    print(result_values)
    print()
    print("Expected result:")
    print(expected)

    np.testing.assert_allclose(result_values, expected, rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    main()
