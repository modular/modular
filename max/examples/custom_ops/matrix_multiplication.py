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

# DOC: max/develop/custom-ops-matmul.mdx

from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray


def matrix_multiplication(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
    algorithm: str,
    session: InferenceSession,
    device: Device,
) -> Buffer:
    dtype = DType.float32

    # Create driver tensors from the input arrays, and move them to the
    # accelerator.
    a_tensor = Buffer.from_numpy(a).to(device)
    b_tensor = Buffer.from_numpy(b).to(device)

    mojo_kernels = Path(__file__).parent / "kernels"

    # Configure our simple one-operation graph.
    with Graph(
        "matrix_multiplication_graph",
        input_types=[
            TensorType(
                dtype,
                shape=a_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=b_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        a_value, b_value = graph.inputs
        # The matrix multiplication custom operation takes in two matrices and
        # produces a result, with the specific algorithm that is used chosen
        # via compile-time parameterization.
        output = ops.custom(
            name="matrix_multiplication",
            device=DeviceRef.from_device(device),
            values=[a_value, b_value],
            out_types=[
                TensorType(
                    dtype=a_value.tensor.dtype,
                    shape=[a_value.tensor.shape[0], b_value.tensor.shape[1]],
                    device=DeviceRef.from_device(device),
                )
            ],
            parameters={"algorithm": algorithm},
        )[0].tensor
        graph.output(output)

    # Compile the graph.
    print("Compiling...")
    compiled = session.compile(graph)
    model = session.init(compiled)

    # Perform the calculation on the target device.
    print("Executing...")
    result = model.execute(a_tensor, b_tensor)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Buffer)
    return result.to(CPU())


if __name__ == "__main__":
    M = 256
    K = 256
    N = 256

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Set up an inference session for running the graph.
    session = InferenceSession(
        devices=[device],
    )

    # Dyadic inputs are exactly representable by the Tensor Core TF32 path.
    rng = np.random.default_rng(0)
    a = (rng.integers(-8, 9, size=(M, K)) / 16).astype(np.float32)
    b = (rng.integers(-8, 9, size=(K, N)) / 16).astype(np.float32)

    # Independent fp64 reference; the fp32 cast is the best-achievable result.
    expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)

    # First, perform the matrix multiplication in NumPy.
    print("A:")
    print(a)
    print()

    print("B:")
    print(b)
    print()

    print("Expected result:")
    print(expected)
    print()

    if accelerator_count() > 0:
        # Then, test the various versions of matrix multiplication operations.
        naive_result = matrix_multiplication(a, b, "naive", session, device)
        print("Naive matrix multiplication:")
        np.testing.assert_allclose(
            naive_result.to_numpy(), expected, rtol=1e-5, atol=1e-5
        )
        print(naive_result.to_numpy())
        print()

        coalescing_result = matrix_multiplication(
            a, b, "coalescing", session, device
        )
        print("Coalescing matrix multiplication:")
        np.testing.assert_allclose(
            coalescing_result.to_numpy(), expected, rtol=1e-5, atol=1e-5
        )
        print(coalescing_result.to_numpy())
        print()

        tiled_result = matrix_multiplication(a, b, "tiled", session, device)
        print("Tiled matrix multiplication:")
        np.testing.assert_allclose(
            tiled_result.to_numpy(), expected, rtol=1e-5, atol=1e-5
        )
        print(tiled_result.to_numpy())
        print()

        tiled_register_result = matrix_multiplication(
            a, b, "tiled_register", session, device
        )
        print("Shared memory and register tiling matrix multiplication:")
        np.testing.assert_allclose(
            tiled_register_result.to_numpy(), expected, rtol=1e-5, atol=1e-5
        )
        print(tiled_register_result.to_numpy())
        print()

        block_tiled_result = matrix_multiplication(
            a, b, "block_tiled", session, device
        )
        print("2D block tiled matrix multiplication:")
        np.testing.assert_allclose(
            block_tiled_result.to_numpy(), expected, rtol=1e-5, atol=1e-5
        )
        print(block_tiled_result.to_numpy())
        print()

        block_tiled_vectorized_result = matrix_multiplication(
            a, b, "block_tiled_vectorized", session, device
        )
        print("2D block tiled matrix multiplication (vectorized):")
        np.testing.assert_allclose(
            block_tiled_vectorized_result.to_numpy(),
            expected,
            rtol=1e-5,
            atol=1e-5,
        )
        print(block_tiled_vectorized_result.to_numpy())
        print()

        # Note: Apple silicon GPUs do not yet support Tensor Core operations.
        if device.api != "metal":
            tensor_core_result = matrix_multiplication(
                a, b, "tensor_core", session, device
            )
            print("Matrix multiplication using Tensor Cores:")
            np.testing.assert_allclose(
                tensor_core_result.to_numpy(), expected, rtol=1e-5, atol=1e-5
            )
            print(tensor_core_result.to_numpy())
            print()
    else:
        print(
            "No MAX-compatible accelerator detected, only running a naive"
            " matrix multiplication:"
        )

        naive_result = matrix_multiplication(a, b, "naive", session, device)
        print("Naive matrix multiplication:")
        np.testing.assert_allclose(
            naive_result.to_numpy(), expected, rtol=1e-5, atol=1e-5
        )
        print(naive_result.to_numpy())
        print()
