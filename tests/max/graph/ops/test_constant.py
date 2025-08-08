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

import re

import numpy as np
from hypothesis import assume, given
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def test_constant() -> None:
    with Graph("constants", input_types=()) as graph:
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3))
        const = ops.constant(
            const, DType.from_numpy(const.dtype), device=DeviceRef.CPU()
        )

        graph.output(const)

        assert "0, 1, 2, 3, 4, 5" in str(graph._mlir_op)


def test_constant_transpose() -> None:
    with Graph("constants", input_types=()) as graph:
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3)).T
        const = ops.constant(
            const, DType.from_numpy(const.dtype), device=DeviceRef.CPU()
        )

        graph.output(const)

        assert "0, 3, 1, 4, 2, 5" in str(graph._mlir_op)


@given(dtype=...)
def test_scalar_constant(dtype: DType) -> None:
    # Can represent an integer value
    assume(dtype != DType.bool)
    # Not supported by numpy
    assume(dtype != DType.bfloat16)
    with Graph("scalar", input_types=()) as graph:
        const = 7.2
        const = ops.constant(const, dtype, device=DeviceRef.CPU())

        graph.output(const)

        if dtype.is_float():
            expected = rf"mo.constant {{value = #M.dense_array<7\..*> : tensor<{dtype._mlir}>}}"
        else:
            expected = rf"mo.constant {{value = #M.dense_array<7> : tensor<{dtype._mlir}>}}"
        assert re.search(expected, str(graph._mlir_op))


@given(name=..., type=...)
def test_constant_external(name: str, type: TensorType):
    with Graph("constants", input_types=()) as graph:
        weight = ops.constant_external(name, type)
        assert weight.type == type
