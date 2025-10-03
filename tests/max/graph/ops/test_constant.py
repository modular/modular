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
import pytest
from conftest import float_dtypes, integral_dtypes
from hypothesis import given
from max.driver import Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def test_constant_from_numpy() -> None:
    with Graph("constants", input_types=()) as graph:
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3))
        graph.output(ops.constant(const))

        assert "0, 1, 2, 3, 4, 5" in str(graph._mlir_op)


def test_constant_0d() -> None:
    with Graph("constants", input_types=()):
        result = ops.constant(1, DType.float32, DeviceRef.CPU())
        assert list(result.shape) == []


def test_constant_1_0d() -> None:
    with Graph("constants", input_types=()):
        input: list[float] = []
        result = ops.constant(input, DType.float32, DeviceRef.CPU())

        assert list(result.shape) == [0]


def test_constant_1_1_0d() -> None:
    with Graph("constants", input_types=()):
        input: list[list[float]] = [[]]
        result = ops.constant(input, DType.float32, DeviceRef.CPU())

        assert list(result.shape) == [1, 0]


def test_constant_from_driver_tensor() -> None:
    with Graph("constants", input_types=()):
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3))
        result = ops.constant(Tensor.from_dlpack(const))

        assert result.type == TensorType(DType.int64, [2, 3], DeviceRef.CPU())


def test_constant_with_device() -> None:
    with Graph("constants", input_types=()):
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3))
        result = ops.constant(const, device=DeviceRef.GPU())
        assert result.type == TensorType(DType.int64, [2, 3], DeviceRef.GPU())


def test_constant_with_dtype() -> None:
    with Graph("constants", input_types=()):
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3))
        result = ops.constant(const, dtype=DType.int64)
        assert result.type == TensorType(DType.int64, [2, 3], DeviceRef.CPU())


def test_constant_with_incorrect_dtype() -> None:
    with Graph("constants", input_types=()):
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3))
        with pytest.raises(ValueError):
            ops.constant(const, dtype=DType.int8)


def test_array_constant() -> None:
    with Graph("constants", input_types=()):
        result = ops.constant(
            [[0, 1, 2], [3, 4, 5]], dtype=DType.int8, device=DeviceRef.GPU()
        )
        assert result.type == TensorType(DType.int8, [2, 3], DeviceRef.GPU())


def test_array_constant_nonsquare() -> None:
    with Graph("constants", input_types=()):
        with pytest.raises(ValueError):
            ops.constant(
                [[0, 1, 2], [3, 4]], dtype=DType.int8, device=DeviceRef.GPU()
            )


def test_array_constant_no_dtype() -> None:
    with Graph("constants", input_types=()):
        with pytest.raises(TypeError):
            ops.constant([[0, 1, 2], [3, 4, 5]], dtype=DType.int8)


def test_array_constant_no_device() -> None:
    with Graph("constants", input_types=()):
        with pytest.raises(TypeError):
            ops.constant([[0, 1, 2], [3, 4, 5]], device=DeviceRef.CPU())


def test_constant_transpose() -> None:
    with Graph("constants", input_types=()) as graph:
        const = np.array([0, 1, 2, 3, 4, 5]).astype(np.int64).reshape((2, 3)).T
        graph.output(ops.constant(const))

        assert "0, 3, 1, 4, 2, 5" in str(graph._mlir_op)


@given(dtype=float_dtypes())
def test_scalar_constant_float(dtype: DType) -> None:
    with Graph("scalar", input_types=()) as graph:
        const = ops.constant(7.2, dtype, device=DeviceRef.CPU())
        graph.output(const)

        expected = (
            r"mo.constant {value = #M.dense_array<7\..*> : tensor<([^>]*)>}"
        )
        assert (match := re.search(expected, str(graph._mlir_op)))
        assert match.group(1).lower() == dtype._mlir


@given(dtype=float_dtypes())
def test_scalar_constant_float_negative(dtype: DType) -> None:
    with Graph("scalar", input_types=()) as graph:
        const = ops.constant(-1.0, dtype, device=DeviceRef.CPU())
        graph.output(const)

        expected = (
            r"mo.constant {value = #M.dense_array<-1\..*> : tensor<([^>]*)>}"
        )
        assert (match := re.search(expected, str(graph._mlir_op))), str(graph)
        assert match.group(1).lower() == dtype._mlir


@given(dtype=integral_dtypes())
def test_scalar_constant_integral(dtype: DType) -> None:
    with Graph("scalar", input_types=()) as graph:
        const = ops.constant(1, dtype, device=DeviceRef.CPU())
        graph.output(const)

        expected = rf"mo.constant {{value = #M.dense_array<1> : tensor<{dtype._mlir}>}}"
        assert re.search(expected, str(graph._mlir_op))


@given(name=..., type=...)
def test_constant_external(name: str, type: TensorType):
    with Graph("constants", input_types=()) as graph:
        weight = ops.constant_external(name, type)
        assert weight.type == type
