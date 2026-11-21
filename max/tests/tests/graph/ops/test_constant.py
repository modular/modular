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

import math
import re
import sys
import warnings
from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import pytest
from conftest import constant_float_dtypes, integral_dtypes
from hypothesis import given
from hypothesis import strategies as st
from max._core import graph as _graph
from max._core.dialects import mo
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType, ops
from max.graph.ops.constant import NestedArray, Number, shape
from max.tensor import Tensor

# TODO(GEX-3550): creating a device-resident `Tensor` on the Apple GPU
# (Metal) hangs indefinitely on BuildBuddy macOS workers, so the two tests
# below that exercise that code path time out the whole target.
_SKIP_APPLE_GPU = pytest.mark.skipif(
    accelerator_count() == 0 or sys.platform == "darwin",
    reason="requires at least 1 non-Apple GPU",
)


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
        result = ops.constant(Buffer.from_dlpack(const))

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


@given(dtype=constant_float_dtypes())
def test_scalar_constant_float(dtype: DType) -> None:
    with Graph("scalar", input_types=()) as graph:
        const = ops.constant(7.2, dtype, device=DeviceRef.CPU())
        graph.output(const)

        expected = r"mo.constant <#M.dense_array<7\..*> : tensor<([^>]*)>>"
        assert (match := re.search(expected, str(graph._mlir_op)))
        assert match.group(1).lower() == dtype._mlir


@given(dtype=constant_float_dtypes())
def test_scalar_constant_float_negative(dtype: DType) -> None:
    with Graph("scalar", input_types=()) as graph:
        const = ops.constant(-1.0, dtype, device=DeviceRef.CPU())
        graph.output(const)

        expected = r"mo.constant <#M.dense_array<-1\..*> : tensor<([^>]*)>>"
        assert (match := re.search(expected, str(graph._mlir_op))), str(graph)
        assert match.group(1).lower() == dtype._mlir


@given(dtype=integral_dtypes())
def test_scalar_constant_integral(dtype: DType) -> None:
    with Graph("scalar", input_types=()) as graph:
        const = ops.constant(1, dtype, device=DeviceRef.CPU())
        graph.output(const)

        expected = rf"mo.constant <#M.dense_array<1> : tensor<{dtype._mlir}>>"
        assert re.search(expected, str(graph._mlir_op))


@given(name=..., type=...)
def test_constant_external(name: str, type: TensorType) -> None:
    with Graph("constants", input_types=()):
        weight = ops.constant_external(name, type)
        assert weight.type == type


@_SKIP_APPLE_GPU
def test_constant_from_gpu_buffer_warns() -> None:
    """ops.constant warns and copies to CPU for device-resident data."""
    gpu_tensor = Tensor(
        [1.0, 2.0, 3.0, 4.0], dtype=DType.float32, device=Accelerator(0)
    )

    with Graph("gpu_constant", input_types=()) as graph:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ops.constant(gpu_tensor)
        assert len(w) == 1
        assert "ops.constant" in str(w[0].message)
        assert result.type == TensorType(DType.float32, [4], DeviceRef.GPU(0))
        graph.output(result)


@_SKIP_APPLE_GPU
def test_constant_from_gpu_buffer_error_policy() -> None:
    """ops.constant raises ValueError under strict device placement."""
    gpu_tensor = Tensor(
        [1.0, 2.0, 3.0, 4.0], dtype=DType.float32, device=Accelerator(0)
    )

    with (
        Graph(
            "gpu_constant",
            input_types=(),
            strict_device_placement=DevicePlacementPolicy.Error,
        ),
        pytest.raises(ValueError, match=r"ops\.constant"),
    ):
        ops.constant(gpu_tensor)


_LITERAL_DTYPES = [d for d in DType if d.size_in_bits >= 8]

Literal: TypeAlias = NestedArray | float


def _int_range(dtype: DType) -> tuple[int, int]:
    if dtype == DType.bool:
        return 0, 1
    bits = dtype.size_in_bits
    if dtype.is_signed_integral():
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def _literal_via_buffer(
    value: Literal, dtype: DType, mlir_type: mo.TensorType
) -> object:
    """Builds the attribute the way ``ops.constant`` once did: through a Buffer."""
    tensor = Buffer(dtype, shape(value), device=CPU())

    def elements(v: object, idx: Sequence[int]) -> float:
        for i in idx:
            assert isinstance(v, Sequence)
            v = v[i]
        assert isinstance(v, int | float)
        return v

    for idx in tensor._iterate_indices():
        v = elements(value, idx)
        if not dtype.is_float():
            lo, hi = _int_range(dtype)
            if not lo <= int(v) <= hi:
                raise ValueError(
                    "Unsafe cast: Refusing to implicitly promote external "
                    f"array with value {v} out of range for DType {dtype}."
                )
        tensor[idx] = v
    return _graph.array_attr(Buffer.from_dlpack(tensor), mlir_type)


def _outcome(build: Callable[[], object]) -> tuple[str, ...]:
    try:
        return ("ok", str(build()))
    except Exception as e:
        return ("raised", type(e).__name__, str(e))


def _literal_values(dtype: DType) -> list[Literal]:
    values: list[Literal] = [
        0,
        1,
        -1,
        0.1,
        -0.1,
        1e-8,
        1e30,
        1e300,
        math.nan,
        math.inf,
        -math.inf,
        True,
        False,
        2**64,
        -(2**64),
        1.5,
        [1, 2, 3, 4],
        (1, 2, 3, 4),
        [[1, 2, 3], [4, 5, 6]],
        ((1, 2, 3), (4, 5, 6)),
        [[[1, 2, 3, 4]]],
        [1, 2.5, True, -3],
        [[0.1, 1e-8, 2], [-1, 0, 3.5]],
        [0.0] * 16,
    ]
    if not dtype.is_float():
        lo, hi = _int_range(dtype)
        values += [lo, hi, lo - 1, hi + 1, [lo, hi], [[lo, 0], [hi, 1]]]
    else:
        values += [[1.0, math.nan, math.inf, -math.inf]]
    return values


def _nonnegative(value: Number | NestedArray) -> Literal:
    if isinstance(value, Sequence):
        return [_nonnegative(v) for v in value]
    assert isinstance(value, int | float)
    return abs(value)


def _assert_matches_buffer_path(value: Literal, dtype: DType) -> None:
    if dtype == DType.float8_e8m0fnu:
        # The format has no sign bit, so a negative value has no encoding.
        value = _nonnegative(value)
    mlir_type = TensorType(dtype, shape(value), DeviceRef.CPU()).to_mlir()
    expected = _outcome(lambda: _literal_via_buffer(value, dtype, mlir_type))
    actual = _outcome(lambda: _graph.array_attr(value, mlir_type))
    assert actual == expected, (dtype, value)


@pytest.mark.parametrize("dtype", _LITERAL_DTYPES, ids=str)
def test_literal_attr_matches_buffer_path(dtype: DType) -> None:
    with Graph("parity", input_types=()):
        for value in _literal_values(dtype):
            _assert_matches_buffer_path(value, dtype)


@given(
    dtype=st.sampled_from(_LITERAL_DTYPES),
    value=st.lists(
        st.lists(
            st.one_of(
                st.integers(min_value=-(2**65), max_value=2**65),
                st.floats(allow_nan=True, allow_infinity=True),
                st.booleans(),
            ),
            min_size=2,
            max_size=2,
        ),
        min_size=1,
        max_size=3,
    ),
)
def test_literal_attr_matches_buffer_path_random(
    dtype: DType, value: NestedArray
) -> None:
    with Graph("parity", input_types=()):
        _assert_matches_buffer_path(value, dtype)


def test_array_attr_overloads_keep_their_arguments_apart() -> None:
    with Graph("overloads", input_types=()):
        mlir_type = TensorType(DType.float32, [2], DeviceRef.CPU()).to_mlir()
        buffer = Buffer.from_dlpack(np.array([1.0, 2.0], dtype=np.float32))
        from_buffer = _graph.array_attr(buffer, mlir_type)
        assert _graph.array_attr([1.0, 2.0], mlir_type) == from_buffer
        assert _graph.array_attr((1.0, 2.0), mlir_type) == from_buffer

        scalar_type = TensorType(DType.float32, [], DeviceRef.CPU()).to_mlir()
        scalar = Buffer.from_dlpack(np.array(3.0, dtype=np.float32))
        assert _graph.array_attr(3.0, scalar_type) == _graph.array_attr(
            scalar, scalar_type
        )


def test_array_attr_literal_shape_mismatch() -> None:
    with Graph("mismatch", input_types=()):
        mlir_type = TensorType(DType.int32, [2, 2], DeviceRef.CPU()).to_mlir()
        with pytest.raises(ValueError):
            _graph.array_attr([[1, 2], [3]], mlir_type)
        with pytest.raises(ValueError):
            _graph.array_attr([1, 2], mlir_type)
        with pytest.raises(ValueError):
            _graph.array_attr(1, mlir_type)


@pytest.mark.parametrize(
    "np_dtype",
    [np.bool_, np.int8, np.int32, np.uint8, np.uint64, np.float16, np.float32],
)
def test_constant_from_numpy_matches_buffer_path(
    np_dtype: type[np.generic],
) -> None:
    base = np.arange(12).reshape(3, 4).astype(np_dtype)
    for array in (base, base.T, base[::2], np.array(1, dtype=np_dtype)):
        with Graph("numpy_parity", input_types=()) as expected_graph:
            buffer = Buffer.from_dlpack(np.ascontiguousarray(array))
            expected_graph.output(ops.constant(buffer))
        with Graph("numpy_parity", input_types=()) as graph:
            graph.output(ops.constant(array))
        assert str(graph._mlir_op) == str(expected_graph._mlir_op)
