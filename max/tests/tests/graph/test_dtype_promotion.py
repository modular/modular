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
"""Test the max.graph dtype promotion."""

import numpy as np
import pytest
from conftest import (
    constant_float_dtypes,
    float_dtypes,
    int_value_in_range,
    integral_dtypes,
    value_in_range,
    value_out_of_range,
)
from hypothesis import assume, given
from hypothesis import strategies as st
from max.driver import CPU, Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, dtype_promotion

shared_dtypes = st.shared(st.from_type(DType))
shared_float_dtypes = st.shared(float_dtypes())
shared_integral_dtypes = st.shared(integral_dtypes())


@given(dtype=integral_dtypes(), scalar=...)
def test_promote_weak_dtypes__python_float__int_dtype(
    dtype: DType, scalar: float
) -> None:
    with Graph("promote_weak_dtypes"):
        with pytest.raises(ValueError, match="Unsafe cast"):
            _ = dtype_promotion._promote_to_strong(
                scalar, dtype, DeviceRef.CPU()
            )


@given(dtype=constant_float_dtypes(), scalar=...)
def test_promote_weak_dtypes__python_float(dtype: DType, scalar: float) -> None:
    with Graph("promote_weak_dtypes"):
        result = dtype_promotion._promote_to_strong(
            scalar, dtype, DeviceRef.CPU()
        )
        assert result.dtype == dtype
        assert result.shape == []


@given(dtype=shared_dtypes, scalar=shared_dtypes.flatmap(value_in_range))
def test_promote_weak_dtypes__python_int__in_range(
    dtype: DType, scalar: int
) -> None:
    with Graph("promote_weak_dtypes"):
        result = dtype_promotion._promote_to_strong(
            scalar, dtype, DeviceRef.CPU()
        )
        assert result.dtype == dtype
        assert result.shape == []


@given(dtype=shared_dtypes, scalar=shared_dtypes.flatmap(value_out_of_range))
def test_promote_weak_dtypes__python_int__out_of_range(
    dtype: DType, scalar: int
) -> None:
    with Graph("promote_weak_dtypes"):
        with pytest.raises(ValueError, match="Unsafe cast"):
            _ = dtype_promotion._promote_to_strong(
                scalar, dtype, DeviceRef.CPU()
            )


@given(dtype=constant_float_dtypes(), scalar=...)
def test_promote_weak_dtypes__numpy_float(dtype: DType, scalar: float) -> None:
    with Graph("promote_weak_dtypes"):
        np_const = np.array(scalar)
        result = dtype_promotion._promote_to_strong(
            np_const, dtype, DeviceRef.CPU()
        )
        assert result.dtype == dtype
        assert result.shape == []


@given(dtype=integral_dtypes(), scalar=...)
def test_promote_weak_dtypes__numpy_float_to_int(
    dtype: DType, scalar: float
) -> None:
    with Graph("promote_weak_dtypes"):
        np_const = np.array(scalar)
        with pytest.raises(ValueError, match="Unsafe cast"):
            _ = dtype_promotion._promote_to_strong(
                np_const, dtype, DeviceRef.CPU()
            )


@given(dtype=shared_dtypes, scalar=shared_dtypes.flatmap(int_value_in_range))
def test_promote_weak_dtypes__numpy_int__in_range(
    dtype: DType, scalar: int
) -> None:
    # Need to fit the value in a numpy array
    assume(-(2**63) <= scalar < 2**63)
    with Graph("promote_weak_dtypes"):
        np_const = np.array(scalar, dtype=np.int64)
        result = dtype_promotion._promote_to_strong(
            np_const, dtype, DeviceRef.CPU()
        )
        assert result.dtype == dtype
        assert result.shape == []


@given(dtype=shared_dtypes, scalar=shared_dtypes.flatmap(value_out_of_range))
def test_promote_weak_dtypes__numpy_int__out_of_range(
    dtype: DType, scalar: int
) -> None:
    # Need to fit the value in a numpy array
    assume(-(2**63) <= scalar < 2**63)
    with Graph("promote_weak_dtypes"):
        np_const = np.array(scalar, dtype=np.int64)
        with pytest.raises(ValueError, match="Unsafe cast"):
            _ = dtype_promotion._promote_to_strong(
                np_const, dtype, DeviceRef.CPU()
            )


@given(dtype=float_dtypes(), scalar=...)
def test_promote_weak_dtypes__tensor_float(dtype: DType, scalar: float) -> None:
    with Graph("promote_weak_dtypes"):
        const = Tensor(DType.float64, [], CPU())
        const[()] = scalar
        result = dtype_promotion._promote_to_strong(
            const, dtype, DeviceRef.CPU()
        )
        assert result.dtype == dtype
        assert result.shape == []


@given(dtype=integral_dtypes(), scalar=...)
def test_promote_weak_dtypes__tensor_float_to_int(
    dtype: DType, scalar: float
) -> None:
    with Graph("promote_weak_dtypes"):
        const = Tensor(DType.float64, [], CPU())
        const[()] = scalar
        with pytest.raises(ValueError, match="Unsafe cast"):
            _ = dtype_promotion._promote_to_strong(
                const, dtype, DeviceRef.CPU()
            )


@given(dtype=shared_dtypes, scalar=shared_dtypes.flatmap(int_value_in_range))
def test_promote_weak_dtypes__tensor_int__in_range(
    dtype: DType, scalar: int
) -> None:
    # Need to fit the value in a tensor array
    assume(-(2**63) <= scalar < 2**63)
    with Graph("promote_weak_dtypes"):
        const = Tensor(DType.int64, [], CPU())
        const[()] = scalar
        result = dtype_promotion._promote_to_strong(
            const, dtype, DeviceRef.CPU()
        )
        assert result.dtype == dtype
        assert result.shape == []


@given(dtype=shared_dtypes, scalar=shared_dtypes.flatmap(value_out_of_range))
def test_promote_weak_dtypes__tensor_int__out_of_range(
    dtype: DType, scalar: int
) -> None:
    # Need to fit the value in a tensor array
    assume(-(2**63) <= scalar < 2**63)
    with Graph("promote_weak_dtypes"):
        const = Tensor(DType.int64, [], CPU())
        const[()] = scalar
        with pytest.raises(ValueError, match="Unsafe cast"):
            _ = dtype_promotion._promote_to_strong(
                const, dtype, DeviceRef.CPU()
            )
