# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph dtype operations."""

from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType

int_dtype = st.sampled_from(
    [
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    ]
)

float_dtype = st.sampled_from(
    [
        DType.float8_e4m3,
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
        DType.float8_e5m2,
        DType.float8_e5m2fnuz,
        DType.bfloat16,
        DType.float16,
        DType.float32,
        DType.float64,
    ]
)


@given(dtype=int_dtype | float_dtype)
def test_numpy_roundtrip(dtype: DType):
    # There is no float8 / bf16 in numpy, so we cannot roundtrip float8 / bf16
    if dtype in [
        DType.float8_e4m3,
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
        DType.float8_e5m2,
        DType.float8_e5m2fnuz,
        DType.bfloat16,
    ]:
        return
    np_dtype = dtype.to_numpy()
    assert dtype == DType.from_numpy(np_dtype)


@given(int_dtype=int_dtype, float_dtype=float_dtype)
def test_is_integral(int_dtype: DType, float_dtype: DType):
    assert int_dtype.is_integral()
    assert not float_dtype.is_integral()


@given(int_dtype=int_dtype, float_dtype=float_dtype)
def test_is_float(int_dtype: DType, float_dtype: DType):
    assert not int_dtype.is_float()
    assert float_dtype.is_float()
