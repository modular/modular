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
        DType.float4_e2m1fn,
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


def test_roundtrip() -> None:
    for dtype in DType:
        assert isinstance(dtype, DType)
        assert DType(dtype._mlir) == dtype


@given(dtype=int_dtype | float_dtype)
def test_numpy_roundtrip(dtype: DType) -> None:
    # There is no f4 / float8 / bf16 in numpy, so we cannot roundtrip
    # f4 / float8 / bf16
    if dtype in [
        DType.float4_e2m1fn,
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
def test_is_integral(int_dtype: DType, float_dtype: DType) -> None:
    assert int_dtype.is_integral()
    assert not float_dtype.is_integral()


@given(int_dtype=int_dtype, float_dtype=float_dtype)
def test_is_float(int_dtype: DType, float_dtype: DType) -> None:
    assert not int_dtype.is_float()
    assert float_dtype.is_float()


def test_dtype_alignment() -> None:
    assert DType.bool.align == 1
    assert DType.int8.align == 1
    assert DType.int16.align == 2
    assert DType.int32.align == 4
    assert DType.int64.align == 8
    assert DType.uint8.align == 1
    assert DType.uint16.align == 2
    assert DType.uint32.align == 4
    assert DType.uint64.align == 8
    assert DType.float16.align == 2
    assert DType.float32.align == 4
    assert DType.float64.align == 8
    assert DType.bfloat16.align == 2
    assert DType.float4_e2m1fn.align == 1
    assert DType.float8_e4m3fn.align == 1
    assert DType.float8_e4m3fnuz.align == 1
    assert DType.float8_e5m2.align == 1
    assert DType.float8_e5m2fnuz.align == 1
