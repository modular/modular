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


from collections import OptionalReg
from math import exp2

import testing
from buffer import NDBuffer
from builtin._location import __call_location, _SourceLocation
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from testing.testing import _assert_cmp_error

from utils.numerics import FPUtils


# ===----------------------------------------------------------------------=== #
# assert_almost_equal
# ===----------------------------------------------------------------------=== #


@always_inline
fn assert_almost_equal[
    dtype: DType,
    //,
](
    x: UnsafePointer[Scalar[dtype]],
    y: type_of(x),
    num_elements: Int,
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
) raises:
    for i in range(num_elements):
        testing.assert_almost_equal(
            x[i],
            y[i],
            msg=String(msg, " at i=", i),
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
            location=location.or_else(__call_location()),
        )


@always_inline
fn assert_almost_equal(
    x: NDBuffer,
    y: type_of(x),
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
) raises:
    for i in range(x.num_elements()):
        testing.assert_almost_equal(
            x.data[i],
            y.data[i],
            msg=String(msg, " at ", x.get_nd_index(i)),
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
            location=location.or_else(__call_location()),
        )


# ===----------------------------------------------------------------------=== #
# assert_equal
# ===----------------------------------------------------------------------=== #


@always_inline
fn assert_equal(
    x: NDBuffer,
    y: type_of(x),
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
) raises:
    for i in range(x.num_elements()):
        testing.assert_equal(
            x.data[i],
            y.data[i],
            msg=String(msg, " at ", x.get_nd_index(i)),
            location=location.or_else(__call_location()),
        )


# ===----------------------------------------------------------------------=== #
# assert_with_measure
# ===----------------------------------------------------------------------=== #


@always_inline
fn _assert_with_measure_impl[
    dtype: DType,
    //,
    measure: fn[dtype: DType] (
        LegacyUnsafePointer[mut=False, Scalar[dtype]],
        LegacyUnsafePointer[mut=False, Scalar[dtype]],
        Int,
    ) -> Float64,
](
    x: UnsafePointer[Scalar[dtype], ...],
    y: type_of(x),
    n: Int,
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
    threshold: OptionalReg[Float64] = None,
) raises:
    comptime sqrt_eps = exp2(-0.5 * FPUtils[dtype].mantissa_width()).cast[
        DType.float64
    ]()
    var m = measure(
        x.address_space_cast[AddressSpace.GENERIC](),
        y.address_space_cast[AddressSpace.GENERIC](),
        n,
    )
    var t = threshold.or_else(sqrt_eps)
    if m > t:
        raise _assert_cmp_error["`left > right`, left = measure"](
            String(m),
            String(t),
            msg=msg,
            loc=location.or_else(__call_location()),
        )


@always_inline
fn assert_with_measure[
    measure: fn[dtype: DType] (
        LegacyUnsafePointer[mut=False, Scalar[dtype]],
        LegacyUnsafePointer[mut=False, Scalar[dtype]],
        Int,
    ) -> Float64,
](
    x: NDBuffer,
    y: type_of(x),
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
    threshold: OptionalReg[Float64] = None,
) raises:
    _assert_with_measure_impl[measure](
        x.data,
        y.data,
        x.num_elements(),
        msg=msg,
        location=location.or_else(__call_location()),
        threshold=threshold,
    )


@always_inline
fn pytorch_like_tolerances_for[dtype: DType]() -> Tuple[Float64, Float64]:
    # Returns (rtol, atol) modeled after PyTorch defaults.
    @parameter
    if dtype == DType.float16:
        return (1e-3, 1e-5)
    elif dtype == DType.bfloat16:
        return (1.6e-2, 1e-5)
    elif dtype == DType.float32:
        return (1.3e-6, 1e-5)
    elif dtype == DType.float64:
        return (1e-7, 1e-7)
    else:
        return (0.0, 0.0)
