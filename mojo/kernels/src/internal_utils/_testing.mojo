# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections import OptionalReg
from math import exp2

import testing
from buffer import NDBuffer
from builtin._location import __call_location, _SourceLocation
from memory import UnsafePointer
from testing.testing import _assert_cmp_error

from utils.numerics import FPUtils

from ._utils import HostNDBuffer, TestTensor

# ===----------------------------------------------------------------------=== #
# assert_almost_equal
# ===----------------------------------------------------------------------=== #


@always_inline
fn assert_almost_equal[
    type: DType, //,
](
    x: UnsafePointer[Scalar[type]],
    y: __type_of(x),
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
    y: __type_of(x),
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


@always_inline
fn assert_almost_equal(
    x: HostNDBuffer,
    y: __type_of(x),
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
) raises:
    return assert_almost_equal(
        x.tensor,
        y.tensor,
        msg=msg,
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
        location=location.or_else(__call_location()),
    )


@always_inline
fn assert_almost_equal(
    x: TestTensor,
    y: __type_of(x),
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
) raises:
    return assert_almost_equal(
        x.ndbuffer,
        y.ndbuffer,
        msg=msg,
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
    y: __type_of(x),
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


@always_inline
fn assert_equal(
    x: HostNDBuffer,
    y: __type_of(x),
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
) raises:
    return assert_equal(
        x.tensor,
        y.tensor,
        msg=msg,
        location=location.or_else(__call_location()),
    )


@always_inline
fn assert_equal(
    x: TestTensor,
    y: __type_of(x),
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
) raises:
    return assert_equal(
        x.ndbuffer,
        y.ndbuffer,
        msg=msg,
        location=location.or_else(__call_location()),
    )


# ===----------------------------------------------------------------------=== #
# assert_with_measure
# ===----------------------------------------------------------------------=== #


@always_inline
fn _assert_with_measure_impl[
    type: DType, //,
    measure: fn[type: DType] (
        UnsafePointer[Scalar[type]], UnsafePointer[Scalar[type]], Int
    ) -> Float64,
](
    x: UnsafePointer[Scalar[type], **_],
    y: __type_of(x),
    n: Int,
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
    threshold: OptionalReg[Float64] = None,
) raises:
    alias sqrt_eps = exp2(-0.5 * FPUtils[type].mantissa_width()).cast[
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
    measure: fn[type: DType] (
        UnsafePointer[Scalar[type]], UnsafePointer[Scalar[type]], Int
    ) -> Float64,
](
    x: NDBuffer,
    y: __type_of(x),
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
fn assert_with_measure[
    measure: fn[type: DType] (
        UnsafePointer[Scalar[type]], UnsafePointer[Scalar[type]], Int
    ) -> Float64,
](
    x: HostNDBuffer,
    y: __type_of(x),
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
    threshold: OptionalReg[Float64] = None,
) raises:
    _assert_with_measure_impl[measure](
        x.tensor.data,
        y.tensor.data,
        x.tensor.num_elements(),
        msg=msg,
        location=location.or_else(__call_location()),
        threshold=threshold,
    )


@always_inline
fn assert_with_measure[
    measure: fn[type: DType] (
        UnsafePointer[Scalar[type]], UnsafePointer[Scalar[type]], Int
    ) -> Float64,
](
    x: TestTensor,
    y: __type_of(x),
    msg: String = "",
    *,
    location: OptionalReg[_SourceLocation] = None,
    threshold: OptionalReg[Float64] = None,
) raises:
    return assert_with_measure[measure](
        x.ndbuffer,
        y.ndbuffer,
        msg=msg,
        location=location.or_else(__call_location()),
        threshold=threshold,
    )


# ===----------------------------------------------------------------------=== #
# utils
# ===----------------------------------------------------------------------=== #


fn _minmax[
    type: DType, //
](x: UnsafePointer[Scalar[type]], N: Int) -> Tuple[Scalar[type], Scalar[type]]:
    var max_val = x[0]
    var min_val = x[0]
    for i in range(1, N):
        if x[i] > max_val:
            max_val = x[i]
        if x[i] < min_val:
            min_val = x[i]
    return (min_val, max_val)


fn compare[
    dtype: DType, verbose: Bool = True
](
    x: UnsafePointer[Scalar[dtype]],
    y: UnsafePointer[Scalar[dtype]],
    num_elements: Int,
    *,
    msg: String = "",
) -> SIMD[dtype, 4]:
    var atol = UnsafePointer[Scalar[dtype]].alloc(num_elements)
    var rtol = UnsafePointer[Scalar[dtype]].alloc(num_elements)

    for i in range(num_elements):
        var d = abs(x[i] - y[i])
        var e = abs(d / y[i])
        atol[i] = d
        rtol[i] = e

    var atol_minmax = _minmax(atol, num_elements)
    var rtol_minmax = _minmax(rtol, num_elements)
    if verbose:
        if msg:
            print(msg)
        print("AbsErr-Min/Max", atol_minmax[0], atol_minmax[1])
        print("RelErr-Min/Max", rtol_minmax[0], rtol_minmax[1])
        print("==========================================================")
    atol.free()
    rtol.free()
    return SIMD[dtype, 4](
        atol_minmax[0], atol_minmax[1], rtol_minmax[0], rtol_minmax[1]
    )
