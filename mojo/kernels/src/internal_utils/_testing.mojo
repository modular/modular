# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections import Optional

import testing
from buffer import NDBuffer
from builtin._location import __call_location, _SourceLocation
from testing.testing import _assert_cmp_error

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
    location: Optional[_SourceLocation] = None,
    atol: Scalar[type] = 1e-08,
    rtol: Scalar[type] = 1e-05,
    equal_nan: Bool = False,
) raises:
    for i in range(num_elements):
        testing.assert_almost_equal(
            x[i],
            y[i],
            msg=msg + " at i=" + str(i),
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
    location: Optional[_SourceLocation] = None,
    atol: Scalar[x.type] = 1e-08,
    rtol: Scalar[x.type] = 1e-05,
    equal_nan: Bool = False,
) raises:
    for i in range(x.num_elements()):
        testing.assert_almost_equal(
            x.data[i],
            y.data[i],
            msg=msg + " at " + str(x.get_nd_index(i)),
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
    location: Optional[_SourceLocation] = None,
    atol: Scalar[x.type] = 1e-08,
    rtol: Scalar[x.type] = 1e-05,
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
    location: Optional[_SourceLocation] = None,
    atol: Scalar[x.type] = 1e-08,
    rtol: Scalar[x.type] = 1e-05,
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
    location: Optional[_SourceLocation] = None,
) raises:
    for i in range(x.num_elements()):
        testing.assert_equal(
            x.data[i],
            y.data[i],
            msg=msg + " at " + str(x.get_nd_index(i)),
            location=location.or_else(__call_location()),
        )


@always_inline
fn assert_equal(
    x: HostNDBuffer,
    y: __type_of(x),
    msg: String = "",
    *,
    location: Optional[_SourceLocation] = None,
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
    location: Optional[_SourceLocation] = None,
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
    ) capturing -> Bool,
](
    x: UnsafePointer[Scalar[type], *_],
    y: __type_of(x),
    n: Int,
    msg: String = "",
    *,
    location: Optional[_SourceLocation] = None,
) raises:
    if not measure(
        x.bitcast[address_space = AddressSpace.GENERIC](),
        y.bitcast[address_space = AddressSpace.GENERIC](),
        n,
    ):
        raise _assert_cmp_error["`left == right` comparison"](
            str(x), str(y), msg=msg, loc=location.or_else(__call_location())
        )


@always_inline
fn assert_with_measure[
    measure: fn[type: DType] (
        UnsafePointer[Scalar[type]], UnsafePointer[Scalar[type]], Int
    ) capturing -> Bool
](
    x: NDBuffer,
    y: __type_of(x),
    msg: String = "",
    *,
    location: Optional[_SourceLocation] = None,
) raises:
    _assert_with_measure_impl[measure](
        x.data,
        y.data,
        x.num_elements(),
        msg=msg,
        location=location.or_else(__call_location()),
    )


@always_inline
fn assert_with_measure[
    measure: fn[type: DType] (
        UnsafePointer[Scalar[type]], UnsafePointer[Scalar[type]], Int
    ) capturing -> Bool
](
    x: HostNDBuffer,
    y: __type_of(x),
    msg: String = "",
    *,
    location: Optional[_SourceLocation] = None,
) raises:
    _assert_with_measure_impl[measure](
        x.tensor.data,
        y.tensor.data,
        x.tensor.num_elements(),
        msg=msg,
        location=location.or_else(__call_location()),
    )


@always_inline
fn assert_with_measure[
    measure: fn[type: DType] (
        UnsafePointer[Scalar[type]], UnsafePointer[Scalar[type]], Int
    ) capturing -> Bool
](
    x: TestTensor,
    y: __type_of(x),
    msg: String = "",
    *,
    location: Optional[_SourceLocation] = None,
) raises:
    return assert_with_measure[measure](
        x.ndbuffer,
        y.ndbuffer,
        msg=msg,
        location=location.or_else(__call_location()),
    )
