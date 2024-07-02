# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from buffer import NDBuffer
import testing
from ._utils import TestTensor
from collections import Optional
from builtin._location import _SourceLocation


fn assert_almost_equal(
    x: DTypePointer,
    y: __type_of(x),
    num_elements: Int,
    msg: String = "",
    *,
    location: Optional[_SourceLocation] = None,
    atol: Scalar[x.type] = 1e-08,
    rtol: Scalar[x.type] = 1e-05,
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
            location=location,
        )


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
            location=location,
        )


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
        location=location,
    )


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
            location=location,
        )


fn assert_equal(
    x: TestTensor,
    y: __type_of(x),
    msg: String = "",
    *,
    location: Optional[_SourceLocation] = None,
) raises:
    return assert_equal(x.ndbuffer, y.ndbuffer, msg=msg, location=location)


fn _minmax(x: DTypePointer, N: Int) -> Tuple[Scalar[x.type], Scalar[x.type]]:
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
    x: DTypePointer[dtype],
    y: DTypePointer[dtype],
    num_elements: Int,
    *,
    msg: String = "",
) -> SIMD[dtype, 4]:
    var atol = DTypePointer[dtype].alloc(num_elements)
    var rtol = DTypePointer[dtype].alloc(num_elements)

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
