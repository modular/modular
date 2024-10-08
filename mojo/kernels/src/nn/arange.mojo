# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceil, iota

from buffer import NDBuffer
from register import *

from utils.index import IndexList

# ===----------------------------------------------------------------------===#
# Arange op
# ===----------------------------------------------------------------------===#


@mogg_register("mo.range")
@mogg_elementwise
@mogg_takes_indices()
@always_inline
fn arange[
    type: DType, simd_width: Int
](
    start_buf: NDBuffer[type, 1],
    stop_buf: NDBuffer[type, 1],
    step_buf: NDBuffer[type, 1],
    index: IndexList[1],
) -> SIMD[type, simd_width]:
    return start_buf[0] + (iota[type, simd_width](index[0]) * step_buf[0])


@mogg_register_shape_func("mo.range")
@always_inline
fn arange_shape[
    type: DType,
    single_thread_blocking_override: Bool,
](
    start_buf: NDBuffer[type, 1],
    stop_buf: NDBuffer[type, 1],
    step_buf: NDBuffer[type, 1],
) raises -> IndexList[1]:
    var start: Scalar[type] = start_buf[0]
    var stop: Scalar[type] = stop_buf[0]
    var step: Scalar[type] = step_buf[0]
    if step == 0:
        raise Error("[range] step must be non-zero")

    @parameter
    if start.type.is_integral():
        if step > 0 and stop < start:
            raise Error("[range] requires (start <= stop) for positive step")

        if step < 0 and start < stop:
            raise Error("[range] requires (stop <= start) for negative step")

        return IndexList[1](len(range(int(start), int(stop), int(step))))
    else:
        return IndexList[1](int(ceil(abs(stop - start) / abs(step))))
