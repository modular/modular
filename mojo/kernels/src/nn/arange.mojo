# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import abs, ceil, div_ceil, iota
from sys.info import simdwidthof

from memory.buffer import NDBuffer
from utils._annotations import *

from utils.index import StaticIntTuple
from utils.list import DimList

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
    start_buf: NDBuffer[1, DimList.create_unknown[1](), type],
    stop_buf: NDBuffer[1, DimList.create_unknown[1](), type],
    step_buf: NDBuffer[1, DimList.create_unknown[1](), type],
    index: StaticIntTuple[1],
) -> SIMD[type, simd_width]:
    return start_buf[0] + (iota[type, simd_width](index[0]) * step_buf[0])


@always_inline
fn arange_shape[
    type: DType,
    single_thread_blocking_override: Bool,
](
    start_buf: NDBuffer[1, DimList.create_unknown[1](), type],
    stop_buf: NDBuffer[1, DimList.create_unknown[1](), type],
    step_buf: NDBuffer[1, DimList.create_unknown[1](), type],
) raises -> StaticIntTuple[1]:
    let start: Scalar[type] = start_buf[0]
    let stop: Scalar[type] = stop_buf[0]
    let step: Scalar[type] = step_buf[0]
    if step == 0:
        raise Error("[range] step must be non-zero")

    @parameter
    if start.type.is_integral():
        if step > 0 and stop < start:
            raise Error("[range] requires (start <= stop) for positive step")

        if step < 0 and start < stop:
            raise Error("[range] requires (stop <= start) for negative step")

        return StaticIntTuple[1](len(range(start, stop, step)))
    else:
        return StaticIntTuple[1](int(ceil(abs(stop - start) / abs(step))))
