# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import NDBuffer
from Index import StaticIntTuple
from Functional import _elementwise_impl
from List import DimList
from runtime.llcl import OutputChainPtr
from math import abs, ceil, div_ceil, iota
from sys.info import simdwidthof
from MOGGDecorators import *
from TypeUtilities import rebind

# ===----------------------------------------------------------------------===#
# Arange op
# ===----------------------------------------------------------------------===#


@always_inline
fn _arange_get_numelems[
    type: DType
](start: SIMD[type, 1], stop: SIMD[type, 1], step: SIMD[type, 1]) -> Int:
    @parameter
    if type.is_integral():
        return range(start.to_int(), stop.to_int(), step.to_int()).__len__()
    else:
        return ceil(abs(stop - start) / abs(step)).to_int()


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
) -> StaticIntTuple[1]:
    let numElems = _arange_get_numelems(start_buf[0], stop_buf[0], step_buf[0])
    return StaticIntTuple[1](numElems)
