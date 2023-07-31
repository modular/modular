# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import NDBuffer
from DType import DType
from Index import StaticIntTuple
from Functional import _elementwise_impl
from List import DimList
from LLCL import OutputChainPtr
from Math import abs, ceil, div_ceil, iota
from SIMD import SIMD
from TargetInfo import simdwidthof
from TypeUtilities import rebind


# ===----------------------------------------------------------------------===#
# Arange op
# ===----------------------------------------------------------------------===#


@always_inline
fn _arange_get_numelems[
    type: DType
](start: SIMD[type, 1], stop: SIMD[type, 1], step: SIMD[type, 1]) -> Int:
    let diff = abs(stop - start)

    @parameter
    if type.is_integral():
        return div_ceil(diff.to_int(), abs(step).to_int())
    else:
        return ceil(diff / abs(step)).to_int()


@always_inline
fn arange[
    type: DType,
    single_thread_blocking_override: Bool,
](
    start_buf: NDBuffer[1, DimList.create_unknown[1](), type],
    stop_buf: NDBuffer[1, DimList.create_unknown[1](), type],
    step_buf: NDBuffer[1, DimList.create_unknown[1](), type],
    output_buffer: NDBuffer[1, DimList.create_unknown[1](), type],
    out_chain: OutputChainPtr,
):
    let start = start_buf[0]
    let stop = stop_buf[0]
    let step = step_buf[0]
    let numElems = _arange_get_numelems(start, stop, step)

    @parameter
    @always_inline
    fn range_lambda[
        simd_width: Int, rank: Int
    ](out_index: StaticIntTuple[rank]):
        let value = start + (iota[type, simd_width](out_index[0]) * step)
        output_buffer.simd_store[simd_width](
            rebind[StaticIntTuple[1]](out_index), value
        )

    alias simd_width = simdwidthof[type]()

    if numElems % simd_width != 0:
        _elementwise_impl[1, 1, single_thread_blocking_override, range_lambda](
            StaticIntTuple[1](numElems), out_chain
        )
    else:
        _elementwise_impl[
            1, simd_width, single_thread_blocking_override, range_lambda
        ](StaticIntTuple[1](numElems), out_chain)


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
