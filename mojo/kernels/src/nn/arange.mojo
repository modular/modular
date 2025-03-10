# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceil, iota

from buffer import NDBuffer
from register import *

from utils.index import IndexList

# ===-----------------------------------------------------------------------===#
# Arange op
# ===-----------------------------------------------------------------------===#


@register_internal("mo.range")
@always_inline
fn arange[
    type: DType, simd_width: Int
](
    start: Scalar[type],
    stop: Scalar[type],
    step: Scalar[type],
    index: IndexList[1],
) -> SIMD[type, simd_width]:
    return start + (iota[type, simd_width](index[0]) * step)


@always_inline
fn arange_shape[
    type: DType,
    single_thread_blocking_override: Bool,
](
    start: Scalar[type],
    stop: Scalar[type],
    step: Scalar[type],
) raises -> IndexList[1]:
    if step == 0:
        raise Error("[range] step must be non-zero")

    @parameter
    if start.type.is_integral():
        if step > 0 and stop < start:
            raise Error("[range] requires (start <= stop) for positive step")

        if step < 0 and start < stop:
            raise Error("[range] requires (stop <= start) for negative step")

        return IndexList[1](len(range(Int(start), Int(stop), Int(step))))
    else:
        return IndexList[1](Int(ceil(abs(stop - start) / abs(step))))
