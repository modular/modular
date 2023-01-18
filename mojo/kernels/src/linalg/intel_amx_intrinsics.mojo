# ===----------------------------------------------------------------------===#
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------===#
#
# This file contains wrappers around Intel AMX intrinsics. See
# https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&techs=AMX
#
# ===----------------------------------------------------------------------===#

from SIMD import SIMD
from Int import Int


struct __tile:
    """An AMX tile representation"""

    var buf: __mlir_type[`!pop.array<1024, si32>`]
    var rows: SIMD[1, __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`]
    var cols: SIMD[1, __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`]


fn to_si8(
    val: Int,
) -> SIMD[1, __mlir_attr.`#kgen.dtype.constant<si8> : !kgen.dtype`]:
    """Converts an input integer to an si8 value"""
    return __mlir_op.`pop.cast`[_type : __mlir_type.`!pop.scalar<si8>`](
        val.value
    )


fn _tile_dpbssd(dst: Int, a: Int, b: Int):
    """
    Compute dot-product of bytes in tiles with a source/destination accumulator.
    Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with
    corresponding signed 8-bit integers in b, producing 4 intermediate 32-bit
    results. Sum these 4 results with the corresponding 32-bit integer in dst,
    and store the 32-bit result back to tile dst.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_dpbssd
    """
    var dst_si8 = to_si8(dst)
    var a_si8 = to_si8(dst)
    var b_si8 = to_si8(dst)
    __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.tdpbssd",
        _type:[],
    ](dst_si8.value, a_si8.value, b_si8.value)
