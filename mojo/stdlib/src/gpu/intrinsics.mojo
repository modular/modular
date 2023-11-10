# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs intrinsics operations."""

from .ptx_assembly import ptx_assembly
from .sys import is_sm_greater_equal
from memory.unsafe import DTypePointer, bitcast
from sys.intrinsics import llvm_intrinsic
from sys.info import alignof


# ===----------------------------------------------------------------------===#
# ldg
# ===----------------------------------------------------------------------===#


@always_inline
fn ldg[type: DType](x: DTypePointer[type]) -> SIMD[type, 1]:
    """Load a register variable from global state space via non-coherent cache.
    """

    alias alignment = Int32(alignof[SIMD[type, 1]]())

    @parameter
    if type == DType.uint8 or type == DType.int8:
        return bitcast[type, 1](
            llvm_intrinsic["llvm.nvvm.ldg.global.i.i8", SIMD[DType.int8, 1]](
                x.bitcast[DType.int8](), alignment
            )
        )
    elif type == DType.uint16 or type == DType.int16:
        return bitcast[type, 1](
            llvm_intrinsic["llvm.nvvm.ldg.global.i.i16", SIMD[DType.int16, 1]](
                x.bitcast[DType.int16](), alignment
            )
        )
    elif type == DType.uint32 or type == DType.int32:
        return bitcast[type, 1](
            llvm_intrinsic["llvm.nvvm.ldg.global.i.i32", SIMD[DType.int32, 1]](
                x.bitcast[DType.int32](), alignment
            )
        )
    elif type == DType.uint64 or type == DType.int64:
        return bitcast[type, 1](
            llvm_intrinsic["llvm.nvvm.ldg.global.i.i64", SIMD[DType.int64, 1]](
                x.bitcast[DType.int64](), alignment
            )
        )
    elif type == DType.float32:
        return llvm_intrinsic["llvm.nvvm.ldg.global.f.f32", SIMD[type, 1]](
            x, alignment
        )
    elif type == DType.float64:
        return llvm_intrinsic["llvm.nvvm.ldg.global.f.f64", SIMD[type, 1]](
            x, alignment
        )
    else:
        constrained[False, "Unhandled DType"]()
        return 0


# ===----------------------------------------------------------------------===#
# warpgroup_reg
# ===----------------------------------------------------------------------===#


fn warpgroup_reg_alloc[count: Int]():
    """Provides a hint to the system to update the maximum number of per-thread
    registers owned by the executing warp to the value specified by the
    imm-reg-count operand.

    Used to request additional registers such that the absolute per-thread
    maximum register count is increased from its current value to imm-reg-count.
    """

    @parameter
    if is_sm_greater_equal[90]():
        ptx_assembly[
            "setmaxnreg.inc.sync.aligned.u32 $0", NoneType, constraints="i"
        ](UInt32(count))


fn warpgroup_reg_dealloc[count: Int]():
    """Provides a hint to the system to update the maximum number of per-thread
    registers owned by the executing warp to the value specified by the
    imm-reg-count operand.

    Used to release extra registers such that the absolute per-thread maximum
    register count is reduced from its current value to imm-reg-count.
    """

    @parameter
    if is_sm_greater_equal[90]():
        ptx_assembly[
            "setmaxnreg.dec.sync.aligned.u32 $0", NoneType, constraints="i"
        ](UInt32(count))
