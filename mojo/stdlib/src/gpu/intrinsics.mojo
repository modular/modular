# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs intrinsics operations."""

from sys._assembly import inlined_assembly
from sys.info import alignof
from sys.intrinsics import llvm_intrinsic

from memory import UnsafePointer
from memory.unsafe import bitcast

from .sys import is_sm_greater_equal

# ===----------------------------------------------------------------------===#
# ldg
# ===----------------------------------------------------------------------===#


@always_inline
fn ldg[type: DType](x: UnsafePointer[Scalar[type]]) -> Scalar[type]:
    """Load a register variable from global state space via non-coherent cache.
    """

    alias alignment = Int32(alignof[Scalar[type]]())

    @parameter
    if type is DType.uint8 or type is DType.int8:
        return bitcast[type, 1](
            llvm_intrinsic["llvm.nvvm.ldg.global.i.i8", Int8](
                x.bitcast[Int8](), alignment
            )
        )
    elif type is DType.uint16 or type is DType.int16:
        return bitcast[type, 1](
            llvm_intrinsic["llvm.nvvm.ldg.global.i.i16", Int16](
                x.bitcast[Int16](), alignment
            )
        )
    elif type is DType.uint32 or type is DType.int32:
        return bitcast[type, 1](
            llvm_intrinsic["llvm.nvvm.ldg.global.i.i32", Int32](
                x.bitcast[Int32](), alignment
            )
        )
    elif type is DType.uint64 or type is DType.int64:
        return bitcast[type, 1](
            llvm_intrinsic["llvm.nvvm.ldg.global.i.i64", Int64](
                x.bitcast[Int64](), alignment
            )
        )
    elif type is DType.float32:
        return llvm_intrinsic["llvm.nvvm.ldg.global.f.f32", Scalar[type]](
            x, alignment
        )
    elif type is DType.float64:
        return llvm_intrinsic["llvm.nvvm.ldg.global.f.f64", Scalar[type]](
            x, alignment
        )
    elif type is DType.bfloat16:
        return llvm_intrinsic["llvm.nvvm.ldg.global.f.bf16", Scalar[type]](
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
        inlined_assembly[
            "setmaxnreg.inc.sync.aligned.u32 $0;", NoneType, constraints="i"
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
        inlined_assembly[
            "setmaxnreg.dec.sync.aligned.u32 $0;", NoneType, constraints="i"
        ](UInt32(count))


# ===----------------------------------------------------------------------===#
# Convertion
# ===----------------------------------------------------------------------===#


@always_inline
fn convert[
    src_type: DType, dst_type: DType, width: Int
](src: SIMD[src_type, width]) -> SIMD[dst_type, width]:
    """Convert data types with different precisions.
    This is for conversions not covered by `cast`."""

    constrained[
        src_type is DType.float32 and dst_type is DType.bfloat16 and width == 2,
        "Only support 2xfp32 to 2xbf16 conversion",
    ]()

    var bf16x2_as_uint32 = inlined_assembly[
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        UInt32,
        Float32,
        Float32,
        constraints="=r,f,f",
    ](src[1].cast[DType.float32](), src[0].cast[DType.float32]())

    # Reinterpret cast uint32 to 2 bf16.
    var ptr = UnsafePointer.address_of(bf16x2_as_uint32).bitcast[
        SIMD[dst_type, width]
    ]()

    var result = ptr[0]
    _ = bf16x2_as_uint32
    return result
