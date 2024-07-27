# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs intrinsics operations."""

from sys._assembly import inlined_assembly
from sys.info import alignof, bitwidthof
from sys.intrinsics import llvm_intrinsic
from builtin.dtype import _int_type_of_width

from memory import UnsafePointer
from memory.unsafe import bitcast

from .sys import is_sm_greater_equal

# ===----------------------------------------------------------------------===#
# ldg
# ===----------------------------------------------------------------------===#


@always_inline
fn _bitwidthof_str[type: DType]() -> StringLiteral:
    alias bitwidth = bitwidthof[type]()

    @parameter
    if bitwidth == 8:
        return "8"
    elif bitwidth == 16:
        return "16"
    elif bitwidth == 32:
        return "32"
    elif bitwidth == 64:
        return "64"
    constrained[False, "invalid dtype"]()
    return "invalid"


@always_inline
fn ldg[type: DType](x: UnsafePointer[Scalar[type]]) -> Scalar[type]:
    """Load a register variable from global state space via non-coherent cache.
    """
    constrained[type.is_numeric(), "the type must be numeric"]()

    alias prefix = "llvm.nvvm.ldg.global."
    alias suffix = _bitwidthof_str[type]()

    alias alignment = Int32(alignof[type]())

    @parameter
    if type.is_integral():
        alias integral_type = _int_type_of_width[bitwidthof[type]()]()
        return bitcast[type, 1](
            llvm_intrinsic[prefix + "i.i" + suffix, Scalar[integral_type]](
                x.bitcast[integral_type](), alignment
            )
        )

    constrained[type.is_floating_point(), "the type must be floating point"]()
    return llvm_intrinsic[prefix + "f.f" + suffix, Scalar[type]](x, alignment)


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
        constraints="=r,f,f",
        has_side_effect=False,
    ](rebind[Float32](src[1]), rebind[Float32](src[0]))

    return rebind[SIMD[dst_type, width]](
        bitcast[DType.bfloat16, 2](bf16x2_as_uint32)
    )
