# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs intrinsics operations."""

from sys.intrinsics import _mlirtype_is_eq


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
    ptx_assembly[
        "setmaxnreg.dec.sync.aligned.u32 $0", NoneType, constraints="i"
    ](UInt32(count))


# ===----------------------------------------------------------------------===#
# ptx_assembly
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn ptx_assembly[
    asm: StringLiteral,
    result_type: AnyType,
    /,
    has_side_effect: Bool = True,
]() -> result_type:
    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = "".value,
                hasSideEffects = __mlir_attr.unit,
            ]()
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = "".value,
            ]()
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = "".value,
                hasSideEffects = __mlir_attr.unit,
            ]()
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = "".value,
            ]()


@always_inline("nodebug")
fn ptx_assembly[
    asm: StringLiteral,
    result_type: AnyType,
    arg0_type: AnyType,
    /,
    constraints: StringLiteral = "r",
    has_side_effect: Bool = True,
](arg0: arg0_type) -> result_type:
    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0)


@always_inline("nodebug")
fn ptx_assembly[
    asm: StringLiteral,
    result_type: AnyType,
    arg0_type: AnyType,
    arg1_type: AnyType,
    /,
    constraints: StringLiteral = "r,r",
    has_side_effect: Bool = True,
](arg0: arg0_type, arg1: arg1_type) -> result_type:
    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1)
