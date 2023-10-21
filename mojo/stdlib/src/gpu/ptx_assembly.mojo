# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes the ptx_assembly function."""

from sys.intrinsics import _mlirtype_is_eq


# ===----------------------------------------------------------------------===#
# 0-arg
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


# ===----------------------------------------------------------------------===#
# 1-arg
# ===----------------------------------------------------------------------===#


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


# ===----------------------------------------------------------------------===#
# 2-arg
# ===----------------------------------------------------------------------===#


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


# ===----------------------------------------------------------------------===#
# 3-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn ptx_assembly[
    asm: StringLiteral,
    result_type: AnyType,
    arg0_type: AnyType,
    arg1_type: AnyType,
    arg2_type: AnyType,
    /,
    constraints: StringLiteral = "r,r,r",
    has_side_effect: Bool = True,
](arg0: arg0_type, arg1: arg1_type, arg2: arg2_type) -> result_type:
    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2)


# ===----------------------------------------------------------------------===#
# 4-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn ptx_assembly[
    asm: StringLiteral,
    result_type: AnyType,
    arg0_type: AnyType,
    arg1_type: AnyType,
    arg2_type: AnyType,
    arg3_type: AnyType,
    /,
    constraints: StringLiteral = "r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type, arg1: arg1_type, arg2: arg2_type, arg3: arg3_type
) -> result_type:
    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3)


# ===----------------------------------------------------------------------===#
# 5-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn ptx_assembly[
    asm: StringLiteral,
    result_type: AnyType,
    arg0_type: AnyType,
    arg1_type: AnyType,
    arg2_type: AnyType,
    arg3_type: AnyType,
    arg4_type: AnyType,
    /,
    constraints: StringLiteral = "r,r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type,
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
    arg4: arg4_type,
) -> result_type:
    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg3)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4)


# ===----------------------------------------------------------------------===#
# 6-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn ptx_assembly[
    asm: StringLiteral,
    result_type: AnyType,
    arg0_type: AnyType,
    arg1_type: AnyType,
    arg2_type: AnyType,
    arg3_type: AnyType,
    arg4_type: AnyType,
    arg5_type: AnyType,
    /,
    constraints: StringLiteral = "r,r,r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type,
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
    arg4: arg4_type,
    arg5: arg5_type,
) -> result_type:
    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg3, arg5)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5)


# ===----------------------------------------------------------------------===#
# 7-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn ptx_assembly[
    asm: StringLiteral,
    result_type: AnyType,
    arg0_type: AnyType,
    arg1_type: AnyType,
    arg2_type: AnyType,
    arg3_type: AnyType,
    arg4_type: AnyType,
    arg5_type: AnyType,
    arg6_type: AnyType,
    /,
    constraints: StringLiteral = "r,r,r,r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type,
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
    arg4: arg4_type,
    arg5: arg5_type,
    arg6: arg6_type,
) -> result_type:
    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg3, arg5, arg6)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6)


# ===----------------------------------------------------------------------===#
# 8-arg
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn ptx_assembly[
    asm: StringLiteral,
    result_type: AnyType,
    arg0_type: AnyType,
    arg1_type: AnyType,
    arg2_type: AnyType,
    arg3_type: AnyType,
    arg4_type: AnyType,
    arg5_type: AnyType,
    arg6_type: AnyType,
    arg7_type: AnyType,
    /,
    constraints: StringLiteral = "r,r,r,r,r,r,r,r",
    has_side_effect: Bool = True,
](
    arg0: arg0_type,
    arg1: arg1_type,
    arg2: arg2_type,
    arg3: arg3_type,
    arg4: arg4_type,
    arg5: arg5_type,
    arg6: arg6_type,
    arg7: arg7_type,
) -> result_type:
    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():

        @parameter
        if has_side_effect:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        else:
            __mlir_op.`pop.inline_asm`[
                _type=None,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        return rebind[result_type](None)
    else:

        @parameter
        if has_side_effect:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
                hasSideEffects = __mlir_attr.unit,
            ](arg0, arg1, arg2, arg3, arg3, arg5, arg6, arg7)
        else:
            return __mlir_op.`pop.inline_asm`[
                _type=result_type,
                assembly = asm.value,
                constraints = constraints.value,
            ](arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
