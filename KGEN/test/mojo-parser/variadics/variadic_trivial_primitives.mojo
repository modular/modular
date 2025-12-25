# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s

from std.builtin.variadics import *

# CHECK-LABEL: lit.alias.decl *"T`0x": meta<!lit.struct<#Tuple <:variadic<!AnyType>
# CHECH-SAME: [!Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int]
comptime T = Tuple[*VariadicSplat[Int, 10]]


comptime VA_SIZE[*Ts: AnyType] = variadic_size(Ts)
# CHECK: lit.alias.decl *"Folded`{{.*}}": !Int = <{3}>
comptime Folded = VA_SIZE[Int, Int, Int]


# CHECK-LABEL: lit.fn @"foo
fn foo(
    t1: Tuple[Int, Int, Int], t2: Tuple[FloatDyn, FloatDyn, FloatDyn]
) -> Tuple[
    # CHECK: %__result__: !lit.ref<{{.*}}> [!Int, !Int, !Int, !FloatDyn, !FloatDyn, !FloatDyn]>
    *VariadicConcat[type_of(t1).element_types, type_of(t2).element_types]
]:
    pass
