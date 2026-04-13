# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s

from std.builtin.variadics import *

# CHECK-LABEL: lit.alias.decl *"T`0x": meta<!lit.struct<#Tuple <:param_list<!Movable>
# CHECH-SAME: [!Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int]
comptime T = Tuple[*TypeList.splat[Trait=Movable, 10, Int]()]


comptime VA_SIZE[*Ts: AnyType] = Ts.size
# CHECK: lit.alias.decl *"Folded`{{.*}}": !Int = <sugar_member_alias{{.*}}{3})>
comptime Folded = VA_SIZE[Int, Int, Int]

comptime AddOne[i: Int]: Int = i + 1

# Tabulate: [0, 1, 2, 3, 4] from index identity
comptime TabulateIndices = ParameterList.tabulate[5, AddOne]
# CHECK: lit.alias.decl *"TabulateIndices`{{.*}}:param_list<!Int> [{1}, {2}, {3}, {4}, {5}]>>


# CHECK-LABEL: lit.fn @"foo
def foo(
    t1: Tuple[Int, Int, Int], t2: Tuple[FloatDyn, FloatDyn, FloatDyn]
) -> Tuple[
    # CHECK: %__result__: !lit.ref<{{.*}}> [!Int, !Int, !Int, !FloatDyn, !FloatDyn, !FloatDyn]>
    *TypeList[Variadic.concat_types[T=Movable, type_of(t1).element_types.values, type_of(t2).element_types.values]]()
]:
    pass
