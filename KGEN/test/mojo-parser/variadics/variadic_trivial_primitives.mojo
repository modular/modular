# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s

# CHECK-LABEL: lit.alias.decl *"T`0x": meta<!lit.struct<#Tuple <:param_list<!AnyType_Movable>
# CHECK-SAME: [!Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int]
comptime T = Tuple[*TypeList.splat[Trait=Movable, 10, Int]()]


comptime VA_SIZE[*Ts: AnyType] = Ts.size
# CHECK: lit.alias.decl *"Folded`{{.*}}": !alias_Int1 = <sugar_member_alias{{.*}}rebind(:!Int {:scalar<index> 3})))>
comptime Folded = VA_SIZE[Int, Int, Int]

comptime AddOne[i: Int]: Int = i + 1

# Tabulate: [0, 1, 2, 3, 4] from index identity
comptime TabulateIndices = ParameterList.tabulate[5, AddOne]
# CHECK: lit.alias.decl *"TabulateIndices`{{.*}}:param_list<!Int> [{:scalar<index> 1}, {:scalar<index> 2}, {:scalar<index> 3}, {:scalar<index> 4}, {:scalar<index> 5}]>>

# `TypeList.of` infers the element trait from its arguments, so no explicit
# `Trait=` keyword argument is required.
# CHECK-LABEL: lit.alias.decl *"TLOf{{[^"]*}}": meta<!lit.struct<#TypeList <:meta<!AnyType> !AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDeletable_Movable_RegisterPassable_TrivialRegisterPassable
# CHECK-SAME: :param_list<!AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDeletable_Movable_RegisterPassable_TrivialRegisterPassable> [!Int, !Bool]
comptime TLOf = TypeList.of[Int, Bool]
