# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.alias.decl *"gen_idx_to_int{{.*}}": non_struct_type = <!lit.generator<<"Idx": !Int>!{{.*}}>>
comptime gen_idx_to_int = __generator_type[Idx: Int] Int

# CHECK: lit.alias.decl *"gen_empty{{.*}}": non_struct_type = <!lit.generator<<>!{{.*}}>>
comptime gen_empty = __generator_type Int

# CHECK: lit.alias.decl *"gen_two_params{{.*}}": non_struct_type = <!lit.generator<<"T": !AnyType, "U": !AnyType>!kgen.param<:!AnyType *(0,0)>>>
comptime gen_two_params = __generator_type[T: AnyType, U: AnyType] T

# Nested: outer param flows into the generator body type.
# CHECK: lit.alias.decl *"gen_with_outer{{.*}}": !lit.generator<<"ToT": !AnyType>non_struct_type> = <#kgen.gen<!lit.generator<<"Idx": !Int>!kgen.param<:!AnyType *(1,0)>>>>
comptime gen_with_outer[ToT: AnyType] = __generator_type[Idx: Int] ToT

# CHECK: lit.fn @"use_as_param_type[{{.*}}"<Mapper: !lit.generator<<"Idx": !Int>!{{.*}}>>
def use_as_param_type[
    Mapper: __generator_type[Idx: Int] Int
]():
    pass
