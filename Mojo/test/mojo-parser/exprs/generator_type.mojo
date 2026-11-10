# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


# The call's inferred origins bind the signature that `function[0]` produced,
# not the `[index: Int]` generator `function[0]` already fully bound.
# CHECK-LABEL: lit.fn @"call_inferred_sig_through_generator_type
# CHECK: lit.call tail[{{.*}}bind_params({{.*}}bind_params({{.*}} function, :!Int {:scalar<index> 0}), :origin<false> *"args.origin{{.*}}, {{.*}} *"args.origin{{.*}})]
def call_inferred_sig_through_generator_type[
    Args: TypeList[Trait=AnyType, ...],
    function: __generator_type[index: Int] def(*args: *Args) thin -> None,
](*args: *Args):
    function[0](*args)


# Same shape with a non-function inner generator: `g[0]` instantiates to
# `[n: Int] Int`, so `[1]` binds that, not `g`.
# CHECK-LABEL: lit.fn @"bind_through_nested_generator_type
def bind_through_nested_generator_type[
    g: __generator_type[index: Int] __generator_type[n: Int] Int
]() -> Int:
    comptime h = g[0]
    comptime x = h[1]
    return x
