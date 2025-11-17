# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s


from stdlib.builtin.variadics import *

comptime ToFloatMapper[From: AnyType] = FloatDyn
comptime AnyToFloat[Ts: VariadicOf[AnyType]] = MapTypeToType[
    To = type_of(FloatDyn), Variadic=Ts, Mapper=ToFloatMapper
]


fn unfoldable[*Ts: AnyType](int_tuple: Tuple[*Ts]) -> Tuple[*AnyToFloat[Ts]]:
    pass


# CHECK: lit.fn @"foldable(::Tuple[::Int, ::Int, ::Int])"
# CHECK-SAME: %__result__: !lit.ref<@stdlib::@builtin::@stubs::@Tuple<:variadic<!AnyType> [!FloatDyn, !FloatDyn, !FloatDyn]>
fn foldable(int_tuple: Tuple[Int, Int, Int]) -> type_of(unfoldable(int_tuple)):
    pass


# // -----

from stdlib.builtin.variadics import *


struct DepT[T: AnyType]:
    pass


comptime ToDepT[From: VariadicOf[AnyType], i: __mlir_type.index] = DepT[From[i]]
comptime AnyToDepT[Ts: VariadicOf[AnyType]] = MapVariadicAndIdxToType[
    To=AnyType, Variadic=Ts, Mapper=ToDepT
]


fn unfoldable[*Ts: AnyType](t: Tuple[*Ts]) -> Tuple[*AnyToDepT[Ts]]:
    pass


# CHECK: lit.fn @"foldable(::Tuple[::Int, ::FloatDyn, ::Int])"
# CHECK-SAME: %__result__: !lit.ref<@stdlib::@builtin::@stubs::@Tuple<:variadic<!AnyType>
# CHECK-SAME: [@variadic_map::@DepT<:!AnyType !Int>, @variadic_map::@DepT<:!AnyType !FloatDyn>, @variadic_map::@DepT<:!AnyType !Int>]>
fn foldable(t: Tuple[Int, FloatDyn, Int]) -> type_of(unfoldable(t)):
    pass
