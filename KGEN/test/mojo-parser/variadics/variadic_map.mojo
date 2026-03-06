# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s


from std.builtin.variadics import *

comptime ToFloatMapper[From: AnyType] = FloatDyn
comptime AnyToFloat[Ts: Variadic.TypesOfTrait[AnyType]] = Variadic.map_types_to_types[
    From=AnyType, To=type_of(FloatDyn), element_types=Ts, Mapper=ToFloatMapper
]


fn unfoldable[*Ts: AnyType](int_tuple: Tuple[*Ts]) -> Tuple[*AnyToFloat[Ts]]:
    pass


# CHECK: lit.fn @"foldable(::Tuple[::Int, ::Int, ::Int])"
# CHECK-SAME: %__result__: !lit.ref<!lit.struct<#Tuple <:variadic<!AnyType> [!FloatDyn, !FloatDyn, !FloatDyn]>>
fn foldable(int_tuple: Tuple[Int, Int, Int]) -> type_of(unfoldable(int_tuple)):
    pass


# // -----

from std.builtin.variadics import *


struct DepT[T: AnyType]:
    pass


comptime ToDepT[From: Variadic.TypesOfTrait[AnyType], i: Int] = DepT[From[i]]
comptime AnyToDepT[Ts: Variadic.TypesOfTrait[AnyType]] = MapVariadicAndIdxToType[
    From=AnyType, To=AnyType, VariadicType=Ts, Mapper=ToDepT
]


fn unfoldable[*Ts: AnyType](t: Tuple[*Ts]) -> Tuple[*AnyToDepT[Ts]]:
    pass


# CHECK: lit.fn @"foldable(::Tuple[::Int, ::FloatDyn, ::Int])"
# CHECK-SAME: %__result__: !lit.ref<!lit.struct<#Tuple <:variadic<!AnyType>
# CHECK-SAME: [{{.*}}@DepT<:!AnyType !Int>, {{.*}}@DepT<:!AnyType !FloatDyn>, {{.*}}@DepT<:!AnyType !Int>]>
fn foldable(t: Tuple[Int, FloatDyn, Int]) -> type_of(unfoldable(t)):
    pass
