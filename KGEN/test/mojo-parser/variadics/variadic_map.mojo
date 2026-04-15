# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


from std.builtin.variadics import *

comptime ToFloatMapper[From: Movable] = FloatDyn
comptime AnyToFloat[*Ts: Movable] = Variadic.map_types_to_types[
    From=Movable, To=type_of(FloatDyn), element_types=Ts, Mapper=ToFloatMapper
]().upcast[Movable]


def unfoldable[*Ts: Movable](int_tuple: Tuple[*Ts]) -> Tuple[*AnyToFloat[*Ts]()]:
    pass


# CHECK: lit.fn @"foldable(::Tuple[::Int, ::Int, ::Int, *?])"
# CHECK-SAME: %__result__: !lit.ref<!lit.struct<#Tuple <:param_list<!Movable> [!FloatDyn, !FloatDyn, !FloatDyn], 
def foldable(int_tuple: Tuple[Int, Int, Int]) -> type_of(unfoldable(int_tuple)):
    pass

struct DepT[T: AnyType](Movable):
    pass


comptime ToDepT[From: Variadic.TypesOfTrait[Movable], i: Int] = DepT[From[i]]
comptime AnyToDepT[
    Ts: Variadic.TypesOfTrait[Movable]
] = MapVariadicAndIdxToType[
    From=Movable, To=Movable, ParamListType=Ts, Mapper=ToDepT
]


def unfoldable2[*Ts: Movable](t: Tuple[*Ts]) -> Tuple[*TypeList[AnyToDepT[Ts.values]]()]:
    pass


# CHECK: lit.fn @"foldable2(::Tuple[::Int, ::FloatDyn, ::Int, *?])"
# CHECK-SAME: %__result__: !lit.ref<!lit.struct<#Tuple <:param_list<!Movable>
# CHECK-SAME: [{{.*}}@DepT<:!AnyType !Int>, {{.*}}@DepT<:!AnyType !FloatDyn>, {{.*}}@DepT<:!AnyType !Int>]
def foldable2(t: Tuple[Int, FloatDyn, Int]) -> type_of(unfoldable2(t)):
    pass
