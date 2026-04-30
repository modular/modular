# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s 2 3 | FileCheck %s

from std.utils.type_functions import ConditionalType
from std.sys.intrinsics import _type_is_eq


struct ConditionalArraySize[T: AnyType]:
    comptime Size: Int = 0 if conforms_to(Self.T, ImplicitlyCopyable) else 1
    var array: InlineArray[Int, Self.Size]

    def __init__(out self):
        self.array = {fill = 0}


def generic(conditional: ConditionalArraySize) raises:
    print(conditional.array.size)


struct ConditionalValueType[T: AnyType]:
    comptime Type = ConditionalType[
        Trait=Defaultable & ImplicitlyDestructible,
        If=conforms_to(Self.T, ImplicitlyCopyable),
        Then=List[Int],
        Else=String,
    ]

    var value: Self.Type

    def __init__(out self):
        self.value = {}


def main() raises:
    comptime IsImplicitlyCopyable: AnyType = Int
    comptime IsNotImplicitlyCopyable: AnyType = List[Int]

    var size_0 = ConditionalArraySize[IsImplicitlyCopyable]()
    # CHECK: 0
    generic(size_0)
    # CHECK: 0
    print(size_0.array.size)

    var size_1 = ConditionalArraySize[IsNotImplicitlyCopyable]()
    # CHECK: 1
    generic(size_1)
    # CHECK: 1
    print(size_1.array.size)

    var _list_int = ConditionalValueType[IsImplicitlyCopyable]()
    # CHECK: True
    print(_type_is_eq[List[Int], type_of(_list_int.value)]())

    var _string = ConditionalValueType[IsNotImplicitlyCopyable]()
    # CHECK: True
    print(_type_is_eq[String, type_of(_string.value)]())
