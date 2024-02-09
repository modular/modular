# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from utils.variant import Variant
from kernel_utils.dynamic_tuple import *

alias General = Variant[Int, Float32, String]


# FIXME: This is a horrible hack around Mojo's lack or proper trait inheritance
struct GeneralDelegate(ElementDelegate):
    @always_inline
    @staticmethod
    fn is_equal[T: CollectionElement](a: Variant[T], b: Variant[T]) -> Bool:
        if a.isa[Int]() and b.isa[Int]():
            return a.get[Int]() == b.get[Int]()
        elif a.isa[Float32]() and b.isa[Float32]():
            return a.get[Float32]() == b.get[Float32]()
        elif a.isa[String]() and b.isa[String]():
            return a.get[String]() == b.get[String]()
        else:
            trap(Error("Unexpected data type."))
        return False

    @always_inline
    @staticmethod
    fn to_string[T: CollectionElement](a: Variant[T]) -> String:
        if a.isa[General]():
            let v = a.get[General]()
            if v.isa[Int]():
                return v.get[Int]()
            if v.isa[Float32]():
                return v.get[Float32]()
            if v.isa[String]():
                return v.get[String]()
        trap(Error("Unexpected data type."))
        return "#"


alias GeneralTupleBase = DynamicTupleBase[General, GeneralDelegate]
alias GeneralElement = GeneralTupleBase.Element
alias GeneralTuple = DynamicTuple[General, GeneralDelegate]


def main():
    print("Hello General!")

    var gt = GeneralTuple(
        General(1),
        GeneralTuple(
            General(Float32(3.5)),
            General(String("Mojo")),
        ),
    )

    # CHECK: 1
    # CHECK: 3.5
    # CHECK: Mojo
    # CHECK: (1, (3.5, Mojo))
    print(gt[0].value().get[Int]())
    print(gt[1][0].value().get[Float32]())
    print(gt[1][1].value().get[String]())
    print(gt)

    # CHECK: (7, (3.5, Mojo))
    gt[0] = General(7)
    print(gt)
