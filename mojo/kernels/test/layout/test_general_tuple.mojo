# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from os import abort

from layout.dynamic_tuple import *
from testing import assert_equal, assert_not_equal

from utils.variant import Variant

alias General = Variant[Int, Float32, String]


# FIXME: This is a horrible hack around Mojo's lack or proper trait inheritance
struct GeneralDelegate(ElementDelegate):
    @always_inline
    @staticmethod
    fn is_equal[T: CollectionElement](va: Variant[T], vb: Variant[T]) -> Bool:
        if va.isa[General]() and vb.isa[General]():
            var a = va[General]
            var b = vb[General]
            if a.isa[Int]() and b.isa[Int]():
                return a[Int] == b[Int]
            elif a.isa[Float32]() and b.isa[Float32]():
                return a[Float32] == b[Float32]
            elif a.isa[String]() and b.isa[String]():
                return a[String] == b[String]
        abort("Unexpected data type.")
        return False

    @always_inline
    @staticmethod
    fn format_element_to[
        T: CollectionElement, W: Writer
    ](inout writer: W, a: Variant[T]):
        if not a.isa[General]():
            abort("Unexpected data type.")

        var v = a[General]

        if v.isa[Int]():
            writer.write(v[Int])
        if v.isa[Float32]():
            # FIXME(#37912):
            #   Implement a Mojo float formatting algorithm that can be used
            #   format floating point values even on GPU, and use it here.
            writer.write("<UnsupportedFormattedFloat:#37912>")
        if v.isa[String]():
            writer.write(v[String])


alias GeneralTupleBase = DynamicTupleBase[General, GeneralDelegate]
alias GeneralElement = GeneralTupleBase.Element
alias GeneralTuple = DynamicTuple[General, GeneralDelegate]


# CHECK-LABEL: test_tuple_general
fn test_tuple_general() raises:
    print("== test_tuple_general")

    # Test General tuple operations

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
    # CHECK: (1, (<UnsupportedFormattedFloat:#37912>, Mojo))
    print(gt[0].value()[Int])
    print(gt[1][0].value()[Float32])
    print(gt[1][1].value()[String])
    print(gt)

    # CHECK: (7, (<UnsupportedFormattedFloat:#37912>, Mojo))
    gt[0] = General(7)
    print(gt)

    # Test General tuple comparison

    var gt2 = GeneralTuple(
        General(2),
        GeneralTuple(
            General(Float32(3.5)),
            General(String("Mojo")),
        ),
    )
    assert_not_equal(gt, gt2)

    gt2[0] = General(7)
    assert_equal(gt, gt2)


def main():
    test_tuple_general()
