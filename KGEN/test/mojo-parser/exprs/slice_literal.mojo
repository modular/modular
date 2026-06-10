# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

from std.builtin.builtin_slice import ContiguousSlice


struct Variant[*Ts: Movable]:
    @implicit
    def __init__[T: Movable](out self, var value: T):
        pass


# The variadic parameter resolves to `[::Int, ::ContiguousSlice]`, proving the
# integer literal and the slice literal each select the matching `Variant` arm.
# CHECK: lit.fn @"foo[KGENParamList[slice_literal::Variant[::Int, ::ContiguousSlice, *?]]
# CHECK-SAME: sourceName = "foo"
def foo[*elts: Variant[Int, ContiguousSlice]]() -> NoneType:
    pass


# def foo[*elts: Variant[Int, Slice]]() -> NoneType:
#     pass

# IMPORTANT: the ^ code would likely lead to an runtime error, to make it work, we need to
#
# 1st, make `def Variant.__init__[T: Movable](out self, var value: T):` have a where clause
#      to reject `T: ContiguousSlice`.
# 2nd, add an implicit conversion:
#      `def Variant.__init__(out self, var value: ContiguousSlice) where Ts.contains(Slice) :`


# CHECK: lit.fn @"test()"
# CHECK: lit.call tail @slice_literal::@"foo[
# CHECK-SAME: @std::@builtin::@builtin_slice::@ContiguousSlice::@"__init__(::Optional[::Int],::Optional[::Int],::NoneType,::NoneType)"
def test():
    foo[0, :, 0, :]()
