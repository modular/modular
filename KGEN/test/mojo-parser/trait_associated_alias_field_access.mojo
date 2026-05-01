# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# Regression test: accessing a trait-associated type alias through a generic
# struct was incorrectly rejected because get_witness was not folded when
# computing the struct field type for IR emission and verification.


trait HasOutput:
    comptime Output: TrivialRegisterPassable


@fieldwise_init
struct IntImpl(HasOutput, TrivialRegisterPassable):
    comptime Output = Int


@fieldwise_init
struct Wrap[T: HasOutput](TrivialRegisterPassable):
    var value: Self.T.Output


# CHECK-LABEL: lit.fn @"generic_access()
def generic_access():
    var w = Wrap[IntImpl](Int())
    # CHECK: lit.ref.struct.ger {{.*}}[value]
    # CHECK-SAME: sugar_member_alias({{.*}}, "Output", !Int)
    var _v = w.value
