# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# This tests that the builtin ImplicitlyCopyable and Movable traits are pulled from the
# Mojo Parser Tests stubs and not from the Mojo Stdlib package.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait ToCastInto:
    def test(self):
        ...


# CHECK-LABEL: lit.fn @"trait_downcast_reg_type
def trait_downcast_reg_type[T: TrivialRegisterPassable](x: T):
    # CHECK: lit.var.decl "y" var : !lit.ref<:!TrivialRegisterPassable_ToCastInto downcast(:!TrivialRegisterPassable T), mut *"y`1">
    var y = trait_downcast[ToCastInto](x)
    y.test()


# CHECK-LABEL: lit.fn @"trait_downcast_anytype
def trait_downcast_anytype[T: AnyType](x: T):
    # CHECK: lit.var.decl "y" ref : !lit.ref<!lit.ref<:!AnyType_ToCastInto downcast(:!AnyType T), imm *"x`">, mut *"y`1">
    ref y = trait_downcast[ToCastInto](x)
    y.test()


struct ListIterator[T: Copyable]:
    var t: Self.T


struct List[T: Movable]:
    var t: Self.T

    comptime Iterator = ListIterator[downcast[Self.T, Copyable]]

    def iter(self) -> Self.Iterator:
        pass


def sink[T: ToCastInto](x: T):
    pass


def foo[T: Movable & ToCastInto](l: List[T]):
    var iter = l.iter()
    # Make sure ToCastInfo conformance survives the downcast.
    # CHECK: lit.call @{{.*}}::@"sink{{.*}}
    sink(iter.t)
