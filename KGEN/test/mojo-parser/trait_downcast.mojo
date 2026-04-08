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
    # CHECK: lit.var.decl "y" var : !lit.ref<:!ToCastInto downcast(:!TrivialRegisterPassable T), mut *"y`1">
    var y = trait_downcast[ToCastInto](x)
    y.test()


# CHECK-LABEL: lit.fn @"trait_downcast_anytype
def trait_downcast_anytype[T: AnyType](x: T):
    # CHECK = lit.var.decl "y" ref : !lit.ref<!lit.ref<:!ToCastInto downcast(:!AnyType T), {{.*}}">, {{.*}}">
    ref y = trait_downcast[ToCastInto](x)
    y.test()
