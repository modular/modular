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

# CHECK: !AnyType_ExplicitlyCopyable_ImplicitlyCopyable_Movable_UnknownDestructibility = !lit.trait<
# CHECK-SAME: @stdlib::@builtin::@stubs::@AnyType,
# CHECK-SAME: @stdlib::@builtin::@stubs::@ExplicitlyCopyable,
# CHECK-SAME: @stdlib::@builtin::@stubs::@ImplicitlyCopyable,
# CHECK-SAME: @stdlib::@builtin::@stubs::@Movable,
# CHECK-SAME: @stdlib::@builtin::@stubs::@UnknownDestructibility>


# CHECK: lit.struct.decl @BoxedInt(!AnyType_ExplicitlyCopyable_ImplicitlyCopyable_Movable_UnknownDestructibility)
@fieldwise_init
struct BoxedInt(ExplicitlyCopyable, ImplicitlyCopyable, Movable):
    var value: Index
