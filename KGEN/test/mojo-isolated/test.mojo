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

# CHECK: !AnyType_Copyable_ImplicitlyCopyable_Movable_UnknownDestructibility = !lit.trait<
# CHECK-SAME: @std::@builtin::@stubs::@AnyType,
# CHECK-SAME: @std::@builtin::@stubs::@Copyable,
# CHECK-SAME: @std::@builtin::@stubs::@ImplicitlyCopyable,
# CHECK-SAME: @std::@builtin::@stubs::@Movable,
# CHECK-SAME: @std::@builtin::@stubs::@UnknownDestructibility>


# CHECK: lit.struct.decl @BoxedInt(!AnyType_Copyable_ImplicitlyCopyable_Movable_UnknownDestructibility)
@fieldwise_init
struct BoxedInt(ImplicitlyCopyable):
    var value: Int
