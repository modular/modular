# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# This tests that the builtin Copyable and Movable traits are pulled from the
# Mojo Parser Tests stubs and not from the Mojo Stdlib package.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s | FileCheck %s


# CHECK: lit.struct.decl @BoxedInt(
# CHECK-SAME: @"$stdlib"::@"$builtin"::@"$stubs"::@Copyable>, trait<@"$stdlib"::@"$builtin"::@"$stubs"::@Movable>
@value
struct BoxedInt:
    var value: Int
