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

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK: lit.struct.decl @BoxedInt(!UnknownDestructibility, !Copyable, !AnyType[!Copyable], !Movable, !ExplicitlyCopyable)
@value
struct BoxedInt:
    var value: Index
