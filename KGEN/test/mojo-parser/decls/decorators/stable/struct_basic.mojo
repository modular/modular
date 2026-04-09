# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test that @stable decorator is recognized on structs and sets the
# hasStableDecorator attribute in the IR.


# CHECK: lit.struct.decl @StableStruct
# CHECK-SAME: hasStableDecorator
@stable
struct StableStruct:
    pass


# CHECK: lit.struct.decl @UnstableStruct
# CHECK-NOT: hasStableDecorator
# CHECK-SAME: sourceName
struct UnstableStruct:
    pass


# Verify @stable works when combined with other decorators.
# The choice of @fieldwise_init is arbitrary - any struct decorator
# works. This test ensures decorator composition doesn't break @stable.
# CHECK: lit.struct.decl @StableWithOtherDecorators
# CHECK-SAME: hasStableDecorator
@stable
@fieldwise_init
struct StableWithOtherDecorators(RegisterPassable):
    pass
