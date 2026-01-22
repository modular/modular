# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test that @stable decorator is recognized on functions and methods.

# CHECK: lit.fn @"stable_function()"
# CHECK-SAME: hasStableDecorator
@stable
fn stable_function():
    pass


# CHECK: lit.fn @"unstable_function()"
# CHECK-NOT: hasStableDecorator
# CHECK-SAME: sourceName
fn unstable_function():
    pass


# The struct must be @stable to allow @stable members inside it.
# CHECK: lit.struct.decl @TestStruct
# CHECK-SAME: hasStableDecorator
@stable
struct TestStruct:
    # CHECK: lit.fn @"stable_method{{.*}}TestStruct)"
    # CHECK-SAME: hasStableDecorator
    @stable
    fn stable_method(self):
        pass

    # CHECK: lit.fn @"unstable_method{{.*}}TestStruct)"
    # CHECK-NOT: hasStableDecorator
    # CHECK-SAME: sourceName
    fn unstable_method(self):
        pass

    # CHECK: lit.fn @"stable_static()"
    # CHECK-SAME: hasStableDecorator
    @stable
    @staticmethod
    fn stable_static():
        pass
