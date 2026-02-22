# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test that @stable(since="version") stores the version string in the IR.


# CHECK: lit.fn @"fn_since_version()"
# CHECK-SAME: hasStableDecorator
# CHECK-SAME: stableSinceVersion = "1.0"
@stable(since="1.0")
fn fn_since_version():
    pass


# CHECK: lit.fn @"fn_since_relaxed_semver()"
# CHECK-SAME: hasStableDecorator
# CHECK-SAME: stableSinceVersion = "2.1.3rc1"
@stable(since="2.1.3rc1")
fn fn_since_relaxed_semver():
    pass


# Bare @stable must NOT produce a stableSinceVersion attribute.
# CHECK: lit.fn @"fn_bare_stable()"
# CHECK-SAME: hasStableDecorator
# CHECK-NOT: stableSinceVersion
@stable
fn fn_bare_stable():
    pass


# @stable(since=) also works on structs and their members.
# CHECK: lit.struct.decl @StableStruct
# CHECK-SAME: hasStableDecorator
# CHECK-SAME: stableSinceVersion = "1.0"
@stable(since="1.0")
struct StableStruct:
    # CHECK: lit.fn @"method{{.*}}StableStruct)"
    # CHECK-SAME: hasStableDecorator
    # CHECK-SAME: stableSinceVersion = "1.1"
    @stable(since="1.1")
    fn method(self):
        pass
