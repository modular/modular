# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Verify that assert is a no-op when assertions are disabled.
# %mojo-no-debug-no-assert does not pass -D ASSERT=all, so assertions with
# the default assert_mode="none" are not enabled.
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s


def main():
    # These would trap if assertions were enabled, but they should be no-ops.
    assert False
    assert False, "this should be a no-op"

    # CHECK: assertions disabled, all passed
    print("assertions disabled, all passed")
