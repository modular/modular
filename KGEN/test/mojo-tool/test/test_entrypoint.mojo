# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo test "%s" --no-execute --keep-entrypoint --entrypoint-path "%t-entrypoint"
# RUN: MOJO_TEST_RUN_ALL=1 not "%t-entrypoint" | FileCheck %s --check-prefix=CHECK-RUN-ALL

# CHECK-RUN-ALL: test_entrypoint.mojo:{{.*}}:{{.*}}: AssertionError: condition was unexpectedly False

from testing import assert_true


fn `test_unit.failure`() raises:
    assert_true(False)
    return


fn test_unit_pass():
    return
