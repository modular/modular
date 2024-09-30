# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from testing import assert_true


fn test_unit_fail() raises:
    assert_true(False)
    return


fn test_unit_pass():
    return
