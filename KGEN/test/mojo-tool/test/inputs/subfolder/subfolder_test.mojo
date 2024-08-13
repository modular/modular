# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from testing import assert_true


fn test_fn() raises:
    assert_true(True)


fn `test_backtick.name.required`():
    return


fn `test_backtick.fails`() raises:
    assert_true(False)


fn test_fails() raises:
    assert_true(False)
