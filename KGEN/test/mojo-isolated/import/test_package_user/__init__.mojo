# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This file tests importing and using a package at the same level as the
# parent package.

import test_package

fn using_test_package():
  test_package.module.function()
