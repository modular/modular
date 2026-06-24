# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This file tests importing and using a package at the same level as the
# parent package. The function is imported explicitly from the submodule — a
# bare `import test_package` would not expose `test_package.module`.

from test_package.module import function


def using_test_package():
    function()
