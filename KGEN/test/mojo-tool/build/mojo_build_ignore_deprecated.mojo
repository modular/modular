# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that `mojo build --ignore-deprecated=<name>` (the GNU `=` form, as used
# by BUILD.bazel `copts`) suppresses only the named deprecation warning, while
# an unrelated `@deprecated` declaration still warns.

# RUN: %mojo-build-no-werror --ignore-deprecated=ignoredFn %s 2>&1 | FileCheck %s


# CHECK-NOT: warning: ignoredFn is deprecated
@deprecated("ignoredFn is deprecated")
def ignoredFn():
    pass


# CHECK: warning: notIgnoredFn is deprecated
@deprecated("notIgnoredFn is deprecated")
def notIgnoredFn():
    pass


def main():
    ignoredFn()
    notIgnoredFn()
    return
