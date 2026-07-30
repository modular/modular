# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that `--ignore-deprecated=<name>` suppresses only the named
# declaration's deprecation warning during doc generation, while other
# `@deprecated` declarations still warn. Covers the `mojo doc`/`kgen-doc`
# path used by the `mojo_doc` Bazel rule (a separate CLI/config surface from
# the main parser's `--ignore-deprecated`, see StabilityMarkers.cpp).

# RUN: kgen-doc --ignore-deprecated=ignoredFn %s -o /dev/null 2>&1 | FileCheck %s


@deprecated("ignoredFn is deprecated")
def ignoredFn():
    pass


@deprecated("notIgnoredFn is deprecated")
def notIgnoredFn():
    pass


def caller():
    # CHECK-NOT: warning: ignoredFn is deprecated
    ignoredFn()
    # CHECK: warning: notIgnoredFn is deprecated
    notIgnoredFn()
