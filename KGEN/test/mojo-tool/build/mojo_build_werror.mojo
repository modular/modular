# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror converts warnings to errors for `mojo build`

# RUN: not mojo build --diagnose-missing-doc-strings -Werror %s -o %t 2>&1 | FileCheck %s

# CHECK: error: unknown argument 'y' in doc string
# CHECK-NOT: warning: unknown argument 'y' in doc string


fn f(x: Int):
    """This is a function with an invalid doc string.

    Args:
        y: This argument doesn't appear in the argument list.
    """
    pass


def main():
    f(42)
