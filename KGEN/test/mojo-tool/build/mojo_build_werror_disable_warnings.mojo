# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror takes precedence over --disable-warnings for `mojo build`

# RUN: not mojo build --diagnose-missing-doc-strings -Werror --disable-warnings %s -o %t 2>&1 | FileCheck %s --check-prefix=BOTH-FLAGS
# RUN: not mojo build --diagnose-missing-doc-strings -Werror %s -o %t 2>&1 | FileCheck %s --check-prefix=ONLY-WERROR

# BOTH-FLAGS: error: unknown argument 'y' in doc string
# BOTH-FLAGS-NOT: warning: unknown argument 'y' in doc string

# ONLY-WERROR: error: unknown argument 'y' in doc string
# ONLY-WERROR-NOT: warning: unknown argument 'y' in doc string


fn f(x: Int):
    """This is a function with an invalid doc string.

    Args:
        y: This argument doesn't appear in the argument list.
    """
    pass


def main():
    f(42)
