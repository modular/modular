# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror takes precedence over --disable-warnings for `mojo run`
# When both flags are passed, warnings should be promoted to errors first,
# then the error should NOT be suppressed (only warnings are suppressed by --disable-warnings)

# RUN: not mojo run --diagnose-missing-doc-strings -Werror --disable-warnings %s 2>&1 | FileCheck %s --check-prefix=BOTH-FLAGS
# RUN: not mojo run --diagnose-missing-doc-strings -Werror %s 2>&1 | FileCheck %s --check-prefix=ONLY-WERROR

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
