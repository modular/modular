# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror converts warnings to errors for `mojo doc`

# RUN: not mojo doc --diagnose-missing-doc-strings -Werror %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-WERROR

# CHECK-WERROR: error: unknown argument 'y' in doc string
# CHECK-WERROR-NOT: warning: unknown argument 'y' in doc string

# Test that --validate-doc-strings is a deprecated alias for -Werror.
# RUN: not mojo doc --diagnose-missing-doc-strings --validate-doc-strings %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-DEPRECATED
# CHECK-DEPRECATED: warning: --validate-doc-strings is deprecated, use -Werror instead
# CHECK-DEPRECATED: error: unknown argument 'y' in doc string


fn f(x: Int):
    """This is a function with an invalid doc string.

    Args:
        y: This argument doesn't appear in the argument list.
    """
    pass
