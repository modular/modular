# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror takes precedence over --disable-warnings for `mojo run`
# When both flags are passed, warnings should be promoted to errors first,
# then the error should NOT be suppressed (only warnings are suppressed by --disable-warnings)

# RUN: not mojo run -Werror --disable-warnings %s 2>&1 | FileCheck %s --check-prefix=BOTH-FLAGS
# RUN: not mojo run -Werror %s 2>&1 | FileCheck %s --check-prefix=ONLY-WERROR

# BOTH-FLAGS: error: assignment to 'foo' was never used
# BOTH-FLAGS-NOT: warning: assignment to 'foo' was never used

# ONLY-WERROR: error: assignment to 'foo' was never used
# ONLY-WERROR-NOT: warning: assignment to 'foo' was never used


def main():
    var foo = 1
