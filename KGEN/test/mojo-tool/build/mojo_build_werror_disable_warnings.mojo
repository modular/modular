# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror takes precedence over --disable-warnings for `mojo build`

# RUN: not mojo build -Werror --disable-warnings %s -o %t 2>&1 | FileCheck %s --check-prefix=BOTH-FLAGS
# RUN: not mojo build -Werror %s -o %t 2>&1 | FileCheck %s --check-prefix=ONLY-WERROR

# BOTH-FLAGS: error: assignment to 'foo' was never used
# BOTH-FLAGS-NOT: warning: assignment to 'foo' was never used

# ONLY-WERROR: error: assignment to 'foo' was never used
# ONLY-WERROR-NOT: warning: assignment to 'foo' was never used


def main():
    var foo = 1
