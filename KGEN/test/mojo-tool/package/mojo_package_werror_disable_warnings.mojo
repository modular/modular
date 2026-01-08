# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror takes precedence over --disable-warnings for `mojo package`

# RUN: not mojo package -Werror --disable-warnings %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=BOTH-FLAGS
# RUN: not mojo package -Werror %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=ONLY-WERROR

# BOTH-FLAGS: error: assignment to 'foo' was never used
# BOTH-FLAGS-NOT: warning: assignment to 'foo' was never used

# ONLY-WERROR: error: assignment to 'foo' was never used
# ONLY-WERROR-NOT: warning: assignment to 'foo' was never used
