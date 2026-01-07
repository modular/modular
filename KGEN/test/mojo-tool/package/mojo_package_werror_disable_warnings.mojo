# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror takes precedence over --disable-warnings for `mojo package`

# RUN: not mojo package --diagnose-missing-doc-strings -Werror --disable-warnings %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=BOTH-FLAGS
# RUN: not mojo package --diagnose-missing-doc-strings -Werror %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=ONLY-WERROR

# BOTH-FLAGS: error: unknown argument 'y' in doc string
# BOTH-FLAGS-NOT: warning: unknown argument 'y' in doc string

# ONLY-WERROR: error: unknown argument 'y' in doc string
# ONLY-WERROR-NOT: warning: unknown argument 'y' in doc string
