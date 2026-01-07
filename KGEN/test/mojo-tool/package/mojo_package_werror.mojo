# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that -Werror converts warnings to errors for `mojo package`

# RUN: not mojo package --diagnose-missing-doc-strings -Werror %S/test_package_werror 2>&1 | FileCheck %s

# CHECK: error: unknown argument 'y' in doc string
# CHECK-NOT: warning: unknown argument 'y' in doc string
