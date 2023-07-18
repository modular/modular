# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo package %mojo_cpu_build_arch %S/test_package --name mojo-test-pkg -debug-level hello 2>&1 | FileCheck %s -check-prefix DEBUG_LEVEL
# DEBUG_LEVEL: invalid debug level 'hello', expected one of: `none` (the default value), `line-tables`, or `full`
