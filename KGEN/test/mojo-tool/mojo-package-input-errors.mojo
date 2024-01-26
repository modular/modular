# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo package %S/test_package -debug-level hello 2>&1 | FileCheck %s -check-prefix DEBUG_LEVEL
# DEBUG_LEVEL: invalid debug level 'hello', expected one of: `none` (the default value), `line-tables`, or `full`

# RUN: not mojo package %S/test_package -o test.mojonot 2>&1 | FileCheck %s -check-prefix OUTPUT_FILE
# OUTPUT_FILE: output path must have a '.mojopkg' or '.📦' extension
