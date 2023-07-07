# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics -I=%S %s

from test_package.module import function
from test_package.test_nested_package.module import nested_function

# CHECK: lit.package @"$test_package"
# CHECK:  lit.file_module @"$module"
# CHECK:    lit.func @"function()"
# CHECK:  lit.package @"$test_nested_package"
# CHECK:    lit.file_module @"$module"
# CHECK:      lit.func @"nested_function()"
# CHECK:      lit.func @"call_nested_function()"
# CHECK:        kgen.call @"$test_package"::@"$test_nested_package"::@"$module"::@"nested_function()"

# CHECK-LABEL: lit.func @"test_function_calls()"
# CHECK:  kgen.call @"$test_package"::@"$module"::@"function()"
# CHECK:  kgen.call @"$test_package"::@"$test_nested_package"::@"$module"::@"nested_function()"

fn test_function_calls():
  function()
  nested_function()
