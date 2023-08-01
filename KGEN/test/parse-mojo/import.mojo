# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics -split-input-file -I=%S %s | FileCheck %s

# Test import of a module, and we properly allow import of an imported decl.

from imported_module import *

# CHECK-LABEL: lit.func @"import_of_import
# CHECK-SAME: @"$SIMD"::@SIMD<
fn import_of_import(arg: Float64):
  pass


# // -----

# Test import of a package.

from test_package.module import function
from test_package.test_nested_package.module import nested_function
from test_package import *
import Builtin

# CHECK-LABEL: lit.func @"test_function_calls($Builtin::$Int::Int)"
# CHECK:  kgen.call @"$test_package"::@"$module"::@"function()"
# CHECK:  kgen.call @"$test_package"::@"$test_nested_package"::@"$module"::@"nested_function()"
# CHECK:  kgen.call @"$test_package"::@"$__init__"::@"method_defined_in_init()"()

# CHECK-LABEL: lit.package @"$test_package"
# CHECK:  lit.file_module @"$module"
# CHECK:    lit.func @"function()"
# CHECK:      lit.func @"call_nested_function()"
# CHECK:        kgen.call @"$test_package"::@"$test_nested_package"::@"$module"::@"nested_function()"
# CHECK:  lit.package @"$test_nested_package"
# CHECK:    lit.file_module @"$module"
# CHECK:      lit.func @"nested_function()"

fn test_function_calls(arg: Builtin.Int.Int):
  function()
  nested_function()
  method_defined_in_init()
