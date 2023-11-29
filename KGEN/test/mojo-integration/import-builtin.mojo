# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s | FileCheck %s


# CHECK-LABEL: lit.func @"import_of_import
# CHECK-SAME: @"$builtin"::@"$simd"::@SIMD<
fn import_of_import(arg: Float64):
    pass


import builtin


# CHECK-LABEL: lit.func @"test_function_calls($builtin::$int::Int)"
fn test_function_calls(arg: builtin.int.Int):
    pass
