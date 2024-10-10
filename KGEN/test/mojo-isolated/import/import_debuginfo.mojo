# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -verify-diagnostics -mlir-print-debuginfo -I=%S %s | FileCheck %s

# Test import of a module, and we properly allow import of an imported decl.

from imported_module import *


# CHECK-LABEL: lit.func @"foo
fn foo():
    imported_fn()


# CHECK-LABEL: lit.file_module @imported_module

# CHECK-LABEL: lit.func @"imported_fn
# CHECK: } loc(#[[LOC_IMPORTED_FN:.+]])

# CHECK: #[[FILE_IMPORTED_MODULE:.+]] = #debuginfo.file<"{{.*}}/imported_module.mojo"
# CHECK: #[[SP_IMPORTED_FN:.+]] = #debuginfo.subprogram<{{.*}}scope = #[[FILE_IMPORTED_MODULE]]{{.*}}linkageName = "imported_fn()"{{.*}}file = #[[FILE_IMPORTED_MODULE]]
# CHECK: #[[LOC_IMPORTED_FN]] = loc(fused<#[[SP_IMPORTED_FN]]
