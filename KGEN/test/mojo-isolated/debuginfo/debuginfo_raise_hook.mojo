# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -debug-level full -O0 -mlir-print-debuginfo %s | FileCheck %s

# CHECK: %2 = lit.call @stdlib::@builtin::@error::@"__mojo_debugger_raise_hook()"()
# CHECK-NEXT: lit.raise


fn foo() raises:
    raise "exception"
