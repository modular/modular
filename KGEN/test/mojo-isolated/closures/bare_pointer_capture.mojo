# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


alias ptr = __mlir_type.`!kgen.pointer<none>`

# CHECK: lit.struct.field field0 : !kgen.pointer<none>

# CHECK: lit.fn @"__init__({{.*}}%fld0: !kgen.pointer<none>,
# CHECK-SAME: byref_result)


fn bare_ptr(x: ptr):
    fn capture() -> ptr:
        return x
