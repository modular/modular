# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


alias ptr = __mlir_type.`!kgen.pointer<none>`

# CHECK: lit.struct.field field0 : !kgen.pointer<none>

# CHECK: lit.func @"__init__({{.*}}_CI_{{.*}} init_self,
# CHECK-SAME: %fld0: !kgen.pointer<none>


fn bare_ptr(x: ptr):
    fn capture() -> ptr:
        return x
