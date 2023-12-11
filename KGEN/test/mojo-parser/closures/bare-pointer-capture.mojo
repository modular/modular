# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | FileCheck %s

trait Destructable:
    fn __del__(owned self, /):
       ...

trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
       ...

trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
       ...

alias ptr = __mlir_type.`!kgen.pointer<none>`

# CHECK: lit.struct.field field0 : !kgen.pointer<none>

# CHECK: lit.func @"__init__({{.*}}_CI_{{.*}} init_self,
# CHECK-SAME: %fld0[fld0]: !kgen.pointer<none>


fn bare_ptr(x: ptr):
    fn capture() escaping -> ptr:
        return x
