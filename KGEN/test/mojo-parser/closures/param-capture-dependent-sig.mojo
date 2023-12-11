# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | kgen-opt -verify-parameters | FileCheck %s

# CHECK: lit.struct.decl @"_CW_{{.*}}"<p0, |>

alias Int = __mlir_type.index

trait Destructable:
    fn __del__(owned self, /):
       ...

trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
       ...

trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
       ...

@value
@register_passable
struct Foo[B: Int]:
    pass


# CHECK-LABEL: lit.func @"take_closure
# CHECK-SAME: <[[C_TYPE:.*c_type]][c_type]>
# CHECK-SAME: (%arg[closure]: {{.*}}<[[C_TYPE]]>
fn take_closure[c_type: Int](closure: fn (z: Foo[c_type]) escaping -> None):
    closure(Foo[c_type]())
    pass
