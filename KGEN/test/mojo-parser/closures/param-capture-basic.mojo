# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | kgen-opt -verify-parameters | FileCheck %s

trait Destructable:
    fn __del__(owned self, /):
       ...

trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
       ...

trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
       ...


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[A:.*]]: !Int, |>
# CHECK: lit.struct.decl @"_CW_{{.*}}"
# CHECK: lit.func @"__init__{{.*}}"{{.*}}<[[a:.*]][a]: !Int, |>(%self[self]: !lit.ref<mut !wrapper, {{.*}}> init_self, %impl[impl]: !kgen.pointer<@"{{.*}}::@"`_CI_{{.*}}"<:!Int [[a]]>{{.*}}> owned_in_mem, |) -> !kgen.none {{.*}}specialFnKind = 2 : i8
# CHECK: lit.func @"{{.*}}_copyinit_`_CI_{{.*}}"{{.*}}<[[a:.*]][a]: !Int, |>(%arg[ptrToImpl]: !kgen.pointer<pointer<none>> borrow, %other[other]: !kgen.pointer<none> borrow, |) -> !kgen.none {{.*}}specialFnKind = 0 : i8
# CHECK: lit.func @"{{.*}}_dtor_`_CI_{{.*}}"<[[a:.*]][a]: !Int, |>(%self[self]: !kgen.pointer<none>, |) -> !kgen.none {{.*}}specialFnKind = 0 : i8
# CHECK: lit.func @"{{.*}}_call_`_CI_{{.*}}"<[[a:.*]][a]: !Int, |>(%0[*""]: !kgen.pointer<none> borrow, |, %x[x]: !Int borrow) -> !Int {{.*}}specialFnKind = 0 : i8}
fn parameter_capture[a: Int](c: Int) -> fn (x: Int) escaping -> Int:
    fn p_capture(x: Int) escaping -> Int:
        return c + a + x

    return p_capture
