# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | kgen-opt -verify-parameters | FileCheck %s


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[A:.*]]: !Int, |>
# CHECK: lit.struct.decl @"fn{{.*}}"
# CHECK: lit.func @"__init__{{.*}}"{{.*}}<[[a:.*]][a]: !Int, |>(%self: !lit.ref<mut !Int1, {{.*}}> init_self, %impl: !lit.ref<mut @"{{.*}}::@"`_CI_{{.*}}"<:!Int [[a]]>{{.*}}> owned_in_mem, |) -> !kgen.none {{.*}}specialFnKind = 2 : i8
# CHECK: lit.func @"{{.*}}_copyinit_`_CI_{{.*}}"{{.*}}<[[a:.*]][a]: !Int, |>(%other: !kgen.pointer<none>, |) -> !kgen.pointer<none> {{.*}}specialFnKind = 0 : i8
# CHECK: lit.func @"{{.*}}_dtor_`_CI_{{.*}}"<[[a:.*]][a]: !Int, |>(%self: !kgen.pointer<none>, |) -> !kgen.none {{.*}}specialFnKind = 0 : i8
# CHECK: lit.func @"{{.*}}_call_`_CI_{{.*}}"<[[a:.*]][a]: !Int, |>(%0[*""]: !kgen.pointer<none> borrow, |, %x: !Int borrow) -> !Int {{.*}}specialFnKind = 0 : i8}
fn parameter_capture[a: Int](c: Int) -> fn (x: Int) escaping -> Int:
    fn p_capture(x: Int) escaping -> Int:
        return c + a + x

    return p_capture
