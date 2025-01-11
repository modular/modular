# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[A:.*]]: !Int, |>
# CHECK: lit.struct.decl @"fn{{.*}}"
# CHECK: lit.fn @"__init__{{.*}}"<?, a: !Int>[{{.*}}](%impl: !lit.ref<@{{.*}}::@"`_CI_{{.*}}"<:!Int a>{{.*}}> owned_in_mem, |, ?, %self: !lit.ref<!Int1, mut{{.*}}> byref_result) -> !kgen.none {{.*}}specialFnKind = 2 : i8
# CHECK: lit.fn @"{{.*}}_copyinit_`_CI_{{.*}}"{{.*}}<?, a: !Int>(%other: !kgen.pointer<none>, |) -> !kgen.pointer<none> {{.*}}specialFnKind = 0 : i8
# CHECK: lit.fn @"{{.*}}_dtor_`_CI_{{.*}}"<?, a: !Int>(%self: !kgen.pointer<none>, |) -> !kgen.none {{.*}}specialFnKind = 0 : i8
# CHECK: lit.fn @"{{.*}}_call_`_CI_{{.*}}"<?, a: !Int>(%0[*""]: !kgen.pointer<none>, |, %x: !Int) -> !Int {{.*}}specialFnKind = 0 : i8}
fn parameter_capture[a: Int](c: Int) -> fn (x: Int) escaping -> Int:
    fn p_capture(x: Int) -> Int:
        return c + a + x

    return p_capture
