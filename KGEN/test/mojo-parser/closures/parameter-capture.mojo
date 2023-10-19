# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s

# CHECK: lit.struct.decl @"_CI_{{.*}}"<p0[p0]: !Int, |>
# CHECK: lit.func @"__init__{{.*}}_CW_{{.*}}"<[[p0:init.*]][[[p0]]]: !Int, |>(%self[self]: !kgen.pointer<!escaping> init_self, %impl[impl]: !kgen.pointer<@"{{.*}}::@"_CI_{{.*}}"<p0: !Int = [[p0]]>> borrow_in_mem, |) -> !kgen.none attributes {specialFnKind = 2 : i8} {
# CHECK: lit.func @"{{.*}}_copyinit__CI_{{.*}}"<[[copyp0:copy.*]][[[copyp0]]]: !Int, |>(%arg[ptrToImpl]: !kgen.pointer<pointer<none>> borrow, %other[other]: !kgen.pointer<none> borrow_in_mem, |) -> !kgen.none attributes {specialFnKind = 0 : i8} {
# CHECK: lit.func @"{{.*}}_dtor__CI_{{.*}}"<[[delp0:del.*]][[[delp0]]]: !Int, |>(%self[self]: !kgen.pointer<none>, |) -> !kgen.none attributes {specialFnKind = 0 : i8} {
# CHECK: lit.func @"{{.*}}_call__CI_{{.*}}"<[[callp0:call.*]][[[callp0]]]: !Int, |>(%0[*""]: !kgen.pointer<none> borrow_in_mem, |, %x[x]: !Int borrow) -> !Int attributes {specialFnKind = 0 : i8} {
fn parameter_capture[a: Int](c: Int) -> fn (x: Int) escaping -> Int:
   fn p_capture(x: Int) escaping -> Int:
       return c + a + x

   return p_capture

