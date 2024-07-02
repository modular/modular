# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


fn take_func_without_arg_name[f: fn (int) -> None]():
    pass


fn func_with_arg_name(a: int):
    pass


# COM: Issue https://github.com/modularml/mojo/issues/1307
# COM: Test that functions with defaults can be passed where no defaults are expected
fn take_func_without_default[f: fn (a: int) -> None]():
    pass


fn func_with_default(a: int = `0`):
    pass


# CHECK-LABEL: lit.func @"test_passing_funcs
fn test_passing_funcs():
    # CHECK: lit.call @{{.*}}::@"take_func_without_arg_name{{.*}}"<
    # CHECK-SAME: :!lit.signature<(index, |) -> !kgen.none> rebind(:!lit.signature<("a": index) -> !kgen.none>
    take_func_without_arg_name[func_with_arg_name]()

    # CHECK: lit.call @{{.*}}::@"take_func_without_default{{.*}}"<
    # CHECK-SAME: :!lit.signature<("a": index) -> !kgen.none> rebind(:!lit.signature<("a": index = 0) -> !kgen.none>
    take_func_without_default[func_with_default]()
