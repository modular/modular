# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s | FileCheck %s


fn take_func_without_arg_name[f: fn(Int) -> None](): pass

fn func_with_arg_name(a: Int): pass

# COM: Issue https://github.com/modularml/mojo/issues/1307
# COM: Test that functions with defaults can be passed where no defaults are expected
fn take_func_without_default[f: fn(a: Int) -> None](): pass

fn func_with_default(a: Int = `0`): pass

# CHECK-LABEL: lit.func @"test_passing_funcs
fn test_passing_funcs():
    # CHECK: lit.call @{{.*}}::@"take_func_without_arg_name{{.*}}"<
    # CHECK-SAME: :!lit.signature<(index borrow, |) -> !kgen.none> rebind(:!lit.signature<("a": index borrow) -> !kgen.none>
    take_func_without_arg_name[func_with_arg_name]()

    # CHECK: lit.call @{{.*}}::@"take_func_without_default{{.*}}"<
    # CHECK-SAME: :!lit.signature<("a": index borrow) -> !kgen.none> rebind(:!lit.signature<("a": index borrow = 0) -> !kgen.none>
    take_func_without_default[func_with_default]()
