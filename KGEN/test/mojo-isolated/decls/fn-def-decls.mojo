# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages %s | FileCheck %s


# CHECK-LABEL: lit.func @"empty_def()"() -> !kgen.none
# CHECK: lit.end_func
fn empty_def():
    pass


# CHECK-LABEL: lit.func @"slash
# CHECK-SAME: (%a: index borrow, |, %b: index borrow)
fn slash(a: Int, /, b: Int):
    pass


# CHECK-LABEL: lit.func @"trailing_slash
# CHECK-SAME: (%a: index borrow, |)
fn trailing_slash(a: Int, /):
    pass


# CHECK-LABEL: lit.func @"star
# CHECK-SAME: (%a: index borrow, *, %b: index borrow)
fn star(a: Int, *, b: Int):
    pass


# CHECK-LABEL: lit.func @"leading_star
# CHECK-SAME: (*, %a: index borrow)
fn leading_star(*, a: Int):
    pass


# CHECK-LABEL: lit.func @"star_and_slash
# CHECK-LABEL: (%a: index borrow, |, *, %b: index borrow)
fn star_and_slash(a: Int, /, *, b: Int):
    pass


# CHECK-LABEL: lit.func @"star_and_slash_2
# CHECK-SAME: (%a: index borrow, |, %b: index borrow, *, %c: index borrow)
fn star_and_slash_2(a: Int, /, b: Int, *, c: Int):
    pass


# CHECK-LABEL: lit.func @"default_args
# CHECK-SAME: (%a: index borrow, %b: index borrow = 8, *, %c: index borrow, %d: index borrow = 9)
fn default_args(a: Int, b: Int = `8`, *, c: Int, d: Int = `9`):
    pass
