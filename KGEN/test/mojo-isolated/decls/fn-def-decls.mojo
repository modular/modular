# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.func @"empty_def()"() -> !kgen.none
# CHECK: lit.end_func
fn empty_def():
    pass


# CHECK-LABEL: lit.func @"slash
# CHECK-SAME: (%a: index borrow, |, %b: index borrow)
fn slash(a: int, /, b: int):
    pass


# CHECK-LABEL: lit.func @"trailing_slash
# CHECK-SAME: (%a: index borrow, |)
fn trailing_slash(a: int, /):
    pass


# CHECK-LABEL: lit.func @"star
# CHECK-SAME: (%a: index borrow, *, %b: index borrow)
fn star(a: int, *, b: int):
    pass


# CHECK-LABEL: lit.func @"leading_star
# CHECK-SAME: (*, %a: index borrow)
fn leading_star(*, a: int):
    pass


# CHECK-LABEL: lit.func @"star_and_slash
# CHECK-LABEL: (%a: index borrow, |, *, %b: index borrow)
fn star_and_slash(a: int, /, *, b: int):
    pass


# CHECK-LABEL: lit.func @"star_and_slash_2
# CHECK-SAME: (%a: index borrow, |, %b: index borrow, *, %c: index borrow)
fn star_and_slash_2(a: int, /, b: int, *, c: int):
    pass


# CHECK-LABEL: lit.func @"default_args
# CHECK-SAME: (%a: index borrow, %b: index borrow = 8, *, %c: index borrow, %d: index borrow = 9)
fn default_args(a: int, b: int = `8`, *, c: int, d: int = `9`):
    pass
