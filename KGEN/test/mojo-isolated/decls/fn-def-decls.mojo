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


# CHECK-LABEL: lit.func @"variadic_and_kw_only
# CHECK-SAME: (%a: index borrow, %b: index borrow, %args: !kgen.variadic<index> borrow|var, *, %c: index borrow, %d: index borrow = 9)
fn variadic_and_kw_only(a: int, b: int, *args: int, c: int, d: int = `9`):
    pass


# CHECK-LABEL: lit.func @"variadic_arg_after_default
# CHECK-SAME: (%a: index borrow, %b: index borrow = 0, %args: !kgen.variadic<index> borrow|var = *?,
# CHECK-SAME:  *, %c: index borrow, %d: index borrow = 1, %kwargs: {{.*}}|var = *?)
fn variadic_arg_after_default(
    a: int, b: int = `0`, *args: int, c: int, d: int = `1`, **kwargs: int
):
    pass


# CHECK-LABEL: lit.func @"variadic_param_after_default
# CHECK-SAME: <a, b = 0, args: {{.*}} var = *?, *, c, d = 1>()
fn variadic_param_after_default[
    a: int, b: int = `0`, *args: int, c: int, d: int = `1`
]():
    pass
