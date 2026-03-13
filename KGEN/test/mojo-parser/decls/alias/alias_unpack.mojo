# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# TODO: to make `var : T1 = 1` work, we need to make tuple::__method__ parser
# foldable.

# works without parens
comptime _, (b, c) = 1, (2, 3.0)


# TODO(MOCO-2764)
# alias T1, (_, T3) = (Int, (Int, FloatDyn))


def use[T: AnyType](t: T):
    pass


def foo():
    # CHECK: kgen.param.constant: !Int
    # CHECK: kgen.param.constant: !FloatDyn
    use(b)
    use(c)

    # var x: T1
    # var y: T3
