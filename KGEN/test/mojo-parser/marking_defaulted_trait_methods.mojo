# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Basic test for being able to figure out if a function is defaulted.


trait Foo:
    # CHECK: lit.fn @"foo
    # CHECK-SAME: defaultedTraitFn
    fn foo(self) -> Int:
        return Int()

    # CHECK: lit.fn @"foo
    # CHECK-NOT: defaultedTraitFn
    fn foo(self, x: Int) -> Int:
        ...

    # CHECK: lit.fn @"foo
    # CHECK-SAME: defaultedTraitFn
    @staticmethod
    fn foo() -> Int:
        return Int()

    # Make sure we correctly handle cases with params, nesting of braces/parens

    # CHECK: lit.fn @"foo
    # CHECK-SAME: defaultedTraitFn
    fn foo[
        x: Int,
        y: fn[p: Int, f: fn[pp: Int] (x: Int) -> Int] (x: Int, y: Int) -> Int,
    ](self) -> Int:
        return Int()

    # CHECK: lit.fn @"foo
    # CHECK-NOT: defaultedTraitFn
    fn foo[
        x: Int,
        y: fn[p: Int, f: fn[pp: Int] (x: Int) -> Int, z: Int] (
            x: Int, y: Int
        ) -> Int,
    ](self) -> Int:
        ...

    # Make sure special function keywords don't trip up the parsing logic

    # CHECK: lit.fn @"bar
    # CHECK-SAME: defaultedTraitFn
    fn bar(self) capturing raises -> Int:
        return Int()

    # CHECK: lit.fn @"bar
    # CHECK-SAME: attributes {sourceName = "bar", specialFnKind = 0 : i8}
    fn bar(self, x: Int) capturing raises -> Int:
        ...
