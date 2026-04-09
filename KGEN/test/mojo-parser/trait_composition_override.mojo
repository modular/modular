# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Regression test for MOCO-3332: trait composition should accept valid code
# where a child trait re-declares a parent's abstract requirement and both
# traits appear in a composition constraint (e.g. struct S[T: A & B]() where
# B(A)).

# RUN: %parse-mojo-isolated %s --mojo-disable-builtins | FileCheck %s


trait A:
    def foo(self):
        ...


trait B(A):
    def foo(self):
        ...


# CHECK: lit.struct.decl @S<T: !A_B>
struct S[T: A & B]():
    var _value: Self.T
