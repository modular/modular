# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s
# COM: Capture inside a nested escaping closure.


@value
struct MemType:
    var x: Int

    fn __add__(self, rhs: MemType) -> MemType:
        return MemType(rhs.x + self.x)

    fn __add__(self, rhs: Int) -> MemType:
        return MemType(self.x + rhs)


# COM: Check that the parameter capture "A" is forwarded to the outer escaping closure
# CHECK: lit.struct.decl @"`_CI_{{.*}}escaping1"<[[A:.*]]: !Int, |>


# COM: Check that the parameter capture "A" is forwarded to the outer escaping closure
# CHECK: lit.struct.decl @"`_CI_{{.*}}escaping0"<[[A]]: !Int, |>
fn makes_escaping_closure[
    A: Int
](m: MemType) -> fn (n: MemType) escaping -> MemType:
    fn myclosure(n: MemType) -> MemType:
        fn nested_nested(k: MemType, l: MemType) -> MemType:
            return n + k + A

        return nested_nested(n, m)

    return myclosure
