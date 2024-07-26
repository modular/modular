# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -S --gen-lib %s | FileCheck %s


# Test for MOCO-956
# https://linear.app/modularml/issue/MOCO-956/[bug]-segfault-on-struct-recursive-methods
@value
struct FactorialComputer:
    # CHECK: kgen.generator @{{.*}}::FactorialComputer::compute_method{{.*}}_REMOVED_ARG"(%arg0: !pop.scalar<ui8>) -> !pop.scalar<ui8>
    fn compute_method(self, depth: UInt8) -> UInt8:
        if depth == 0:
            return 1
        return depth * self.compute_method(depth - 1)


# CHECK: kgen.generator @{{.*}}::compute_unusedPost{{.*}}_REMOVED_ARG"(%arg0: !pop.scalar<ui8>) -> !pop.scalar<ui8>
fn compute_unusedPost(depth: UInt8, unused: Bool) -> UInt8:
    if depth == 0:
        return 1
    return depth * compute_unusedPost(depth - 1, unused)


# CHECK: kgen.generator @{{.*}}::compute_unusedPre{{.*}}_REMOVED_ARG"(%arg0: !pop.scalar<ui8>) -> !pop.scalar<ui8>
fn compute_unusedPre(unused: Bool, depth: UInt8) -> UInt8:
    if depth == 0:
        return 1
    return depth * compute_unusedPre(unused, depth - 1)


fn main():
    var a = FactorialComputer().compute_method(2)
    print(a)
    var b = compute_unusedPost(2, False)
    print(b)
    var c = compute_unusedPre(False, 2)
    print(c)
