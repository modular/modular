# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mkdir -p %t.closure-dir
# RUN: mojo package %S/inputs/closure -o %t.closure-dir/closure.mojopkg
# RUN: mojo -I %t.closure-dir %s 4 | FileCheck %s
# RUN: kgen-opt %t.closure-dir/closure.mojopkg | FileCheck %s -check-prefix=CHECK-PACK

# CHECK-PACK: lit.trait.decl @"fn(x: Int) -> Int"
# CHECK-PACK: definesClosure

from closure import printIt, defineIt
from sys import argv


fn aThing(y: Int):
    fn myclosure(x: Int) unified {var} -> Int:
        return y + x

    printIt[type_of(myclosure)](myclosure, y)
    defineIt(y)


def main():
    # CHECK: 8
    # CHECK: 8
    var x = atol(argv()[1])
    aThing(x)
