# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -I %S/inputs %s 4 | FileCheck %s

from closure import printIt, defineIt
from std.sys import argv


def aThing(y: Int):
    def myclosure(x: Int) {var} -> Int:
        return y + x

    printIt[type_of(myclosure)](myclosure, y)
    defineIt(y)


def main() raises:
    # CHECK: 8
    # CHECK: 8
    var x = atol(argv()[1])
    aThing(x)
