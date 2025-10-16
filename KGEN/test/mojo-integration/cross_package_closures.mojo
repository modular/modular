# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -I %S/inputs %s 4 | FileCheck %s

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
