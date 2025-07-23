# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


from io.io import _printf


fn raiseErrorIf(cond: Bool) raises -> Int:
    if cond:
        raise Error()
    return 0


def implicitlyPropagate(cond: Bool) -> Int:
    return raiseErrorIf(cond)


fn main():
    # CHECK: first success
    try:
        var a = implicitlyPropagate(False)
        print("first success")
    except e0:
        print("first had an error")

    # CHECK-NEXT: second had an error
    try:
        var b = implicitlyPropagate(True)
        print("second success")
    except e1:
        print("second had an error")

    # CHECK-NEXT: third: 0
    _printf["third: "]()
    try:
        print(raiseErrorIf(False))
    except e2:
        print("bad!")

    # CHECK-NEXT: done
    print("done")
