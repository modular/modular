# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s


def unsafe_factorial(next: Int, thusFar: Int) -> Int:
    if next > 1:
        return unsafe_factorial(next - 1, thusFar * next)
    if next <= 1:
        return thusFar
    return 1


def another_unsafe_factorial(next: Int, thusFar: Int) -> Int:
    """This checks to make sure we can properly handle recursive chains."""
    if next > 1:
        return yet_another_unsafe_factorial(next - 1, thusFar * next)
    if next <= 1:
        return thusFar
    return 1


def yet_another_unsafe_factorial(next: Int, thusFar: Int) -> Int:
    return another_unsafe_factorial(next, thusFar)


def main():
    var x = unsafe_factorial(3, 1)
    var y = another_unsafe_factorial(3, 1)
    # CHECK: 6
    print(x)
    # CHECK: 6
    print(y)
