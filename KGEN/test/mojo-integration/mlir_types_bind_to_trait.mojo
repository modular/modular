# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full -O0 %s 2 3 | FileCheck %s

comptime TestFn = fn (x: Int) raises -> Tuple[Bool, Int]


@no_inline
fn printIt[T: AnyType & ImplicitlyCopyable & Movable]():
    if T.__del__is_trivial:
        print("del trivial")
    if T.__moveinit__is_trivial:
        print("move trivial")
    if T.__copyinit__is_trivial:
        print("copy trivial")


def main():
    # CHECK: del trivial
    # CHECK: move trivial
    # CHECK: copy trivial
    printIt[TestFn]()
