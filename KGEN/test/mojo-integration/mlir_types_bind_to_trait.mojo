# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full -O0 %s 2 3 | FileCheck %s

comptime Testdef = def(x: Int) thin raises -> Tuple[Bool, Int]


@no_inline
def printIt[T: AnyType & ImplicitlyCopyable]():
    if T.__del__is_trivial:
        print("del trivial")
    if T.__move_ctor_is_trivial:
        print("move trivial")
    if T.__copy_ctor_is_trivial:
        print("copy trivial")


def main() raises:
    # CHECK: del trivial
    # CHECK: move trivial
    # CHECK: copy trivial
    printIt[Testdef]()
