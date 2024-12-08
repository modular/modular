# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s


from gpu.host import Dim


# CHECK-LABEL: test_dim
fn test_dim():
    print("== test_dim")

    # CHECK: (x=4, y=1, z=2)
    print(str(Dim(4, 1, 2)))
    # CHECK: (x=4, y=2)
    print(str(Dim(4, 2)))
    # CHECK: (x=4, )
    print(str(Dim(4)))

    # CHECK: (x=4, y=5)
    print(str(Dim((4, 5))))

    # CHECK: (x=4, y=2, z=3)
    print(str(Dim((4, 2, 3))))


fn main():
    test_dim()
