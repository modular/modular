# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# REQUIRES: has_cuda_device
# RUN: %mojo -debug-level full %s | FileCheck %s


from gpu.host import Dim


# CHECK-LABEL: test_dim
fn test_dim():
    print("== test_dim")

    # CHECK: (x=4, y=1, z=2)
    print(Dim(4, 1, 2).__str__())
    # CHECK: (x=4, y=2)
    print(Dim(4, 2).__str__())
    # CHECK: (x=4, )
    print(Dim(4).__str__())

    # CHECK: (x=4, y=5)
    print(Dim((4, 5)).__str__())

    # CHECK: (x=4, y=2, z=3)
    print(Dim((4, 2, 3)).__str__())


fn main():
    test_dim()
