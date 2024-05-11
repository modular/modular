# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s

from max.driver import Device, CPUDescriptor
from tensor import TensorSpec
from utils.index import Index
from testing import assert_equal


def main():
    var dev = Device(CPUDescriptor(numa_id=2))
    assert_equal(str(dev), "Device(type=CPU,numa_id=2)")

    var dt1 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )
    assert_equal(
        str(dt1), "DeviceTensor(Device(type=CPU,numa_id=2),Spec(2x2xfloat32))"
    )

    var dt2 = dev.allocate(
        TensorSpec(DType.float32, 3, 2),
    )
    assert_equal(
        str(dt2), "DeviceTensor(Device(type=CPU,numa_id=2),Spec(3x2xfloat32))"
    )

    var tensor = (dt1^).get_tensor[DType.float32]()
    tensor[Index(0, 0)] = 0
    tensor[Index(0, 1)] = 1
    tensor[Index(1, 0)] = 2
    tensor[Index(1, 1)] = 3
    print(tensor)
