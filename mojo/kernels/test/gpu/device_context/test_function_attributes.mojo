# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu import ThreadIdx
from memory import stack_allocation, UnsafePointer
from gpu.host import DeviceContext
from gpu.host.function import Attribute
from testing import *


# CHECK-LABEL: test_function_attributes
def test_function_attributes():
    fn kernel(x: UnsafePointer[Int]):
        x[0] = ThreadIdx.x()

    # CHECK: tid.x

    with DeviceContext() as ctx:
        var func = ctx.compile_function[kernel, dump_ptx=True]()
        assert_equal(
            func.cuda_function.get_attribute(Attribute.CONST_SIZE_BYTES), 0
        )


def main():
    test_function_attributes()
