# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu import ThreadIdx
from gpu.host import DeviceContext
from gpu.host.func_attribute import Attribute
from memory import UnsafePointer, stack_allocation
from testing import *


# CHECK-LABEL: test_function_attributes
def test_function_attributes():
    fn kernel(x: UnsafePointer[Int]):
        x[0] = ThreadIdx.x

    # CHECK: tid.x

    with DeviceContext() as ctx:
        var func = ctx.compile_function[kernel, dump_asm=True]()
        assert_equal(
            func.test_only_get_attribute(Attribute.CONST_SIZE_BYTES), 0
        )


def main():
    test_function_attributes()
