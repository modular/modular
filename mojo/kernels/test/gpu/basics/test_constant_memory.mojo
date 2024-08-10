# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from gpu.host._compile import _compile_code
from testing import assert_true
from memory import stack_allocation
from memory.reference import _GPUAddressSpace


def test_constant_memory():
    fn alloc[n: Int]() -> UnsafePointer[Float32, _GPUAddressSpace.PARAM]:
        return stack_allocation[
            n, Float32, address_space = _GPUAddressSpace.PARAM
        ]()

    assert_true(".const .align 4 .b8 " in _compile_code[alloc[20]]().asm)
    assert_true(
        "internal addrspace(4) global [20 x float]"
        in _compile_code[alloc[20], emission_kind="llvm"]().asm
    )


def main():
    test_constant_memory()
