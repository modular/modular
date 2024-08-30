# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from gpu.host import DeviceContext, FuncAttribute, ConstantMemoryMapping
from gpu.id import ThreadIdx
from gpu.memory import external_memory, AddressSpace
from gpu.sync import barrier
from sys import sizeof
from memory import stack_allocation


# CHECK-LABEL: test_external_constant_mem
fn test_external_constant_mem(ctx: DeviceContext) raises:
    print("== test_external_constant_mem")

    fn static_constant_kernel(data: UnsafePointer[Float32]):
        var static_constant = stack_allocation[
            16,
            Float32,
            name="static_constant",
            address_space = AddressSpace.PARAM,
            alignment=8,
        ]()
        data[ThreadIdx.x()] = static_constant[ThreadIdx.x()]

    var constant_memory = List[Float32](
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    )

    var func = ctx.compile_function[static_constant_kernel]()

    var res_host_ptr = UnsafePointer[Float32].alloc(16)
    var res_device = ctx.create_buffer[DType.float32](16)

    for i in range(16):
        res_host_ptr[i] = 0

    ctx.enqueue_copy_to_device(res_device, res_host_ptr)

    ctx.enqueue_function(
        func,
        res_device,
        grid_dim=1,
        block_dim=16,
        constant_memory=List[ConstantMemoryMapping](
            ConstantMemoryMapping(
                "static_constant",
                constant_memory.unsafe_ptr().bitcast[NoneType](),
                constant_memory.bytecount(),
            )
        ),
    )

    ctx.enqueue_copy_from_device(res_host_ptr, res_device)

    _ = constant_memory^

    # CHECK: 0.0
    # CHECK: 1.0
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: 6.0
    # CHECK: 7.0
    # CHECK: 8.0
    # CHECK: 9.0
    # CHECK: 10.0
    # CHECK: 11.0
    # CHECK: 12.0
    # CHECK: 13.0
    # CHECK: 14.0
    # CHECK: 15.0
    for i in range(16):
        print(res_host_ptr[i])

    _ = res_device
    res_host_ptr.free()


def main():
    with DeviceContext() as ctx:
        test_external_constant_mem(ctx)
