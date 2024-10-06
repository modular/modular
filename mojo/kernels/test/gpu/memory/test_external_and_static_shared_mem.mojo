# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys.info import alignof, simdwidthof

from gpu.host import DeviceContext, FuncAttribute
from gpu.id import ThreadIdx
from gpu.memory import AddressSpace, external_memory
from gpu.sync import barrier
from memory import UnsafePointer, stack_allocation
from testing import assert_equal


def test_external_shared_mem(ctx: DeviceContext):
    fn dynamic_smem_kernel(data: UnsafePointer[Float32]):
        var sram = stack_allocation[
            16,
            Float32,
            address_space = AddressSpace.SHARED,
        ]()
        var dynamic_sram = external_memory[
            Float32,
            address_space = AddressSpace.SHARED,
            alignment = alignof[Scalar[DType.float32]](),
        ]()
        dynamic_sram[ThreadIdx.x()] = ThreadIdx.x()
        sram[ThreadIdx.x()] = ThreadIdx.x()
        barrier()
        data[ThreadIdx.x()] = dynamic_sram[ThreadIdx.x()] + sram[ThreadIdx.x()]

    # The default limitation is < 48KB for sm_80, 86, 89.
    var func = ctx.compile_function[dynamic_smem_kernel, dump_llvm=True](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(24960),
    )

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
        shared_mem_bytes=24960,
    )

    ctx.enqueue_copy_from_device(res_host_ptr, res_device)

    for i in range(16):
        assert_equal(res_host_ptr[i], 2 * i)

    _ = res_device
    res_host_ptr.free()


def main():
    with DeviceContext() as ctx:
        test_external_shared_mem(ctx)
