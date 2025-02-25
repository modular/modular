# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s

from sys import sizeof

from gpu.host import ConstantMemoryMapping, DeviceContext, FuncAttribute
from gpu.host._compile import _compile_code_asm
from gpu.id import thread_idx
from gpu.memory import AddressSpace, external_memory
from gpu.sync import barrier
from memory import UnsafePointer, stack_allocation
from memory.pointer import _GPUAddressSpace
from testing import assert_equal, assert_true


def test_constant_memory_compile(ctx: DeviceContext):
    fn alloc[
        n: Int
    ]() -> UnsafePointer[Float32, address_space = _GPUAddressSpace.CONSTANT]:
        return stack_allocation[
            n, Float32, address_space = _GPUAddressSpace.CONSTANT
        ]()

    assert_true(".const .align 4 .b8 " in _compile_code_asm[alloc[20]]())
    assert_true(
        "internal addrspace(4) global [20 x float]"
        in _compile_code_asm[alloc[20], emission_kind="llvm"]()
    )


def test_constant_mem(ctx: DeviceContext):
    print("== test_constant_mem")

    fn _fill_impl[
        n: Int
    ]() -> UnsafePointer[Float32, address_space = AddressSpace.CONSTANT]:
        var ptr = stack_allocation[
            n, Float32, address_space = AddressSpace.CONSTANT
        ]()

        @parameter
        for i in range(n):
            ptr[i] = i
        return ptr

    fn static_constant_kernel[n: Int](data: UnsafePointer[Float32]):
        alias val = _fill_impl[n]()
        data[thread_idx.x] = val[thread_idx.x]

    var res_host_ptr = UnsafePointer[Float32].alloc(16)
    var res_device = ctx.enqueue_create_buffer[DType.float32](16)

    for i in range(16):
        res_host_ptr[i] = 0

    ctx.enqueue_copy(res_device, res_host_ptr)

    ctx.enqueue_function[static_constant_kernel[16]](
        res_device, grid_dim=1, block_dim=16
    )

    ctx.enqueue_copy(res_host_ptr, res_device)

    ctx.synchronize()

    for i in range(16):
        assert_equal(res_host_ptr[i], i)

    _ = res_device^
    res_host_ptr.free()


def test_constant_mem_via_func(ctx: DeviceContext):
    print("== test_constant_mem_via_func")

    fn _fill_impl[
        n: Int
    ]() -> UnsafePointer[Float32, address_space = AddressSpace.CONSTANT]:
        var ptr = stack_allocation[
            n, Float32, address_space = AddressSpace.CONSTANT
        ]()

        @parameter
        for i in range(n):
            ptr[i] = i
        return ptr

    fn static_constant_kernel[
        get_constant_memory: fn () -> UnsafePointer[
            Float32, address_space = AddressSpace.CONSTANT
        ]
    ](data: UnsafePointer[Float32]):
        alias val = get_constant_memory()
        data[thread_idx.x] = val[thread_idx.x]

    var res_host_ptr = UnsafePointer[Float32].alloc(16)
    var res_device = ctx.enqueue_create_buffer[DType.float32](16)

    for i in range(16):
        res_host_ptr[i] = 0

    ctx.enqueue_copy(res_device, res_host_ptr)

    ctx.enqueue_function[static_constant_kernel[_fill_impl[20]]](
        res_device, grid_dim=1, block_dim=16
    )

    ctx.enqueue_copy(res_host_ptr, res_device)

    for i in range(16):
        assert_equal(res_host_ptr[i], i)

    _ = res_device^
    res_host_ptr.free()


def test_external_constant_mem(ctx: DeviceContext):
    print("== test_external_constant_mem")

    fn static_constant_kernel(data: UnsafePointer[Float32]):
        var static_constant = stack_allocation[
            16,
            Float32,
            name="static_constant",
            address_space = AddressSpace.CONSTANT,
            alignment=8,
        ]()
        data[thread_idx.x] = static_constant[thread_idx.x]

    var constant_memory = List[Float32](
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    )

    var res_host_ptr = UnsafePointer[Float32].alloc(16)
    var res_device = ctx.enqueue_create_buffer[DType.float32](16)

    for i in range(16):
        res_host_ptr[i] = 0

    ctx.enqueue_copy(res_device, res_host_ptr)

    ctx.enqueue_function[static_constant_kernel](
        res_device,
        grid_dim=1,
        block_dim=16,
        constant_memory=List[ConstantMemoryMapping](
            ConstantMemoryMapping(
                "static_constant",
                constant_memory.unsafe_ptr().bitcast[NoneType](),
                constant_memory.byte_length(),
            )
        ),
    )

    ctx.enqueue_copy(res_host_ptr, res_device)

    _ = constant_memory^

    for i in range(16):
        assert_equal(res_host_ptr[i], i)

    _ = res_device^
    res_host_ptr.free()


def main():
    with DeviceContext() as ctx:
        test_constant_memory_compile(ctx)
        test_constant_mem(ctx)
        test_constant_mem_via_func(ctx)
        test_external_constant_mem(ctx)
