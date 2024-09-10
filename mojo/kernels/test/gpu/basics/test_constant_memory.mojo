# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host import DeviceContext, FuncAttribute, ConstantMemoryMapping
from gpu.id import ThreadIdx
from gpu.memory import external_memory, AddressSpace
from gpu.sync import barrier
from sys import sizeof
from memory import stack_allocation
from gpu.host._compile import _compile_code
from memory import stack_allocation
from memory.reference import _GPUAddressSpace
from testing import assert_true, assert_equal


def test_constant_memory_compile(ctx: DeviceContext):
    fn alloc[n: Int]() -> UnsafePointer[Float32, _GPUAddressSpace.PARAM]:
        return stack_allocation[
            n, Float32, address_space = _GPUAddressSpace.PARAM
        ]()

    assert_true(".const .align 4 .b8 " in _compile_code[alloc[20]]().asm)
    assert_true(
        "internal addrspace(4) global [20 x float]"
        in _compile_code[alloc[20], emission_kind="llvm"]().asm
    )


def test_constant_mem(ctx: DeviceContext):
    print("== test_constant_mem")

    fn _fill_impl[n: Int]() -> UnsafePointer[Float32, AddressSpace.PARAM]:
        var ptr = stack_allocation[
            n, Float32, address_space = AddressSpace.PARAM
        ]()

        @parameter
        for i in range(n):
            ptr[i] = i
        return ptr

    fn static_constant_kernel[n: Int](data: UnsafePointer[Float32]):
        alias val = _fill_impl[n]()
        data[ThreadIdx.x()] = val[ThreadIdx.x()]

    var func = ctx.compile_function[static_constant_kernel[16]]()

    var res_host_ptr = UnsafePointer[Float32].alloc(16)
    var res_device = ctx.create_buffer[DType.float32](16)

    for i in range(16):
        res_host_ptr[i] = 0

    ctx.enqueue_copy_to_device(res_device, res_host_ptr)

    ctx.enqueue_function(func, res_device, grid_dim=1, block_dim=16)

    ctx.enqueue_copy_from_device(res_host_ptr, res_device)

    for i in range(16):
        assert_equal(res_host_ptr[i], i)

    _ = res_device
    res_host_ptr.free()


def test_constant_mem_via_func(ctx: DeviceContext):
    print("== test_constant_mem_via_func")

    fn _fill_impl[n: Int]() -> UnsafePointer[Float32, AddressSpace.PARAM]:
        var ptr = stack_allocation[
            n, Float32, address_space = AddressSpace.PARAM
        ]()

        @parameter
        for i in range(n):
            ptr[i] = i
        return ptr

    fn static_constant_kernel[
        get_constant_memory: fn () -> UnsafePointer[Float32, AddressSpace.PARAM]
    ](data: UnsafePointer[Float32]):
        alias val = get_constant_memory()
        data[ThreadIdx.x()] = val[ThreadIdx.x()]

    var func = ctx.compile_function[static_constant_kernel[_fill_impl[20]]]()

    var res_host_ptr = UnsafePointer[Float32].alloc(16)
    var res_device = ctx.create_buffer[DType.float32](16)

    for i in range(16):
        res_host_ptr[i] = 0

    ctx.enqueue_copy_to_device(res_device, res_host_ptr)

    ctx.enqueue_function(func, res_device, grid_dim=1, block_dim=16)

    ctx.enqueue_copy_from_device(res_host_ptr, res_device)

    for i in range(16):
        assert_equal(res_host_ptr[i], i)

    _ = res_device
    res_host_ptr.free()


def test_external_constant_mem(ctx: DeviceContext):
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

    for i in range(16):
        assert_equal(res_host_ptr[i], i)

    _ = res_device
    res_host_ptr.free()


def main():
    with DeviceContext() as ctx:
        test_constant_memory_compile(ctx)
        test_constant_mem(ctx)
        test_constant_mem_via_func(ctx)
        test_external_constant_mem(ctx)
