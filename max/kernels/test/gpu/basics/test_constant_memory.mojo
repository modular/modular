# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


from gpu.host import ConstantMemoryMapping, DeviceContext
from gpu.host.compile import _compile_code
from gpu.id import thread_idx
from gpu.memory import AddressSpace
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace
from testing import assert_equal, assert_true


def test_constant_memory_compile(ctx: DeviceContext):
    fn alloc[
        n: Int
    ]() -> UnsafePointer[Float32, address_space = _GPUAddressSpace.CONSTANT]:
        return stack_allocation[
            n, Float32, address_space = _GPUAddressSpace.CONSTANT
        ]()

    assert_true(".const .align 4 .b8 " in _compile_code[alloc[20]]())
    assert_true(
        "internal addrspace(4) global [20 x float]"
        in _compile_code[alloc[20], emission_kind="llvm"]()
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

    var res_device = ctx.enqueue_create_buffer[DType.float32](16)
    _ = res_device.enqueue_fill(0)

    alias kernel = static_constant_kernel[16]
    ctx.enqueue_function_checked[kernel, kernel](
        res_device, grid_dim=1, block_dim=16
    )

    with res_device.map_to_host() as res_host:
        for i in range(16):
            assert_equal(res_host[i], i)


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

    var res_device = ctx.enqueue_create_buffer[DType.float32](16)
    _ = res_device.enqueue_fill(0)

    alias kernel = static_constant_kernel[_fill_impl[20]]
    ctx.enqueue_function_checked[kernel, kernel](
        res_device, grid_dim=1, block_dim=16
    )

    with res_device.map_to_host() as res_host:
        for i in range(16):
            assert_equal(res_host[i], i)


def test_external_constant_mem(ctx: DeviceContext):
    print("== test_external_constant_mem")

    fn static_constant_kernel(data: UnsafePointer[Float32]):
        var static_constant = stack_allocation[
            16,
            Float32,
            name = StaticString("static_constant"),
            address_space = AddressSpace.CONSTANT,
            alignment=8,
        ]()
        data[thread_idx.x] = static_constant[thread_idx.x]

    var constant_memory = List[Float32](
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    )

    var res_device = ctx.enqueue_create_buffer[DType.float32](16)
    _ = res_device.enqueue_fill(0)

    alias kernel = static_constant_kernel
    ctx.enqueue_function_checked[kernel, kernel](
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

    _ = constant_memory^

    with res_device.map_to_host() as res_host:
        for i in range(16):
            assert_equal(res_host[i], i)


def main():
    with DeviceContext() as ctx:
        test_constant_memory_compile(ctx)
        test_constant_mem(ctx)
        test_constant_mem_via_func(ctx)
        test_external_constant_mem(ctx)
