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

from math import align_down, ceildiv

from gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from gpu.host import DeviceContext
from gpu.host.compile import _compile_code
from gpu.host import get_gpu_target
from gpu.intrinsics import (
    buffer_load,
    buffer_load_store_lds,
    buffer_store,
    make_buffer_resource,
)
from gpu.memory import AddressSpace, CacheOperation
from memory import stack_allocation
from testing import assert_equal, assert_true

alias size = 257
alias size_clip = size - 5


fn kernel[dtype: DType, width: Int](a: UnsafePointer[Scalar[dtype]]):
    var aligned_size = align_down(size, width)
    var bc = make_buffer_resource(a, size_clip)
    for i in range(0, aligned_size, width):
        var v = buffer_load[dtype, width](bc, i)
        buffer_store[dtype, width](bc, 0, 2 * v, scalar_offset=i)
    for i in range(aligned_size, size):
        var v = buffer_load[dtype, 1](bc, 0, scalar_offset=i)
        buffer_store[dtype, 1](bc, i, 2 * v)


fn kernel_lds[dtype: DType](a: UnsafePointer[Scalar[dtype]]):
    var a_shared = stack_allocation[
        size, dtype, address_space = AddressSpace.SHARED
    ]()

    var idx = thread_idx.x

    var t0 = block_idx.x * block_dim.x + thread_idx.x
    var bc = make_buffer_resource(a, size_clip)
    for i in range(t0, size, block_dim.x * grid_dim.x):
        a_shared[i] = 0
    barrier()

    for i in range(t0, size, block_dim.x * grid_dim.x):
        buffer_load_store_lds(bc, i, a_shared, i)
    for i in range(t0, size, block_dim.x * grid_dim.x):
        a[i] = 2 * a_shared[i]


# Assembly test kernels for different cache policies
fn cache_policy_kernel_always():
    var dummy_ptr = UnsafePointer[Scalar[DType.float32]]()
    var bc = make_buffer_resource(dummy_ptr, 1024)
    var offset = Int32(thread_idx.x)  # Use dynamic offset to force offen mode
    var v = buffer_load[DType.float32, 4, cache_policy = CacheOperation.ALWAYS](
        bc, offset
    )
    buffer_store[DType.float32, 4, cache_policy = CacheOperation.ALWAYS](
        bc, offset, v
    )


fn cache_policy_kernel_streaming():
    var dummy_ptr = UnsafePointer[Scalar[DType.float32]]()
    var bc = make_buffer_resource(dummy_ptr, 1024)
    var offset = Int32(thread_idx.x)  # Use dynamic offset to force offen mode
    var v = buffer_load[
        DType.float32, 4, cache_policy = CacheOperation.STREAMING
    ](bc, offset)
    buffer_store[DType.float32, 4, cache_policy = CacheOperation.STREAMING](
        bc, offset, v
    )


fn cache_policy_kernel_global():
    var dummy_ptr = UnsafePointer[Scalar[DType.float32]]()
    var bc = make_buffer_resource(dummy_ptr, 1024)
    var offset = Int32(thread_idx.x)  # Use dynamic offset to force offen mode
    var v = buffer_load[DType.float32, 4, cache_policy = CacheOperation.GLOBAL](
        bc, offset
    )
    buffer_store[DType.float32, 4, cache_policy = CacheOperation.GLOBAL](
        bc, offset, v
    )


fn cache_policy_kernel_volatile():
    var dummy_ptr = UnsafePointer[Scalar[DType.float32]]()
    var bc = make_buffer_resource(dummy_ptr, 1024)
    var offset = Int32(thread_idx.x)  # Use dynamic offset to force offen mode
    var v = buffer_load[
        DType.float32, 4, cache_policy = CacheOperation.VOLATILE
    ](bc, offset)
    buffer_store[DType.float32, 4, cache_policy = CacheOperation.VOLATILE](
        bc, offset, v
    )


@always_inline
fn _verify_cache_bits_always(asm: StringSlice) raises -> None:
    # aux = 0x00 - no cache control bits set
    # Should generate: buffer_load_dwordx4 v[...], v..., s[...], 0 offen
    assert_true("buffer_load_dwordx4" in asm)
    assert_true("offen" in asm)
    # Should NOT have sc0, nt, or sc1
    assert_true("sc0" not in asm)
    assert_true(" nt" not in asm)  # Space prefix to avoid matching "int"
    assert_true("sc1" not in asm)


@always_inline
fn _verify_cache_bits_streaming(asm: StringSlice) raises -> None:
    # aux = 0x02 - only NT bit set
    # Should generate: buffer_load_dwordx4 v[...], v..., s[...], 0 offen nt
    assert_true("buffer_load_dwordx4" in asm)
    assert_true("offen" in asm)
    assert_true(" nt" in asm)  # Space prefix to avoid matching "int"
    # Should NOT have sc0 or sc1
    assert_true("sc0" not in asm)
    assert_true("sc1" not in asm)


@always_inline
fn _verify_cache_bits_global(asm: StringSlice) raises -> None:
    # aux = 0x10 - only SC1 bit set
    # Should generate: buffer_load_dwordx4 v[...], v..., s[...], 0 offen sc1
    assert_true("buffer_load_dwordx4" in asm)
    assert_true("offen" in asm)
    assert_true("sc1" in asm)
    # Should NOT have sc0 or nt
    assert_true("sc0" not in asm)
    assert_true(" nt" not in asm)  # Space prefix to avoid matching "int"


@always_inline
fn _verify_cache_bits_volatile(asm: StringSlice) raises -> None:
    # aux = 0x11 - SC0 and SC1 bits set
    # Should generate: buffer_load_dwordx4 v[...], v..., s[...], 0 offen sc0 sc1
    assert_true("buffer_load_dwordx4" in asm)
    assert_true("offen" in asm)
    assert_true("sc0" in asm)
    assert_true("sc1" in asm)
    # Should NOT have nt
    assert_true(" nt" not in asm)  # Space prefix to avoid matching "int"


def test_cache_policy_assembly_always():
    var asm = _compile_code[
        cache_policy_kernel_always, target = get_gpu_target["mi300x"]()
    ]().asm
    _verify_cache_bits_always(asm)


def test_cache_policy_assembly_streaming():
    var asm = _compile_code[
        cache_policy_kernel_streaming, target = get_gpu_target["mi300x"]()
    ]().asm
    _verify_cache_bits_streaming(asm)


def test_cache_policy_assembly_global():
    var asm = _compile_code[
        cache_policy_kernel_global, target = get_gpu_target["mi300x"]()
    ]().asm
    _verify_cache_bits_global(asm)


def test_cache_policy_assembly_volatile():
    var asm = _compile_code[
        cache_policy_kernel_volatile, target = get_gpu_target["mi300x"]()
    ]().asm
    _verify_cache_bits_volatile(asm)


def test_buffer[dtype: DType, width: Int](ctx: DeviceContext):
    a_host_buf = UnsafePointer[Scalar[dtype]].alloc(size)
    a_device_buf = ctx.enqueue_create_buffer[dtype](size)

    for i in range(size):
        a_host_buf[i] = i + 1

    ctx.enqueue_copy(a_device_buf, a_host_buf)

    ctx.enqueue_function[kernel[dtype, width]](
        a_device_buf, grid_dim=1, block_dim=1
    )
    ctx.enqueue_copy(a_host_buf, a_device_buf)

    ctx.synchronize()
    for i in range(size_clip):
        assert_equal(a_host_buf[i], 2 * (i + 1))
    for i in range(size_clip, size):
        assert_equal(a_host_buf[i], i + 1)

    a_host_buf.free()


def test_buffer_lds(ctx: DeviceContext):
    alias dtype = DType.float32
    a_host_buf = UnsafePointer[Scalar[dtype]].alloc(size)
    a_device_buf = ctx.enqueue_create_buffer[dtype](size)

    for i in range(size):
        a_host_buf[i] = i + 1

    ctx.enqueue_copy(a_device_buf, a_host_buf)

    ctx.enqueue_function[kernel_lds[dtype]](
        a_device_buf,
        grid_dim=ceildiv(size, 256),
        block_dim=256,
    )
    ctx.enqueue_copy(a_host_buf, a_device_buf)

    ctx.synchronize()
    for i in range(size_clip):
        assert_equal(a_host_buf[i], 2 * (i + 1))
    for i in range(size_clip, size):
        assert_equal(a_host_buf[i], 0)


def main():
    # Test assembly generation for cache policies (AMD GPU only)
    test_cache_policy_assembly_always()
    test_cache_policy_assembly_streaming()
    test_cache_policy_assembly_global()
    test_cache_policy_assembly_volatile()
    # test_cache_policy_assembly_volatile_streaming()  # Future test

    # Test functional behavior
    with DeviceContext() as ctx:

        @parameter
        for width in [1, 2, 4, 8]:
            test_buffer[DType.bfloat16, width](ctx)

        @parameter
        for width in [1, 2, 4, 8, 16]:
            test_buffer[DType.int8, width](ctx)
        test_buffer_lds(ctx)
