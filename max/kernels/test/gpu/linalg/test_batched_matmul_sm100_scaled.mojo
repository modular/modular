# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from collections import Optional
from hashlib import default_comp_time_hasher
from sys import align_of, size_of

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

# Additional imports for testing
from internal_utils import (
    assert_almost_equal,
    assert_with_measure,
)
from random import rand
from internal_utils._measure import relative_difference
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.bmm import (
    bmm_sm100_blockwise_scaled_fp8,
    batched_matmul_dynamic_scaled_fp8_naive,
    elementwise_epilogue_type,
)
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul

from utils.index import Index, IndexList


def test_batched_matmul_sm100_blockwise_scaled_fp8[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    umma_shape: IndexList[3],
    swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    use_epilogue: Bool = False,
](
    ctx: DeviceContext,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    batch_size: ValOrDim,
):
    comptime BLOCK_SCALE_K = 128
    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], 128)

    comptime assert transpose_b, "transpose_b must be true"

    var M = m.value
    var N = n.value
    var K = k.value
    var bs = batch_size.value

    debug_assert(
        M * size_of[DType.float32]() % 16 == 0,
        "TMA expects M to be divisible by 16 bytes",
    )

    print(
        "== test_sm100_blockwise_scaled_fp8_matmul",
        a_type,
        "problem shape: (",
        bs,
        "x",
        M,
        "x",
        N,
        "x",
        K,
        ")",
        "block_tile_shape: (",
        block_tile_shape[0],
        "x",
        block_tile_shape[1],
        "x",
        block_tile_shape[2],
        ")",
        "transpose_b:",
        transpose_b,
    )

    debug_assert(
        (K % BLOCK_SCALE_K == 0),
        "K must be divisible by BLOCK_SCALE_K",
    )

    comptime static_a_shape = DimList(batch_size.dim, m.dim, k.dim)
    comptime static_b_shape = DimList(
        batch_size.dim, n.dim, k.dim
    ) if transpose_b else DimList(batch_size.dim, k.dim, n.dim)
    comptime static_c_shape = DimList(batch_size.dim, m.dim, n.dim)

    comptime static_a_scales_shape = DimList(
        batch_size.dim, k.dim // BLOCK_SCALE_K, m.dim
    )
    comptime static_b_scales_shape = DimList(
        batch_size.dim, n.dim // BLOCK_SCALE_K, k.dim // BLOCK_SCALE_K
    )

    var dynamic_a_shape = DimList(bs, M, K)
    var dynamic_b_shape = DimList(bs, N, K) if transpose_b else DimList(
        bs, K, N
    )
    var dynamic_c_shape = DimList(bs, M, N)
    var dynamic_a_scales_shape = DimList(bs, K // BLOCK_SCALE_K, M)
    var dynamic_b_scales_shape = DimList(
        bs, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K
    )

    comptime static_a_shape_2D = DimList(m.dim, k.dim)
    comptime static_b_shape_2D = DimList(
        n.dim, k.dim
    ) if transpose_b else DimList(k.dim, n.dim)
    comptime static_c_shape_2D = DimList(m.dim, n.dim)

    comptime static_a_scales_shape_2D = DimList(k.dim // BLOCK_SCALE_K, m.dim)
    comptime static_b_scales_shape_2D = DimList(
        n.dim // BLOCK_SCALE_K, k.dim // BLOCK_SCALE_K
    )

    var dynamic_a_shape_2D = DimList(m.value, k.value)
    var dynamic_b_shape_2D = DimList(
        n.value, k.value
    ) if transpose_b else DimList(k.value, n.value)
    var dynamic_c_shape_2D = DimList(m.value, n.value)
    var dynamic_a_scales_shape_2D = DimList(k.value // BLOCK_SCALE_K, m.value)
    var dynamic_b_scales_shape_2D = DimList(
        n.value // BLOCK_SCALE_K, k.value // BLOCK_SCALE_K
    )

    var a_size = bs * M * K
    var b_size = bs * N * K if transpose_b else bs * K * N
    var c_size = bs * M * N
    var a_scales_size = bs * (K // BLOCK_SCALE_K) * M
    var b_scales_size = bs * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)

    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var a_host = NDBuffer[a_type, 3, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var b_host = NDBuffer[b_type, 3, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host = NDBuffer[c_type, 3, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref = NDBuffer[c_type, 3, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = NDBuffer[a_type, 3, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = NDBuffer[b_type, 3, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = NDBuffer[c_type, 3, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = NDBuffer[c_type, 3, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    var a_scales_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(
        a_scales_size
    )
    var a_scales_host = NDBuffer[DType.float32, 3, _, static_a_scales_shape](
        a_scales_host_ptr, dynamic_a_scales_shape
    )
    var b_scales_host_ptr = UnsafePointer[Scalar[DType.float32]].alloc(
        b_scales_size
    )
    var b_scales_host = NDBuffer[DType.float32, 3, _, static_b_scales_shape](
        b_scales_host_ptr, dynamic_b_scales_shape
    )

    var a_scales_device = ctx.enqueue_create_buffer[DType.float32](
        a_scales_size
    )
    var a_scales_device_nd = NDBuffer[
        DType.float32, 3, _, static_a_scales_shape
    ](a_scales_device.unsafe_ptr(), dynamic_a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[DType.float32](
        b_scales_size
    )
    var b_scales_device_nd = NDBuffer[
        DType.float32, 3, _, static_b_scales_shape
    ](b_scales_device.unsafe_ptr(), dynamic_b_scales_shape)

    var c_tensor = c_device_nd

    @parameter
    @always_inline
    @__copy_capture(c_tensor, M, N)
    fn epilogue_fn[
        dtype: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[dtype, width],) capturing -> None:
        c_tensor.store[alignment=alignment](
            Index(idx[0], idx[1], idx[2]), rebind[SIMD[c_type, width]](val)
        )

    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())
    c_host.zero()
    c_host_ref.zero()

    rand(a_scales_host.data, a_scales_host.num_elements())
    rand(b_scales_host.data, b_scales_host.num_elements())

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host_ptr)

    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a = from_ndbuffer_row_major(a_device_nd)
    var b = from_ndbuffer_row_major(b_device_nd)
    var c = from_ndbuffer_row_major(c_device_nd)
    var a_scales = from_ndbuffer_row_major(a_scales_device_nd)
    var b_scales = from_ndbuffer_row_major(b_scales_device_nd)

    bmm_sm100_blockwise_scaled_fp8[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        elementwise_lambda_fn = Optional[elementwise_epilogue_type](
            epilogue_fn
        ) if use_epilogue else None,
    ](
        c,
        a.get_immutable(),
        b.get_immutable(),
        a_scales.get_immutable(),
        b_scales.get_immutable(),
        ctx,
    )

    ctx.synchronize()

    var c_ref = from_ndbuffer_row_major(c_device_ref_nd)

    batched_matmul_dynamic_scaled_fp8_naive[
        scales_granularity_mnk = Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
        transpose_b=transpose_b,
    ](c_ref, a, b, a_scales, b_scales, ctx)

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_with_measure[relative_difference](
        c_host.data, c_host_ref.data, c_host.num_elements(), threshold=0.001
    )

    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^

    _ = a
    _ = b
    _ = c


def main():
    with DeviceContext() as ctx:
        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape = Index(64, 256, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            dynamic(208),
            static[2048](),
            static[256](),
            dynamic(3),
        )
        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape = Index(64, 32, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            dynamic(400),
            static[128](),
            static[128](),
            dynamic(4),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.float32,
            umma_shape = Index(64, 128, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            dynamic(1024),
            static[2048](),
            static[2048](),
            dynamic(2),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.float32,
            umma_shape = Index(64, 64, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            dynamic(1024),
            static[2048](),
            static[2048](),
            dynamic(5),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape = Index(64, 16, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            dynamic(100),
            static[512](),
            static[256](),
            dynamic(7),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape = Index(64, 8, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            dynamic(96),
            static[1024](),
            static[1024](),
            dynamic(2),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape = Index(64, 64, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            dynamic(120),
            static[1280](),
            static[512](),
            dynamic(5),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape = Index(64, 64, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            dynamic(120),
            static[512](),
            static[128](),
            dynamic(128),
        )
        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape = Index(64, 64, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
            use_epilogue=True,
        ](
            ctx,
            dynamic(120),
            static[128](),
            static[512](),
            dynamic(128),
        )
