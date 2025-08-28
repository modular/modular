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

from collections import OptionalReg
from hashlib import default_comp_time_hasher
from buffer.dimlist import DimList
from buffer import NDBuffer
from linalg.bmm import bmm_sm100_blockwise_scaled_fp8
from sys import size_of
from gpu.host import DeviceContext
from layout._ndbuffer_stub import from_ndbuffer_row_major
from gpu.host._nvidia_cuda import TensorMapSwizzle
from utils.index import Index, IndexList
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from internal_utils._measure import relative_difference

# Additional imports for testing
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    assert_with_measure,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.bmm import elementwise_epilogue_type
from sys import align_of


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
    alias BLOCK_SCALE_K = 128
    alias block_tile_shape = Index(umma_shape[0], umma_shape[1], 128)

    constrained[transpose_b, "transpose_b must be true"]()

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

    alias static_a_shape = DimList(batch_size.dim, m.dim, k.dim)
    alias static_b_shape = DimList(
        batch_size.dim, n.dim, k.dim
    ) if transpose_b else DimList(batch_size.dim, k.dim, n.dim)
    alias static_c_shape = DimList(batch_size.dim, m.dim, n.dim)

    alias static_a_scales_shape = DimList(
        batch_size.dim, k.dim // BLOCK_SCALE_K, m.dim
    )
    alias static_b_scales_shape = DimList(
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

    alias static_a_shape_2D = DimList(m.dim, k.dim)
    alias static_b_shape_2D = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape_2D = DimList(m.dim, n.dim)

    alias static_a_scales_shape_2D = DimList(k.dim // BLOCK_SCALE_K, m.dim)
    alias static_b_scales_shape_2D = DimList(
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

    var a_host = HostNDBuffer[a_type, 3, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[b_type, 3, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[c_type, 3, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 3, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[a_type, 3, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 3, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 3, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 3, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    var a_scales_host = HostNDBuffer[DType.float32, 3, static_a_scales_shape](
        dynamic_a_scales_shape
    )
    var b_scales_host = HostNDBuffer[DType.float32, 3, static_b_scales_shape](
        dynamic_b_scales_shape
    )

    var a_scales_device = DeviceNDBuffer[
        DType.float32, 3, static_a_scales_shape
    ](dynamic_a_scales_shape, ctx=ctx)
    var b_scales_device = DeviceNDBuffer[
        DType.float32, 3, static_b_scales_shape
    ](dynamic_b_scales_shape, ctx=ctx)

    var c_tensor = c_device.tensor

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

    random(a_host.tensor)
    random(b_host.tensor)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    random(a_scales_host.tensor)
    random(b_scales_host.tensor)

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    ctx.enqueue_copy(a_scales_device.buffer, a_scales_host.tensor.data)
    ctx.enqueue_copy(b_scales_device.buffer, b_scales_host.tensor.data)

    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)
    var a_scales = from_ndbuffer_row_major(a_scales_device.tensor)
    var b_scales = from_ndbuffer_row_major(b_scales_device.tensor)

    bmm_sm100_blockwise_scaled_fp8[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        elementwise_lambda_fn = OptionalReg[elementwise_epilogue_type](
            epilogue_fn
        ) if use_epilogue else None,
    ](c, a, b, a_scales, b_scales, ctx)

    ctx.synchronize()

    for batch in range(bs):
        var c_ptr = c_device_ref.tensor.data + batch * M * N
        var a_ptr = a_device.tensor.data + batch * M * K
        var b_ptr = b_device.tensor.data + batch * N * K
        var a_scales_ptr = (
            a_scales_device.tensor.data + batch * (K // BLOCK_SCALE_K) * M
        )
        var b_scales_ptr = b_scales_device.tensor.data + batch * (
            N // BLOCK_SCALE_K
        ) * (K // BLOCK_SCALE_K)

        var c_buffer = NDBuffer[c_type, 2, _, static_c_shape_2D](
            c_ptr, dynamic_c_shape_2D
        )
        var a_buffer = NDBuffer[a_type, 2, _, static_a_shape_2D](
            a_ptr, dynamic_a_shape_2D
        )
        var b_buffer = NDBuffer[b_type, 2, _, static_b_shape_2D](
            b_ptr, dynamic_b_shape_2D
        )
        var a_scales_buffer = NDBuffer[
            DType.float32, 2, _, static_a_scales_shape_2D
        ](a_scales_ptr, dynamic_a_scales_shape_2D)
        var b_scales_buffer = NDBuffer[
            DType.float32, 2, _, static_b_scales_shape_2D
        ](b_scales_ptr, dynamic_b_scales_shape_2D)

        naive_blockwise_scaled_fp8_matmul[
            BLOCK_DIM=16,
            transpose_b=transpose_b,
        ](
            c_buffer,
            a_buffer,
            b_buffer,
            a_scales_buffer,
            b_scales_buffer,
            ctx,
        )

    ctx.synchronize()

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()

    assert_with_measure[relative_difference](
        c_host.tensor, c_host_ref.tensor, threshold=0.001
    )

    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=1e-2,
        rtol=1e-2,
    )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device
    _ = a_scales_host
    _ = b_scales_host
    _ = a_scales_device
    _ = b_scales_device

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
            DType.bfloat16,
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
            DType.bfloat16,
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
            umma_shape = Index(64, 256, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            dynamic(120),
            static[1280](),
            static[512](),
            dynamic(6),
        )
        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape = Index(64, 128, 32),
            swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
            use_epilogue=True,
        ](
            ctx,
            dynamic(128),
            static[128](),
            static[128](),
            dynamic(4),
        )
