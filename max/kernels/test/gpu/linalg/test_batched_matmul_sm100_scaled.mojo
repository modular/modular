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

from std.collections import Optional
from std.sys import size_of

from std.gpu.host import DeviceContext
from std.gpu.host.nvidia.tma import TensorMapSwizzle

# Additional imports for testing
from internal_utils import (
    assert_almost_equal,
    assert_with_measure,
)
from std.random import rand
from internal_utils._measure import relative_difference
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout._utils import ManagedLayoutTensor
from linalg.bmm import (
    bmm_sm100_blockwise_scaled_fp8,
    batched_matmul_dynamic_scaled_fp8_naive,
    elementwise_epilogue_type,
)

from std.utils.index import Index, IndexList

from layout import TileTensor, Coord, CoordLike, Idx, row_major


def test_batched_matmul_sm100_blockwise_scaled_fp8[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    BatchType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    umma_shape: IndexList[3],
    swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    use_epilogue: Bool = False,
](
    ctx: DeviceContext,
    m: MType,
    n: NType,
    k: KType,
    batch_size: BatchType,
) raises:
    comptime BLOCK_SCALE_K = 128
    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], 128)

    comptime assert transpose_b, "transpose_b must be true"

    var M = m.value()
    var N = n.value()
    var K = k.value()
    var bs = batch_size.value()

    assert (
        M * size_of[DType.float32]() % 16 == 0
    ), "TMA expects M to be divisible by 16 bytes"

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

    assert K % BLOCK_SCALE_K == 0, "K must be divisible by BLOCK_SCALE_K"

    var a_shape = row_major(Coord(batch_size, m, k))
    var b_shape = row_major(
        Coord(
            batch_size,
            Idx[NType.static_value if transpose_b else KType.static_value](),
            Idx[KType.static_value if transpose_b else NType.static_value](),
        )
    )
    var c_shape = row_major(Coord(batch_size, m, n))
    var a_scales_shape = row_major(
        Coord(batch_size, Idx[KType.static_value // BLOCK_SCALE_K](), m)
    )
    var b_scales_shape = row_major(
        Coord(
            batch_size,
            Idx[NType.static_value // BLOCK_SCALE_K](),
            Idx[KType.static_value // BLOCK_SCALE_K](),
        )
    )

    var a_shape_2D = row_major(Coord(m, k))
    var b_shape_2D = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value](),
            Idx[KType.static_value if transpose_b else NType.static_value](),
        )
    )
    var c_shape_2D = row_major(Coord(m, n))
    var a_scales_shape_2D = row_major(
        Coord(Idx[KType.static_value // BLOCK_SCALE_K](), m)
    )
    var b_scales_shape_2d = row_major(
        Coord(
            Idx[NType.static_value // BLOCK_SCALE_K](),
            Idx[KType.static_value // BLOCK_SCALE_K](),
        )
    )

    var a_size = bs * M * K
    var b_size = bs * N * K if transpose_b else bs * K * N
    var c_size = bs * M * N
    var a_scales_size = bs * (K // BLOCK_SCALE_K) * M
    var b_scales_size = bs * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)

    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_managed = ManagedLayoutTensor[c_type, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(IndexList[1](c_size)),
        ctx,
    )
    var c_host = TileTensor(c_host_managed.tensor[update=False]().ptr, c_shape)
    var c_host_ref_managed = ManagedLayoutTensor[c_type, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(IndexList[1](c_size)),
        ctx,
    )
    var c_host_ref = TileTensor(
        c_host_ref_managed.tensor[update=False]().ptr, c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = TileTensor(c_device_ref, c_shape)

    var a_scales_host_ptr = alloc[Scalar[DType.float32]](a_scales_size)
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = alloc[Scalar[DType.float32]](b_scales_size)
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[DType.float32](
        a_scales_size
    )
    var a_scales_device_nd = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[DType.float32](
        b_scales_size
    )
    var b_scales_device_nd = TileTensor(b_scales_device, b_scales_shape)

    var c_tensor = c_device_nd

    @parameter
    @always_inline
    @__copy_capture(c_tensor, M, N)
    def epilogue_fn[
        dtype: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[dtype, width],) capturing -> None:
        comptime assert c_tensor.flat_rank >= 3
        c_tensor.store[alignment=alignment](
            Coord(Idx(idx[0]), Idx(idx[1]), Idx(idx[2])),
            rebind[SIMD[c_type, width]](val),
        )

    rand(a_host.ptr, a_host.num_elements())
    rand(b_host.ptr, b_host.num_elements())
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    rand(a_scales_host.ptr, a_scales_host.num_elements())
    rand(b_scales_host.ptr, b_scales_host.num_elements())

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host.ptr)

    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a = a_device_nd.to_layout_tensor()
    var b = b_device_nd.to_layout_tensor()
    var c = c_device_nd.to_layout_tensor()
    var a_scales = a_scales_device_nd.to_layout_tensor()
    var b_scales = b_scales_device_nd.to_layout_tensor()

    bmm_sm100_blockwise_scaled_fp8[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
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

    var c_ref = c_device_ref_nd.to_layout_tensor()

    batched_matmul_dynamic_scaled_fp8_naive[
        scales_granularity_mnk=Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
        transpose_b=transpose_b,
    ](c_ref, a, b, a_scales, b_scales, ctx)

    ctx.synchronize()

    ctx.enqueue_copy(c_host.ptr, c_device)
    ctx.enqueue_copy(c_host_ref.ptr, c_device_ref)
    ctx.synchronize()

    assert_with_measure[relative_difference](
        c_host.ptr, c_host_ref.ptr, c_host.num_elements(), threshold=0.001
    )

    assert_almost_equal(
        c_host.ptr,
        c_host_ref.ptr,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()


def test_batched_matmul_sm100_blockwise_scaled_fp8_non_row_major_c[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    umma_shape: IndexList[3],
    B: Int,
    N: Int,
    K: Int,
    swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
](ctx: DeviceContext, m: Int,) raises:
    comptime BLOCK_SCALE_K = 128
    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], 128)

    comptime assert transpose_b, "transpose_b must be true"

    var M = m
    var bs = B

    # For small M (e.g. M=1), pad only the scales-M dimension to satisfy the
    # 16-byte TMA alignment requirement for FP32 scales.
    var M_aligned_for_scales = ((M + 3) // 4) * 4

    print(
        "== test_sm100_blockwise_scaled_fp8_matmul_non_row_major_c",
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

    assert K % BLOCK_SCALE_K == 0, "K must be divisible by BLOCK_SCALE_K"

    var a_shape = row_major(Coord(Idx[B](), Idx(Int(M)), Idx[K]()))
    var b_shape = row_major(
        Coord(
            Idx[B](),
            Idx[N if transpose_b else K](),
            Idx[K if transpose_b else N](),
        )
    )
    var c_shape = row_major(Coord(Idx[B](), Idx(Int(M)), Idx[N]()))
    var a_scales_shape = row_major(
        Coord(
            Idx[B](),
            Idx[K // BLOCK_SCALE_K](),
            Idx(Int(M_aligned_for_scales)),
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[B](),
            Idx[N // BLOCK_SCALE_K](),
            Idx[K // BLOCK_SCALE_K](),
        )
    )

    var a_size = bs * M * K
    var b_size = bs * N * K if transpose_b else bs * K * N
    var c_size = bs * M * N
    var a_scales_size = bs * (K // BLOCK_SCALE_K) * M_aligned_for_scales
    var b_scales_size = bs * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)

    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_managed = ManagedLayoutTensor[c_type, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(IndexList[1](c_size)),
        ctx,
    )
    var c_host = TileTensor(c_host_managed.tensor[update=False]().ptr, c_shape)
    var c_host_ref_managed = ManagedLayoutTensor[c_type, Layout(UNKNOWN_VALUE)](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(IndexList[1](c_size)),
        ctx,
    )
    var c_host_ref = TileTensor(
        c_host_ref_managed.tensor[update=False]().ptr, c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = TileTensor(c_device_ref, c_shape)

    var a_scales_host_ptr = alloc[Scalar[DType.float32]](a_scales_size)
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = alloc[Scalar[DType.float32]](b_scales_size)
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[DType.float32](
        a_scales_size
    )
    var a_scales_device_nd = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[DType.float32](
        b_scales_size
    )
    var b_scales_device_nd = TileTensor(b_scales_device, b_scales_shape)

    rand(a_host.ptr, a_host.num_elements())
    rand(b_host.ptr, b_host.num_elements())
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    rand(a_scales_host.ptr, a_scales_host.num_elements())
    rand(b_scales_host.ptr, b_scales_host.num_elements())

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host.ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a = a_device_nd.to_layout_tensor()
    var b = b_device_nd.to_layout_tensor()
    var a_scales = a_scales_device_nd.to_layout_tensor()
    var b_scales = b_scales_device_nd.to_layout_tensor()

    comptime c_non_row_major_layout = Layout(
        IntTuple(B, UNKNOWN_VALUE, N),
        IntTuple(N, B * N, 1),
    )
    var c_runtime_layout = RuntimeLayout[c_non_row_major_layout](
        RuntimeTuple[c_non_row_major_layout.shape](bs, M, N),
        RuntimeTuple[c_non_row_major_layout.stride](N, B * N, 1),
    )

    var c = LayoutTensor[c_type, c_non_row_major_layout](
        c_device_nd.ptr, c_runtime_layout
    )
    var c_ref = LayoutTensor[c_type, c_non_row_major_layout](
        c_device_ref_nd.ptr, c_runtime_layout
    )

    bmm_sm100_blockwise_scaled_fp8[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=swizzle,
        b_swizzle=swizzle,
    ](
        c,
        a.get_immutable(),
        b.get_immutable(),
        a_scales.get_immutable(),
        b_scales.get_immutable(),
        ctx,
    )

    ctx.synchronize()

    batched_matmul_dynamic_scaled_fp8_naive[
        scales_granularity_mnk=Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
        transpose_b=transpose_b,
    ](c_ref, a, b, a_scales, b_scales, ctx)

    ctx.synchronize()

    ctx.enqueue_copy(c_host.ptr, c_device)
    ctx.enqueue_copy(c_host_ref.ptr, c_device_ref)
    ctx.synchronize()

    assert_with_measure[relative_difference](
        c_host.ptr, c_host_ref.ptr, c_host.num_elements(), threshold=0.001
    )

    assert_almost_equal(
        c_host.ptr,
        c_host_ref.ptr,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )

    a_host_ptr.free()
    b_host_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()


def main() raises:
    with DeviceContext() as ctx:
        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape=Index(64, 256, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Idx(Int(208)),
            Idx[2048](),
            Idx[256](),
            Idx(Int(3)),
        )
        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape=Index(64, 32, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Idx(Int(400)),
            Idx[128](),
            Idx[128](),
            Idx(Int(4)),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.float32,
            umma_shape=Index(64, 128, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Idx(Int(1024)),
            Idx[2048](),
            Idx[2048](),
            Idx(Int(2)),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.float32,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Idx(Int(1024)),
            Idx[2048](),
            Idx[2048](),
            Idx(Int(5)),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape=Index(64, 16, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Idx(Int(100)),
            Idx[512](),
            Idx[256](),
            Idx(Int(7)),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape=Index(64, 8, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Idx(Int(96)),
            Idx[1024](),
            Idx[1024](),
            Idx(Int(2)),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Idx(Int(120)),
            Idx[1280](),
            Idx[512](),
            Idx(Int(5)),
        )

        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Idx(Int(120)),
            Idx[512](),
            Idx[128](),
            Idx(Int(128)),
        )
        test_batched_matmul_sm100_blockwise_scaled_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
            use_epilogue=True,
        ](
            ctx,
            Idx(Int(120)),
            Idx[128](),
            Idx[512](),
            Idx(Int(128)),
        )

        # test non-row-major layout for C only
        test_batched_matmul_sm100_blockwise_scaled_fp8_non_row_major_c[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
            B=Int(128),
            N=Int(128),
            K=Int(512),
        ](ctx, 12)

        test_batched_matmul_sm100_blockwise_scaled_fp8_non_row_major_c[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
            B=Int(128),
            N=Int(512),
            K=Int(128),
        ](ctx, 12)
