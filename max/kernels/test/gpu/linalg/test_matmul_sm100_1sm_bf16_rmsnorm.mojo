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
#
# Test: RMS norm → bf16 matmul (sm100, 1SM, normal epilogue)
#
# Validates that our Blackwell bf16 matmul produces the same result as vendor
# matmul (cuBLAS) when both receive RMS-normed inputs. Each comparison path
# runs its own separate rms_norm_gpu launch so that this test can later be
# adapted to exercise PDL (Programmatic Dependent Launch).
# ===----------------------------------------------------------------------=== #

from std.sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from std.gpu.host import DeviceContext
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from linalg.utils import elementwise_epilogue_type

from internal_utils import assert_almost_equal
from std.random import rand
from layout import TileTensor, Coord, CoordLike, row_major, Idx
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from std.gpu.primitives.grid_controls import PDLLevel
from nn.normalization import rms_norm_gpu

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


def test_rmsnorm_then_matmul[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    pdl_level: PDLLevel = PDLLevel(),
    prefetch_tiles_n: Int = 0,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var M = m.value()
    var N = n.value()
    var K = k.value()

    print(
        t"rmsnorm->matmul: dtype={a_type} shape=({M}, {N}, {K})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape}"
        t" cta_group={cta_group} swapAB={swapAB}"
        t" prefetch_tiles_n={prefetch_tiles_n}"
    )

    # --- Shapes ---
    var ak_shape = row_major(Coord(m, Idx[KType.static_value]()))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value](),
            Idx[KType.static_value if transpose_b else NType.static_value](),
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]()))

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N

    # --- Host allocations ---
    var a_raw_host_ptr = alloc[Scalar[a_type]](a_size)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var gamma_host_ptr = alloc[Scalar[a_type]](K)
    var c_vendor_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_ours_host_ptr = alloc[Scalar[c_type]](c_size)

    # Random A and B; gamma[i] = (i + K) / K
    rand(a_raw_host_ptr, a_size)
    rand(b_host_ptr, b_size)
    for i in range(K):
        gamma_host_ptr[i] = (Float64(i + K) / Float64(K)).cast[a_type]()

    # --- Device allocations ---
    var a_raw_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_raw_tensor = TileTensor(a_raw_device.unsafe_ptr(), ak_shape)

    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device.unsafe_ptr(), b_shape)

    var gamma_device = ctx.enqueue_create_buffer[a_type](K)
    var gamma_tensor = TileTensor(
        gamma_device.unsafe_ptr(), row_major(Idx[KType.static_value]())
    )

    # Separate normalized-A buffers — one per launch, intentionally independent
    var a_normed_vendor_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_normed_vendor_tensor = TileTensor(
        a_normed_vendor_device.unsafe_ptr(), ak_shape
    )

    var a_normed_ours_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_normed_ours_tensor = TileTensor(
        a_normed_ours_device.unsafe_ptr(), ak_shape
    )

    var c_vendor_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_vendor_tensor = TileTensor(c_vendor_device.unsafe_ptr(), c_shape)

    var c_ours_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ours_tensor = TileTensor(c_ours_device.unsafe_ptr(), c_shape)

    # H→D copies
    ctx.enqueue_copy(a_raw_device, a_raw_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(gamma_device, gamma_host_ptr)

    var epsilon = Scalar[a_type](0.001)
    var weight_offset = Scalar[a_type](0.0)
    var norm_shape = Index(M, K)

    # input_fn: shared by both paths, reads from a_raw (read-only)
    @always_inline
    @__copy_capture(a_raw_tensor)
    @parameter
    def input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[a_type, width]:
        return a_raw_tensor.ptr.load[width=width](
            a_raw_tensor.layout(Coord(coords))
        )

    # -----------------------------------------------------------------------
    # Vendor path: RMS norm launch 1 → a_normed_vendor → cuBLAS matmul
    # -----------------------------------------------------------------------
    @always_inline
    @__copy_capture(a_normed_vendor_tensor)
    @parameter
    def output_fn_vendor[
        width: Int, alignment: Int
    ](coords: IndexList[2], val: SIMD[a_type, width]) -> None:
        a_normed_vendor_tensor.ptr.store[width=width, alignment=alignment](
            a_normed_vendor_tensor.layout(Coord(coords)), val
        )

    rms_norm_gpu[input_fn, output_fn_vendor, multiply_before_cast=True](
        norm_shape, gamma_tensor, epsilon, weight_offset, ctx
    )

    vendor_blas.matmul(
        ctx,
        c_vendor_tensor.to_layout_tensor(),
        a_normed_vendor_tensor.to_layout_tensor(),
        b_tensor.to_layout_tensor(),
        c_row_major=True,
        transpose_b=transpose_b,
    )

    # -----------------------------------------------------------------------
    # Our kernel path: RMS norm launch 2 → a_normed_ours → Blackwell matmul
    # -----------------------------------------------------------------------
    @always_inline
    @__copy_capture(a_normed_ours_tensor)
    @parameter
    def output_fn_ours[
        width: Int, alignment: Int
    ](coords: IndexList[2], val: SIMD[a_type, width]) -> None:
        a_normed_ours_tensor.ptr.store[width=width, alignment=alignment](
            a_normed_ours_tensor.layout(Coord(coords)), val
        )

    rms_norm_gpu[input_fn, output_fn_ours, multiply_before_cast=True](
        norm_shape, gamma_tensor, epsilon, weight_offset, ctx
    )

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        prefetch_tiles_n=prefetch_tiles_n,
    )

    @parameter
    @always_inline
    @__copy_capture(c_ours_tensor)
    def epilogue_fn[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_ours_tensor.ptr.store[
            width=width, alignment=alignment * size_of[c_type]()
        ](c_ours_tensor.layout(Coord(idx)), rebind[SIMD[c_type, width]](val))

    comptime epi = Optional[elementwise_epilogue_type](epilogue_fn)

    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
        elementwise_lambda_fn=epi,
        pdl_level=pdl_level,
    ](
        c_ours_tensor,
        a_normed_ours_tensor,
        b_tensor,
        ctx,
    )

    # -----------------------------------------------------------------------
    # Compare
    # -----------------------------------------------------------------------
    ctx.synchronize()

    ctx.enqueue_copy(c_vendor_host_ptr, c_vendor_device)
    ctx.enqueue_copy(c_ours_host_ptr, c_ours_device)
    ctx.synchronize()

    assert_almost_equal(
        c_ours_host_ptr,
        c_vendor_host_ptr,
        c_size,
        atol=0.0001,
        rtol=1e-2,
    )
    print("\n=== TEST PASSED ===\n")

    # Cleanup
    a_raw_host_ptr.free()
    b_host_ptr.free()
    gamma_host_ptr.free()
    c_vendor_host_ptr.free()
    c_ours_host_ptr.free()
    _ = a_raw_device^
    _ = b_device^
    _ = gamma_device^
    _ = a_normed_vendor_device^
    _ = a_normed_ours_device^
    _ = c_vendor_device^
    _ = c_ours_device^


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        comptime BK = TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[dtype]()
        comptime MMA_K = 16

        # Small-M tests (decode / prefill with tiny batch)
        comptime for m in [1, 2, 4, 8, 16, 32, 48, 63]:
            test_rmsnorm_then_matmul[
                dtype,
                dtype,
                DType.bfloat16,
                block_tile_shape=Index(64, 128, BK),
                mma_shape=Index(64, 128, MMA_K),
                cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                cta_group=1,
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                block_swizzle_size=8,
                swapAB=True,
            ](
                ctx,
                Idx(Int(m)),
                Idx[4096](),
                Idx[4096](),
            )

            test_rmsnorm_then_matmul[
                dtype,
                dtype,
                DType.bfloat16,
                block_tile_shape=Index(64, 128, BK),
                mma_shape=Index(64, 128, MMA_K),
                cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                cta_group=1,
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                block_swizzle_size=8,
                swapAB=True,
            ](
                ctx,
                Idx(Int(m)),
                Idx[8192](),
                Idx[7168](),
            )

        # PDL prefetch tests: same small-M shapes with swapAB=True,
        # pdl_level=OVERLAP_AT_END, prefetch_tiles_n=1
        comptime for m in [1, 2, 4, 8, 16, 32, 48, 63]:
            test_rmsnorm_then_matmul[
                dtype,
                dtype,
                DType.bfloat16,
                block_tile_shape=Index(64, 128, BK),
                mma_shape=Index(64, 128, MMA_K),
                cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                cta_group=1,
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                block_swizzle_size=8,
                swapAB=True,
                pdl_level=PDLLevel.OVERLAP_AT_END,
                prefetch_tiles_n=2,
            ](
                ctx,
                Idx(Int(m)),
                Idx[4096](),
                Idx[4096](),
            )

            test_rmsnorm_then_matmul[
                dtype,
                dtype,
                DType.bfloat16,
                block_tile_shape=Index(64, 128, BK),
                mma_shape=Index(64, 128, MMA_K),
                cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                cta_group=1,
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                block_swizzle_size=8,
                swapAB=True,
                pdl_level=PDLLevel.OVERLAP_AT_END,
                prefetch_tiles_n=2,
            ](
                ctx,
                Idx(Int(m)),
                Idx[8192](),
                Idx[7168](),
            )
