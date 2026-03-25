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
"""2SM (cta_group=2) tests for block_scaled_matmul_small_bn.

Tests the 2CTA cooperative MMA path where two SMs work on a single MMA
instruction. Each CTA loads its own BN=MMA_N/2 columns of B data but
the full MMA_N scale factors.
"""
from std.math import align_up, ceildiv
from std.sys import argv, size_of
import std.itertools
import linalg.matmul.vendor.blas as vendor_blas
from std.gpu.host import DeviceContext
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from std.random import rand
from internal_utils import assert_almost_equal
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100.block_scaled_matmul_small_bn import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    set_scale_factor,
)
from std.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    benchmark: Bool = False,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    num_clc_pipeline_stages: Int = 2,
    SF_VECTOR_SIZE: Int = NVFP4_SF_VECTOR_SIZE,
    use_cpasync_sfb: Optional[Bool] = None,
](
    ctx: DeviceContext,
    m: MType,
    n: NType,
    k: KType,
    alpha: Float32 = 1.0,
) raises:
    var M = m.value()
    var N = n.value()
    var K = k.value()

    print(
        "in/out dtypes=(",
        a_type,
        ", ",
        b_type,
        ", ",
        c_type,
        ", ",
        scales_dtype,
        ") ",
        " problem shape=(",
        M,
        ", ",
        N,
        ", ",
        K,
        ") ",
        "mma_shape=",
        mma_shape,
        " block_tile_shape=",
        block_tile_shape,
        " cta_group=",
        cta_group,
        " cluster_shape=(",
        cluster_shape[0],
        ", ",
        cluster_shape[1],
        ", ",
        cluster_shape[2],
        ")",
        " swapAB=",
        swapAB,
        " k_group_size=",
        k_group_size,
        " SF_VECTOR_SIZE=",
        SF_VECTOR_SIZE,
        " sfb_mode=",
        "cpasync" if (
            use_cpasync_sfb.value() if use_cpasync_sfb else (
                mma_shape[1] < SF_MN_GROUP_SIZE
            )
        ) else "tma",
        sep=" ",
    )

    var a_shape = Coord(m, Idx[KType.static_value // 2]())
    var b_shape = Coord(n, Idx[KType.static_value // 2]())
    var c_shape = Coord(m, n)

    var a_size = M * (K // 2)
    var b_size = N * (K // 2)
    var c_size = M * N

    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var c_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_host_ref_ptr = alloc[Scalar[c_type]](c_size)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)

    # This row major layout coorelates to this
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-scale-factor-a-layout-4x

    # Dim 0: the scale factors cover batches of 128 rows (4 sets of 32 rows to be specifc) so divide to find out how
    # tiles we have over the first mode

    # Dim 1: Assuming NVFP4_SF_VECTOR_SIZE for SF_VECTOR_SIZE, we know each scale factor covers 16 elements. The MMA has K fixed to 64 (32 in fp8),
    # so we divide K by 64 (4 scales) and we get the batch of scales for each mma across that mode.

    # Dim 2: Now in each batch as previosuly mentioned we have 32 rows
    # Dim 3: each column in the row is actually a subrow there are a total of 4 (32 * 4 gives us 128)
    # Dim 4: each subrow has 4 scale factors.

    var a_scales_shape = Coord(
        Idx(ceildiv(M, SF_MN_GROUP_SIZE)),
        Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)](),
        Idx[SF_ATOM_M[0]](),
        Idx[SF_ATOM_M[1]](),
        Idx[SF_ATOM_K](),
    )
    var b_scales_shape = Coord(
        Idx(ceildiv(N, SF_MN_GROUP_SIZE)),
        Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)](),
        Idx[SF_ATOM_M[0]](),
        Idx[SF_ATOM_M[1]](),
        Idx[SF_ATOM_K](),
    )

    var a_scales_total = (
        ceildiv(M, SF_MN_GROUP_SIZE)
        * ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_scales_total = (
        ceildiv(N, SF_MN_GROUP_SIZE)
        * ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var a_scales_host_ptr = alloc[Scalar[scales_dtype]](a_scales_total)
    var b_scales_host_ptr = alloc[Scalar[scales_dtype]](b_scales_total)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )

    var a_tensor = TileTensor(a_device.unsafe_ptr(), row_major(a_shape))
    var b_tensor = TileTensor(b_device.unsafe_ptr(), row_major(b_shape))
    var c_tensor = TileTensor(c_device.unsafe_ptr(), row_major(c_shape))
    var a_scales_tensor = TileTensor(
        a_scales_device.unsafe_ptr(), row_major(a_scales_shape)
    )
    var b_scales_tensor = TileTensor(
        b_scales_device.unsafe_ptr(), row_major(b_scales_shape)
    )
    var c_ref_tensor = TileTensor(c_device_ref.unsafe_ptr(), row_major(c_shape))

    # Initialize matmul operands
    if simple_init():
        var a_host_tt = TileTensor(a_host_ptr, row_major(a_shape))
        var b_host_tt = TileTensor(b_host_ptr, row_major(b_shape))
        comptime assert a_host_tt.flat_rank == 2
        comptime assert b_host_tt.flat_rank == 2
        for m in range(M):
            for k in range(K // 2):
                a_host_tt[m, k] = UInt8(m).cast[a_type]()
        for n in range(N):
            for k in range(K // 2):
                b_host_tt[n, k] = UInt8(n).cast[b_type]()
    else:
        rand(a_host_ptr, a_size, min=0, max=255)
        rand(b_host_ptr, b_size, min=0, max=255)

    var a_scales_host_tt = TileTensor(
        a_scales_host_ptr, row_major(a_scales_shape)
    )
    var b_scales_host_tt = TileTensor(
        b_scales_host_ptr, row_major(b_scales_shape)
    )

    rand(a_scales_host_ptr, a_scales_total)
    rand(b_scales_host_ptr, b_scales_total)
    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
    for idx0 in range(align_up(M, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 >= M or idx1 >= K:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host_tt, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    for idx0 in range(align_up(N, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 >= N or idx1 >= K:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host_tt, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    comptime matmul_config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF4NVF4,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_accum_pipeline_stages=1 if mma_shape[1] in (192, 256) else 2,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        use_cpasync_sfb=use_cpasync_sfb,
        is_small_bn=True,
    )

    comptime K_phys = KType.static_value
    blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
        K=K_phys,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        a_scales_tensor,
        b_scales_tensor,
        ctx,
        alpha,
    )

    vendor_blas.matmul(
        ctx,
        c_ref_tensor,
        a_tensor,
        b_tensor,
        a_scales=a_scales_tensor.as_immut(),
        b_scales=b_scales_tensor.as_immut(),
        transpose_b=transpose_b,
        c_row_major=True,
        alpha=alpha,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_almost_equal(
        c_host_ptr,
        c_host_ref_ptr,
        c_size,
        atol=1e-2,
        rtol=1e-2,
    )
    print("\n=== TEST PASSED ===\n")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scales_host_ptr.free()


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8
        comptime out_dtype = DType.bfloat16
        comptime scales_dtype = NVFP4_SF_DTYPE
        comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32

        # 2SM tests: sweep MMA_N in [16, 32], both sfb modes.
        # Note: MMA_N=24 is valid HW but BN=12 breaks TMA tile layout.
        comptime for mma_n in [16, 32]:
            comptime block_tile = Index(128, mma_n // 2, BK)
            comptime umma = Index(256, mma_n, MMA_K)

            # sfb_mode: 0 = cp.async, 1 = TMA
            comptime for sfb_mode in [0, 1]:
                # Basic cluster_shape=(2,1,1)
                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile,
                    umma,
                    cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    swapAB=True,
                    k_group_size=2,
                    num_clc_pipeline_stages=0,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                    use_cpasync_sfb=(sfb_mode == 0),
                ](ctx, Idx(1), Idx[2304](), Idx[16384]())

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile,
                    umma,
                    cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    swapAB=True,
                    k_group_size=2,
                    num_clc_pipeline_stages=0,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                    use_cpasync_sfb=(sfb_mode == 0),
                ](ctx, Idx(1), Idx[16384](), Idx[2048]())

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile,
                    umma,
                    cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    swapAB=True,
                    k_group_size=2,
                    num_clc_pipeline_stages=0,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                    use_cpasync_sfb=(sfb_mode == 0),
                ](ctx, Idx(1), Idx[6656](), Idx[16384]())

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile,
                    umma,
                    cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    swapAB=True,
                    k_group_size=2,
                    num_clc_pipeline_stages=0,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                    use_cpasync_sfb=(sfb_mode == 0),
                ](ctx, Idx(1), Idx[16384](), Idx[6656]())

                # Larger cluster shapes
                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile,
                    umma,
                    cluster_shape=StaticTuple[Int32, 3](Int32(4), 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=4,
                    swapAB=True,
                    k_group_size=2,
                    num_clc_pipeline_stages=0,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                    use_cpasync_sfb=(sfb_mode == 0),
                ](ctx, Idx(1), Idx[2304](), Idx[16384]())

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile,
                    umma,
                    cluster_shape=StaticTuple[Int32, 3](Int32(2), Int32(2), 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=4,
                    swapAB=True,
                    k_group_size=2,
                    num_clc_pipeline_stages=0,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                    use_cpasync_sfb=(sfb_mode == 0),
                ](ctx, Idx(1), Idx[6656](), Idx[16384]())
