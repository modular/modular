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
from std.hashlib import default_comp_time_hasher
from std.math import align_up, ceildiv
from std.sys import argv, size_of
import std.itertools
import linalg.matmul.vendor.blas as vendor_blas
from std.gpu.host import DeviceContext
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from std.random import rand
from internal_utils import assert_almost_equal
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)
from linalg.matmul.gpu.sm100.block_scaled_matmul_small_bn import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    set_scale_factor,
)
from std.random import random_ui64
from std.builtin.simd import _convert_f32_to_float8_ue8m0
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
        m.value(),
        ", ",
        n.value(),
        ", ",
        k.value(),
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

    var a_shape = row_major(Coord(m, Idx[KType.static_value // 2]()))
    var b_shape = row_major(
        Coord(Idx[NType.static_value](), Idx[KType.static_value // 2]())
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]()))

    var a_size = m.value() * (KType.static_value // 2)
    var b_size = n.value() * (KType.static_value // 2)
    var c_size = m.value() * n.value()

    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = alloc[Scalar[c_type]](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device.unsafe_ptr(), a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device.unsafe_ptr(), b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device.unsafe_ptr(), c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref.unsafe_ptr(), c_shape)

    # This row major layout coorelates to this
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-scale-factor-a-layout-4x

    # Dim 0: the scale factors cover batches of 128 rows (4 sets of 32 rows to be specifc) so divide to find out how
    # tiles we have over the first mode

    # Dim 1: Assuming NVFP4_SF_VECTOR_SIZE for SF_VECTOR_SIZE, we know each scale factor covers 16 elements. The MMA has K fixed to 64 (32 in fp8),
    # so we divide K by 64 (4 scales) and we get the batch of scales for each mma across that mode.

    # Dim 2: Now in each batch as previosuly mentioned we have 32 rows
    # Dim 3: each column in the row is actually a subrow there are a total of 4 (32 * 4 gives us 128)
    # Dim 4: each subrow has 4 scale factors.

    var a_scales_shape = row_major(
        Coord(
            Idx(ceildiv(m.value(), SF_MN_GROUP_SIZE)),
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)](),
            Idx[SF_ATOM_M[0]](),
            Idx[SF_ATOM_M[1]](),
            Idx[SF_ATOM_K](),
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx(ceildiv(n.value(), SF_MN_GROUP_SIZE)),
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)](),
            Idx[SF_ATOM_M[0]](),
            Idx[SF_ATOM_M[1]](),
            Idx[SF_ATOM_K](),
        )
    )

    var a_scales_total = a_scales_shape.product()
    var b_scales_total = b_scales_shape.product()

    var a_scales_host_ptr = alloc[Scalar[scales_dtype]](a_scales_total)
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = alloc[Scalar[scales_dtype]](b_scales_total)
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_tensor = TileTensor(
        a_scales_device.unsafe_ptr(), a_scales_shape
    )
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_tensor = TileTensor(
        b_scales_device.unsafe_ptr(), b_scales_shape
    )

    # LayoutTensors for reference matmul (vendor_blas)
    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var a_scales_lt = a_scales_tensor.to_layout_tensor()
    var b_scales_lt = b_scales_tensor.to_layout_tensor()
    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()

    # Initialize matmul operands
    if simple_init():
        for m in range(m.value()):
            for k in range(k.value() // 2):
                comptime assert a_host.flat_rank >= 2
                a_host[(Idx(m), Idx(k))] = UInt8(m).cast[a_type]()
        for n in range(n.value()):
            for k in range(k.value() // 2):
                comptime assert b_host.flat_rank >= 2
                b_host[(Idx(n), Idx(k))] = UInt8(n).cast[b_type]()
    else:
        rand(a_host.ptr, a_host.num_elements(), min=0, max=255)
        rand(b_host.ptr, b_host.num_elements(), min=0, max=255)

    comptime a_scales_5d_layout = Layout.row_major(
        a_scales_tensor.static_shape[0],
        a_scales_tensor.static_shape[1],
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    comptime b_scales_5d_layout = Layout.row_major(
        b_scales_tensor.static_shape[0],
        b_scales_tensor.static_shape[1],
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )

    var a_scales_tensor_host = LayoutTensor[
        scales_dtype, a_scales_5d_layout, MutAnyOrigin
    ](
        a_scales_host_ptr,
        RuntimeLayout[a_scales_5d_layout].row_major(
            IndexList[5](
                Int(a_scales_host.dim(0)),
                Int(a_scales_host.dim(1)),
                Int(a_scales_host.dim(2)),
                Int(a_scales_host.dim(3)),
                Int(a_scales_host.dim(4)),
            ),
        ),
    )

    var b_scales_tensor_host = LayoutTensor[
        scales_dtype, b_scales_5d_layout, MutAnyOrigin
    ](
        b_scales_host_ptr,
        RuntimeLayout[b_scales_5d_layout].row_major(
            IndexList[5](
                Int(b_scales_host.dim(0)),
                Int(b_scales_host.dim(1)),
                Int(b_scales_host.dim(2)),
                Int(b_scales_host.dim(3)),
                Int(b_scales_host.dim(4)),
            ),
        ),
    )

    rand(a_scales_host.ptr, a_scales_host.num_elements())
    rand(b_scales_host.ptr, b_scales_host.num_elements())
    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
    for idx0 in range(align_up(m.value(), SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(k.value(), SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 >= m.value() or idx1 >= k.value():
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_tensor_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    for idx0 in range(align_up(n.value(), SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(k.value(), SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 >= n.value() or idx1 >= k.value():
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_tensor_host, idx0, idx1, Scalar[scales_dtype](0.0)
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
        c_ref_tensor_lt,
        a_lt,
        b_lt,
        a_scales=a_scales_lt.get_immutable(),
        b_scales=b_scales_lt.get_immutable(),
        transpose_b=transpose_b,
        c_row_major=True,
        alpha=alpha,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_almost_equal(
        c_host.ptr,
        c_host_ref.ptr,
        c_host.num_elements(),
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
    b_scales_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8  # TODO: (KERN-2238): Replace with float4-e2m1fn
        comptime out_dtype = DType.bfloat16
        comptime scales_dtype = NVFP4_SF_DTYPE
        comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32

        comptime for cta_group in [
            1,
        ]:
            comptime for bm in [128]:
                comptime for mma_n in [
                    8,
                    16,
                    32,
                ]:
                    # sfb_mode: 0 = cp.async, 1 = TMA
                    comptime for sfb_mode in [0, 1]:
                        comptime block_tile_shape = Index(
                            bm, mma_n // cta_group, BK
                        )
                        comptime umma_shape = Index(
                            cta_group * bm, mma_n, MMA_K
                        )

                        test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            out_dtype,
                            scales_dtype,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=8,
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                            use_cpasync_sfb=(sfb_mode == 0),
                        ](
                            ctx,
                            Idx(Int(8)),
                            Idx[16](),
                            Idx[256](),
                        )

                        test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            out_dtype,
                            scales_dtype,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=8,
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                            use_cpasync_sfb=(sfb_mode == 0),
                        ](
                            ctx,
                            Idx(Int(1000)),
                            Idx[1024](),
                            Idx[1024 + 32](),
                        )

                        test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            out_dtype,
                            scales_dtype,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=4,
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                            use_cpasync_sfb=(sfb_mode == 0),
                        ](
                            ctx,
                            Idx(Int(512)),
                            Idx[4096](),
                            Idx[1024 + 32](),
                        )

                        test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            out_dtype,
                            scales_dtype,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=0,
                            k_group_size=1,
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                            use_cpasync_sfb=(sfb_mode == 0),
                        ](
                            ctx,
                            Idx(Int(500)),
                            Idx[2048](),
                            Idx[4096](),
                        )

                        test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            out_dtype,
                            scales_dtype,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=2,
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                            use_cpasync_sfb=(sfb_mode == 0),
                        ](
                            ctx,
                            Idx(Int(999)),
                            Idx[256](),
                            Idx[128](),
                        )

                        test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            out_dtype,
                            scales_dtype,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=1,
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                            use_cpasync_sfb=(sfb_mode == 0),
                        ](
                            ctx,
                            Idx(Int(777)),
                            Idx[2560](),
                            Idx[8192](),
                            alpha=0.225,
                        )

                        test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            out_dtype,
                            scales_dtype,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=1,
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                            use_cpasync_sfb=(sfb_mode == 0),
                        ](
                            ctx,
                            Idx(Int(1)),
                            Idx[576](),
                            Idx[7168](),
                            alpha=0.5,
                        )

                        # swapAB tests
                        test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            out_dtype,
                            scales_dtype,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            swapAB=True,
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                            use_cpasync_sfb=(sfb_mode == 0),
                        ](
                            ctx,
                            Idx(Int(16)),
                            Idx[1024](),
                            Idx[1024 + 32](),
                        )

                        test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                            dtype,
                            dtype,
                            out_dtype,
                            scales_dtype,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            swapAB=True,
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                            use_cpasync_sfb=(sfb_mode == 0),
                        ](
                            ctx,
                            Idx(Int(100)),
                            Idx[2560](),
                            Idx[8192](),
                        )

        # Llama-3.1-405B TP8 FP4 shapes (small_bn kernel, M=1)
        comptime small_bn_block_tile = Index(128, 8, BK)
        comptime small_bn_umma = Index(128, 8, MMA_K)

        @parameter
        def test_small_bn[N: Int, K: Int]() raises:
            # sfb_mode: 0 = cp.async, 1 = TMA
            comptime for sfb_mode in [0, 1]:
                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    small_bn_block_tile,
                    small_bn_umma,
                    cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    swapAB=True,
                    k_group_size=2,
                    num_clc_pipeline_stages=0,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                    use_cpasync_sfb=(sfb_mode == 0),
                ](
                    ctx,
                    Idx(Int(1)),
                    Idx[N](),
                    Idx[K](),
                )

        test_small_bn[2304, 16384]()  # Attn.QKVProj
        test_small_bn[16384, 2048]()  # Attn.OutProj
        test_small_bn[6656, 16384]()  # MLP.UpProj / MLP.GateProj
        test_small_bn[13312, 16384]()  # Fused MLP.UpProj + MLP.GateProj
        test_small_bn[16384, 6656]()  # MLP.DownProj
        test_small_bn[7168, 16384]()  # Deepseek
