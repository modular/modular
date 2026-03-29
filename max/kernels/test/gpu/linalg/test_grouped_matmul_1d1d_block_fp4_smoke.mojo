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
"""Smoke test for grouped_matmul_1d1d block-scaled FP4 with TileTensor inputs.

Exercises the grouped 1D-1D block-scaled FP4 matmul kernel directly with
TileTensor arguments, parameterized by scaling kind to support different
block-scaled FP4 variants (e.g. NVFP4, MXFP4).
"""

from std.math import ceildiv
from std.gpu.host import DeviceContext
from std.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.random import rand
from layout import Coord, Idx, RuntimeInt, TileTensor, row_major
from std.utils.index import Index, IndexList

from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
)
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP4_SF_DTYPE,
    MXFP4_SF_VECTOR_SIZE,
)


def _test_grouped_1d1d_block_fp4_impl[
    num_experts: Int,
    N: Int,
    K: Int,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    cta_group: Int = 1,
    mma_n: Int = 128 * cta_group,
    AB_swapped: Bool = (cta_group == 2),
    sf_dtype: DType = NVFP4_SF_DTYPE,
    sf_vector_size: Int = NVFP4_SF_VECTOR_SIZE,
    scaling_kind: UMMAKind = UMMAKind.KIND_MXF4NVF4,
](ctx: DeviceContext, num_active_experts: Int, tokens_per_expert: Int) raises:
    comptime a_type = DType.uint8
    comptime b_type = DType.uint8
    comptime c_type = DType.bfloat16
    comptime packed_K = K // 2

    print(
        "  experts=",
        num_active_experts,
        "/",
        num_experts,
        " tokens=",
        tokens_per_expert,
        " N=",
        N,
        " K=",
        K,
        " mma_n=",
        mma_n,
    )

    var total_tokens = num_active_experts * tokens_per_expert

    # Offsets and expert IDs
    var a_offsets_host = alloc[Scalar[DType.uint32]](num_active_experts + 1)
    var a_scale_offsets_host = alloc[Scalar[DType.uint32]](num_active_experts)
    var expert_ids_host = alloc[Scalar[DType.int32]](num_active_experts)

    var a_scale_dim0 = 0
    a_offsets_host[0] = 0
    for i in range(num_active_experts):
        a_scale_offsets_host[i] = UInt32(
            a_scale_dim0 - Int(a_offsets_host[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(tokens_per_expert)
        a_scale_dim0 += ceildiv(tokens_per_expert, SF_MN_GROUP_SIZE)
        expert_ids_host[i] = Int32(i)

    # Host-side data init (rand for uint8 produces proper [0, 255] values)
    var a_host = alloc[Scalar[a_type]](total_tokens * packed_K)
    var b_host = alloc[Scalar[b_type]](num_experts * N * packed_K)
    rand(a_host, total_tokens * packed_K, min=0, max=255)
    rand(b_host, num_experts * N * packed_K, min=0, max=255)

    # Scale factors
    comptime k_groups = ceildiv(K, sf_vector_size * SF_ATOM_K)
    comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)
    var a_sf_size = (
        a_scale_dim0 * k_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    )
    var b_sf_size = (
        num_experts
        * n_groups
        * k_groups
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var a_sf_host = alloc[Scalar[sf_dtype]](a_sf_size)
    var b_sf_host = alloc[Scalar[sf_dtype]](b_sf_size)
    rand(a_sf_host, a_sf_size)
    rand(b_sf_host, b_sf_size)

    # Device buffers
    var a_buf = ctx.enqueue_create_buffer[a_type](total_tokens * packed_K)
    var b_buf = ctx.enqueue_create_buffer[b_type](num_experts * N * packed_K)
    var c_buf = ctx.enqueue_create_buffer[c_type](total_tokens * N)
    var a_off_buf = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts + 1
    )
    var a_soff_buf = ctx.enqueue_create_buffer[DType.uint32](num_active_experts)
    var eid_buf = ctx.enqueue_create_buffer[DType.int32](num_active_experts)
    var a_sf_buf = ctx.enqueue_create_buffer[sf_dtype](a_sf_size)
    var b_sf_buf = ctx.enqueue_create_buffer[sf_dtype](b_sf_size)

    # Copy to device
    ctx.enqueue_copy(a_buf, a_host)
    ctx.enqueue_copy(b_buf, b_host)
    ctx.enqueue_copy(a_off_buf, a_offsets_host)
    ctx.enqueue_copy(a_soff_buf, a_scale_offsets_host)
    ctx.enqueue_copy(eid_buf, expert_ids_host)
    ctx.enqueue_copy(a_sf_buf, a_sf_host)
    ctx.enqueue_copy(b_sf_buf, b_sf_host)

    # Expert scales
    var es_buf = ctx.enqueue_create_buffer[DType.float32](num_experts)
    var es_host = alloc[Scalar[DType.float32]](num_experts)
    for i in range(num_experts):
        es_host[i] = 1.0
    ctx.enqueue_copy(es_buf, es_host)

    # Construct TileTensors directly from pointers and layouts
    var a_tt = TileTensor(
        a_buf.unsafe_ptr(),
        row_major(Coord(Idx(Int(total_tokens)), Idx[packed_K]())),
    )
    var b_tt = TileTensor(
        b_buf.unsafe_ptr(),
        row_major(Coord(Idx[num_experts](), Idx[N](), Idx[packed_K]())),
    )
    var c_tt = TileTensor(
        c_buf.unsafe_ptr(),
        row_major(Coord(Idx(Int(total_tokens)), Idx[N]())),
    )
    var a_offsets_tt = TileTensor(
        a_off_buf.unsafe_ptr(),
        row_major(
            Idx(
                Int(num_active_experts + 1),
            )
        ),
    )
    var a_scale_offsets_tt = TileTensor(
        a_soff_buf.unsafe_ptr(),
        row_major(
            Idx(
                Int(num_active_experts),
            )
        ),
    )
    var expert_ids_tt = TileTensor(
        eid_buf.unsafe_ptr(),
        row_major(
            Idx(
                Int(num_active_experts),
            )
        ),
    )
    var expert_scales_tt = TileTensor(
        es_buf.unsafe_ptr(),
        row_major(
            Idx[num_experts](),
        ),
    )

    # Scale factor TileTensors (5D and 6D)
    var a_scales_tt = TileTensor(
        a_sf_buf.unsafe_ptr().bitcast[Scalar[sf_dtype]](),
        row_major(
            Coord(
                RuntimeInt[DType.int64](Scalar[DType.int64](a_scale_dim0)),
                Idx[k_groups](),
                Idx[SF_ATOM_M[0]](),
                Idx[SF_ATOM_M[1]](),
                Idx[SF_ATOM_K](),
            )
        ),
    ).as_any_origin()
    var b_scales_tt = TileTensor(
        b_sf_buf.unsafe_ptr().bitcast[Scalar[sf_dtype]](),
        row_major(
            Coord(
                Idx[num_experts](),
                Idx[n_groups](),
                Idx[k_groups](),
                Idx[SF_ATOM_M[0]](),
                Idx[SF_ATOM_M[1]](),
                Idx[SF_ATOM_K](),
            )
        ),
    ).as_any_origin()

    # Launch kernel
    comptime mma_shape = Index(128 * cta_group, mma_n, 32)
    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, sf_dtype, sf_dtype, True
    ](
        scaling_kind=scaling_kind,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        block_swizzle_size=0,
        cta_group=cta_group,
        AB_swapped=AB_swapped,
        k_group_size=1,
        num_accum_pipeline_stages=1 if mma_shape[1] == 256 else 2,
        is_gmm=True,
    )

    grouped_matmul_block_scaled[transpose_b=True, config=config](
        c_tt,
        a_tt,
        a_offsets_tt,
        a_scale_offsets_tt,
        b_tt,
        expert_ids_tt,
        a_scales_tt,
        b_scales_tt,
        expert_scales_tt,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    print("    PASSED")

    a_host.free()
    b_host.free()
    a_sf_host.free()
    b_sf_host.free()
    a_offsets_host.free()
    a_scale_offsets_host.free()
    expert_ids_host.free()
    es_host.free()
    _ = a_buf^
    _ = b_buf^
    _ = c_buf^
    _ = a_off_buf^
    _ = a_soff_buf^
    _ = eid_buf^
    _ = a_sf_buf^
    _ = b_sf_buf^
    _ = es_buf^


def run_grouped_1d1d_block_fp4_smoke_suite[
    sf_dtype: DType,
    sf_vector_size: Int,
    scaling_kind: UMMAKind,
]() raises:
    var ctx = DeviceContext()

    @parameter
    @always_inline
    def test_grouped_1d1d_block_fp4[
        num_experts: Int,
        N: Int,
        K: Int,
        cluster_shape: IndexList[3] = Index(1, 1, 1),
        cta_group: Int = 1,
        mma_n: Int = 128 * cta_group,
        AB_swapped: Bool = (cta_group == 2),
    ](
        ctx: DeviceContext, num_active_experts: Int, tokens_per_expert: Int
    ) raises:
        _test_grouped_1d1d_block_fp4_impl[
            num_experts,
            N,
            K,
            cluster_shape,
            cta_group,
            mma_n,
            AB_swapped,
            sf_dtype=sf_dtype,
            sf_vector_size=sf_vector_size,
            scaling_kind=scaling_kind,
        ](ctx, num_active_experts, tokens_per_expert)

    print("=== Grouped 1D1D NVFP4 Smoke Tests (TileTensor) ===")
    test_grouped_1d1d_block_fp4[4, 128, 256](ctx, 4, 64)
    test_grouped_1d1d_block_fp4[8, 128, 256](ctx, 4, 64)
    test_grouped_1d1d_block_fp4[4, 1024, 1024](ctx, 2, 128)

    print("\n=== Grouped 1D1D NVFP4 MMA_N=64 1SM Smoke Tests (TileTensor) ===")
    test_grouped_1d1d_block_fp4[4, 128, 256, mma_n=64](ctx, 4, 64)
    test_grouped_1d1d_block_fp4[4, 1024, 1024, mma_n=64](ctx, 2, 128)

    print(
        "\n=== Grouped 1D1D NVFP4 MMA_N=64 AB_swapped 1SM Smoke Tests"
        " (TileTensor) ==="
    )
    test_grouped_1d1d_block_fp4[4, 128, 256, mma_n=64, AB_swapped=True](
        ctx, 4, 64
    )
    test_grouped_1d1d_block_fp4[4, 1024, 1024, mma_n=64, AB_swapped=True](
        ctx, 2, 128
    )

    print(
        "\n=== Grouped 1D1D NVFP4 MMA_N=64 AB_swapped 2SM Smoke Tests"
        " (TileTensor) ==="
    )
    # 2SM with mma_n=64: UMMA shape (256, 128, 32), BM=128, BN=64
    test_grouped_1d1d_block_fp4[4, 2048, 1024, Index(2, 1, 1), 2, 64](
        ctx, 4, 64
    )
    test_grouped_1d1d_block_fp4[4, 2048, 1024, Index(2, 1, 1), 2, 64](
        ctx, 2, 256
    )

    print("\n=== Grouped 1D1D NVFP4 2SM Smoke Tests (TileTensor) ===")
    test_grouped_1d1d_block_fp4[4, 2048, 1024, Index(2, 1, 1), 2](ctx, 4, 64)
    test_grouped_1d1d_block_fp4[4, 2048, 1024, Index(2, 1, 1), 2](ctx, 2, 256)

    print("\n=== Grouped 1D1D NVFP4 MMA_N=8 Smoke Tests (TileTensor) ===")
    test_grouped_1d1d_block_fp4[6, 2048, 1024, mma_n=8](ctx, 1, 512)
    test_grouped_1d1d_block_fp4[6, 2048, 1024, mma_n=8](ctx, 1, 129)
    test_grouped_1d1d_block_fp4[1, 128, 256, mma_n=8](ctx, 1, 128)

    print(
        "\n=== Grouped 1D1D NVFP4 MMA_N=8 AB_swapped Smoke Tests"
        " (TileTensor) ==="
    )
    test_grouped_1d1d_block_fp4[6, 2048, 1024, mma_n=8, AB_swapped=True](
        ctx, 1, 512
    )
    test_grouped_1d1d_block_fp4[6, 2048, 1024, mma_n=8, AB_swapped=True](
        ctx, 1, 129
    )
    test_grouped_1d1d_block_fp4[1, 128, 256, mma_n=8, AB_swapped=True](
        ctx, 1, 128
    )

    print("\n=== Grouped 1D1D NVFP4 MMA_N=16 Smoke Tests (TileTensor) ===")
    test_grouped_1d1d_block_fp4[4, 1024, 1024, mma_n=16](ctx, 2, 128)
    test_grouped_1d1d_block_fp4[1, 128, 256, mma_n=16](ctx, 1, 128)

    print(
        "\n=== Grouped 1D1D NVFP4 MMA_N=16 AB_swapped Smoke Tests"
        " (TileTensor) ==="
    )
    test_grouped_1d1d_block_fp4[4, 1024, 1024, mma_n=16, AB_swapped=True](
        ctx, 2, 128
    )
    test_grouped_1d1d_block_fp4[1, 128, 256, mma_n=16, AB_swapped=True](
        ctx, 1, 128
    )

    print("\n=== Grouped 1D1D NVFP4 MMA_N=32 Smoke Tests (TileTensor) ===")
    test_grouped_1d1d_block_fp4[4, 1024, 1024, mma_n=32](ctx, 2, 128)
    test_grouped_1d1d_block_fp4[1, 128, 256, mma_n=32](ctx, 1, 128)

    print(
        "\n=== Grouped 1D1D NVFP4 MMA_N=32 AB_swapped Smoke Tests"
        " (TileTensor) ==="
    )
    test_grouped_1d1d_block_fp4[4, 1024, 1024, mma_n=32, AB_swapped=True](
        ctx, 2, 128
    )
    test_grouped_1d1d_block_fp4[1, 128, 256, mma_n=32, AB_swapped=True](
        ctx, 1, 128
    )

    print("=== ALL TESTS PASSED ===")
    _ = ctx^


def main() raises:
    run_grouped_1d1d_block_fp4_smoke_suite[
        sf_dtype=NVFP4_SF_DTYPE,
        sf_vector_size=NVFP4_SF_VECTOR_SIZE,
        scaling_kind=UMMAKind.KIND_MXF4NVF4,
    ]()
    run_grouped_1d1d_block_fp4_smoke_suite[
        sf_dtype=MXFP4_SF_DTYPE,
        sf_vector_size=MXFP4_SF_VECTOR_SIZE,
        scaling_kind=UMMAKind.KIND_MXF4,
    ]()
