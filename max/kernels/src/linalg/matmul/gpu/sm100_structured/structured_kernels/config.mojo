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
"""SM100 matmul configuration types and utilities.

This module provides configuration structs for SM100 (Blackwell) GPU matmul
operations, including standard matmul and block-scaled matmul variants.
"""

from bit import next_power_of_two
from collections.set import Set
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from itertools.itertools import product
from layout.tensor_core import get_mma_shape
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.math import align_down
from ...tile_scheduler import RasterOrder
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_K_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_VECTOR_SIZE,
    MXFP8_SF_VECTOR_SIZE,
)
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


# ============================================================================
# Shared Configuration Helpers
# ============================================================================


fn _compute_block_tile_shape[
    a_type: DType
](mma_shape: IndexList[3], cta_group: Int) -> IndexList[3]:
    """Compute block tile shape from MMA shape and CTA group."""
    return Index(
        mma_shape[0] // cta_group,
        mma_shape[1] // cta_group,
        128 // size_of[a_type](),
    )


fn _compute_output_tile_shape(
    mma_shape: IndexList[3], cta_group: Int, AB_swapped: Bool
) -> IndexList[2]:
    """Compute output tile shape based on MMA config."""
    # If MMA_M is 256, each of the pair ctas has the entire MMA_N.
    # If MMA_M is 128, each of the pair ctas has 1/2 of MMA_N.
    # If cta_group=1, the cta has the entire MMA_N.
    var c_tile_n = mma_shape[1] if (
        mma_shape[0] == 256 or cta_group == 1
    ) else (mma_shape[1] // 2)
    var output_tile_n = 8
    if c_tile_n % 32 == 0:
        output_tile_n = 32
    elif c_tile_n % 16 == 0:
        output_tile_n = 16
    # MMA_M=128/256 cta_group=2 all use 128 rows in output tile.
    var output_tile_m = 128 if cta_group == 2 else mma_shape[0]
    return Index(output_tile_n, output_tile_m) if AB_swapped else Index(
        output_tile_m, output_tile_n
    )


fn _compute_swizzle_modes(
    output_tile_shape: IndexList[2], AB_swapped: Bool
) -> Tuple[TensorMapSwizzle, TensorMapSwizzle, TensorMapSwizzle]:
    """Compute A, B, C swizzle modes."""
    var a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    var b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    var c_swizzle = TensorMapSwizzle.SWIZZLE_NONE
    if AB_swapped:
        c_swizzle = TensorMapSwizzle.SWIZZLE_128B
    else:
        # When not swapped, output_tile_shape[1] is the N dimension
        var tile_n = output_tile_shape[1]
        if tile_n == 32:
            c_swizzle = TensorMapSwizzle.SWIZZLE_64B
        elif tile_n == 16:
            c_swizzle = TensorMapSwizzle.SWIZZLE_32B
    return (a_swizzle, b_swizzle, c_swizzle)


fn _maximize_pipeline_stages[
    a_type: DType, b_type: DType, c_type: DType
](
    block_tile_shape: IndexList[3],
    output_tile_shape: IndexList[2],
    num_output_stages: Int,
    num_clc_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    extra_smem_per_stage: Int = 0,
) -> Int:
    """Calculate max pipeline stages based on shared memory budget."""
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    var c_smem_bytes = (
        output_tile_shape[0]
        * output_tile_shape[1]
        * num_output_stages
        * size_of[c_type]()
    )
    # Add tmem addr (4 bytes) and tmem dealloc mbar (8 bytes).
    var output_smem_bytes = c_smem_bytes + 12
    # Response 128B, clc mbar 16B, clc-load pipeline mbar 16B.
    var clc_smem_bytes = 160 * num_clc_pipeline_stages
    # Usage by mma-output-pipeline.
    var mma_output_smem_bytes = num_accum_pipeline_stages * 16

    var a_smem_bytes_per_stage = (
        block_tile_shape[0] * block_tile_shape[2] * size_of[a_type]()
    )
    var b_smem_bytes_per_stage = (
        block_tile_shape[1] * block_tile_shape[2] * size_of[b_type]()
    )
    # Include 16 bytes for consumer and producer mbar per stage.
    var AB_smem_per_stage = (
        a_smem_bytes_per_stage
        + b_smem_bytes_per_stage
        + 16
        + extra_smem_per_stage
    )

    return (
        b200_smem - output_smem_bytes - clc_smem_bytes - mma_output_smem_bytes
    ) // AB_smem_per_stage


fn _write_common_config[
    W: Writer,
    a_type: DType,
    c_type: DType,
    transpose_b: Bool,
](
    mut writer: W,
    cta_group: Int,
    mma_shape: IndexList[3],
    cluster_shape: IndexList[3],
    num_pipeline_stages: Int,
    k_group_size: Int,
    num_clc_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    num_output_stages: Int,
    output_tile_shape: IndexList[2],
    AB_swapped: Bool,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    c_swizzle: TensorMapSwizzle,
    block_swizzle_size: Int,
    raster_order: RasterOrder,
    num_split_k: Int,
):
    """Write common config fields to string."""
    writer.write(a_type, "_")
    writer.write(c_type, "_")
    writer.write("cta", cta_group, "_")
    writer.write("mma", mma_shape[0], "x", mma_shape[1], "x", mma_shape[2], "_")
    writer.write(
        "cluster",
        cluster_shape[0],
        "x",
        cluster_shape[1],
        "x",
        cluster_shape[2],
        "_",
    )
    writer.write("stages", num_pipeline_stages, "_")
    writer.write("k_group", k_group_size, "_")
    writer.write("clc", num_clc_pipeline_stages, "_")
    writer.write("accum", num_accum_pipeline_stages, "_")
    writer.write("out", num_output_stages, "_")
    writer.write(output_tile_shape[0], "x", output_tile_shape[1], "_")
    writer.write("swap" if AB_swapped else "noswap", "_")
    writer.write("K_" if transpose_b else "MN_")
    writer.write("asz", a_swizzle.bytes(), "_")
    writer.write("bsz", b_swizzle.bytes(), "_")
    writer.write("csz", c_swizzle.bytes(), "_")
    writer.write("bz", block_swizzle_size, "_", raster_order)
    writer.write("splitk", num_split_k, "_")


@fieldwise_init
struct MatmulConfig[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](Copyable, Equatable, Hashable, Stringable, TrivialRegisterPassable, Writable):
    """Static configuration of GPU matmul."""

    # Mandatory parameters
    var cta_group: Int
    var mma_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var AB_swapped: Bool
    var block_swizzle_size: Int
    var raster_order: RasterOrder

    comptime accum_type = get_accum_type[Self.a_type]()  # TODO: factor b_type

    # Has default values or derivible from mandatory parameters
    var block_tile_shape: IndexList[3]
    var num_split_k: Int
    var num_pipeline_stages: Int
    var num_clc_pipeline_stages: Int
    var num_accum_pipeline_stages: Int
    var num_output_stages: Int
    var output_tile_shape: IndexList[2]
    var a_swizzle: TensorMapSwizzle
    var b_swizzle: TensorMapSwizzle
    var c_swizzle: TensorMapSwizzle

    var k_group_size: Int

    fn __init__(
        out self,
        *,
        cta_group: Int = 2,
        mma_shape: IndexList[3] = get_mma_shape[Self.a_type, Self.accum_type](),
        cluster_shape: IndexList[3] = Index(2, 1, 1),
        AB_swapped: Bool = False,
        num_split_k: Int = 1,
        block_swizzle_size: Int = 0,
        raster_order: RasterOrder = RasterOrder.AlongM,
        k_group_size: Int = 1,
        num_pipeline_stages: Optional[Int] = None,
        num_accum_pipeline_stages: Int = 2,
        num_clc_pipeline_stages: Int = 2,
        extra_smem_per_stage: Int = 0,
    ):
        constrained[Self.a_type == Self.b_type]()

        self.cta_group = cta_group
        self.mma_shape = mma_shape
        self.cluster_shape = cluster_shape
        self.AB_swapped = AB_swapped
        self.block_swizzle_size = block_swizzle_size
        self.raster_order = raster_order
        self.k_group_size = k_group_size

        self.block_tile_shape = _compute_block_tile_shape[Self.a_type](
            mma_shape, cta_group
        )
        self.output_tile_shape = _compute_output_tile_shape(
            mma_shape, cta_group, AB_swapped
        )

        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.num_output_stages = 2
        self.num_split_k = num_split_k

        var swizzles = _compute_swizzle_modes(
            self.output_tile_shape, AB_swapped
        )
        self.a_swizzle = swizzles[0]
        self.b_swizzle = swizzles[1]
        self.c_swizzle = swizzles[2]

        self.num_pipeline_stages = _maximize_pipeline_stages[
            Self.a_type, Self.b_type, Self.c_type
        ](
            self.block_tile_shape,
            self.output_tile_shape,
            self.num_output_stages,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            extra_smem_per_stage,
        )

        if num_pipeline_stages:
            self.num_pipeline_stages = num_pipeline_stages.value()

        # SM100 kernel only supports k grouping when num_pipeline_stages is a multiple of k_group_size.
        self.num_pipeline_stages = align_down(
            self.num_pipeline_stages, self.k_group_size
        )

    fn swap_AB_type(
        self,
    ) -> MatmulConfig[Self.b_type, Self.a_type, Self.c_type, Self.transpose_b]:
        return MatmulConfig[
            Self.b_type, Self.a_type, Self.c_type, Self.transpose_b
        ](
            cta_group=self.cta_group,
            mma_shape=self.mma_shape,
            cluster_shape=self.cluster_shape,
            AB_swapped=self.AB_swapped,
            num_pipeline_stages=self.num_pipeline_stages,
            num_accum_pipeline_stages=self.num_accum_pipeline_stages,
            num_clc_pipeline_stages=self.num_clc_pipeline_stages,
            block_swizzle_size=self.block_swizzle_size,
            raster_order=self.raster_order,
            k_group_size=self.k_group_size,
            num_split_k=self.num_split_k,
        )

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("kernel_")
        _write_common_config[W, Self.a_type, Self.c_type, Self.transpose_b](
            writer,
            self.cta_group,
            self.mma_shape,
            self.cluster_shape,
            self.num_pipeline_stages,
            self.k_group_size,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            self.num_output_stages,
            self.output_tile_shape,
            self.AB_swapped,
            self.a_swizzle,
            self.b_swizzle,
            self.c_swizzle,
            self.block_swizzle_size,
            self.raster_order,
            self.num_split_k,
        )

    fn __repr__(self) -> String:
        return String.write(self)


fn choose_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](M: Int, N: Int, K: Int) -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    constrained[a_type == b_type, "a_type and b_type must be the same"]()

    comptime num_SMs = B200.sm_count
    # Nvidia mma instruction process 32B in K.
    comptime Kbytes_per_mma = 32
    # We use 128B swizzle, tile size in K is 128B over element size.
    comptime BK = 128 // size_of[a_type]()

    comptime M_pivote = 32

    var cta_group = 1 if M < M_pivote else 2
    var swapAB = True if M < M_pivote else False
    var k_group_size = 1  # maybe increased for small M later

    var mma_mn = Tuple[Int, Int](256, 256)
    var min_num_waves = Int.MAX

    # Traverse possible combinations of BM x MMA_N to choose the one minimizes the
    # workload per SM. The computation per SM is the flops (ignoring 2x in 2MNK)
    # timed by max number of ctas per SM i.e. number of waves.
    # We first minimize the number of waves, then use the flops to break tie.

    # For small M, swap A and B so that the small M maps to mma_n since it supports
    # a larger range than mma_m.
    if M < M_pivote:
        for bm, mma_n in product([64, 128], range(8, align_up(M, 8) + 1, 8)):
            num_ctas = ceildiv(M, mma_n) * ceildiv(N, bm)
            num_waves = ceildiv(num_ctas, num_SMs)
            if num_waves < min_num_waves or (
                num_waves == min_num_waves
                and bm * mma_n < mma_mn[0] * mma_mn[1]
            ):
                min_num_waves = num_waves
                mma_mn[0] = bm
                mma_mn[1] = mma_n

    # For large M, use 2xSM mma
    else:

        @parameter
        @always_inline
        fn select_mma_mn(M: Int, N: Int, _swapAB: Bool = False):
            N_alignby16 = align_up(N, 16)
            max_mma_n = min(N_alignby16, 256)
            # In pratice 64x16 mma creates too many ctas and increase L2
            # load volume, ends up hurting performance.
            min_mma_n = min(N_alignby16, 32)
            for bm in [64, 128]:
                for mma_n in range(max_mma_n, min_mma_n - 1, -16):
                    var mma_m = bm * cta_group
                    var num_clusters = ceildiv(M, mma_m) * ceildiv(N, mma_n)
                    var num_waves = ceildiv(num_clusters, num_SMs // cta_group)
                    if num_waves > min_num_waves:
                        break
                    elif num_waves < min_num_waves or (
                        num_waves == min_num_waves
                        and mma_m * mma_n < mma_mn[0] * mma_mn[1]
                    ):
                        min_num_waves = num_waves
                        mma_mn[0] = mma_m
                        mma_mn[1] = mma_n
                        swapAB = _swapAB

        # Swap AB may work better for M = 192 and not-multiple-of-128 values.
        # Capture and update min_num_waves, mma_mn
        select_mma_mn(M, N)
        select_mma_mn(N, M, True)

    # For small mmas, we group multiple tiles per tma-mma synchronization.
    var output_block_size = (mma_mn[0] // cta_group) * mma_mn[1]
    if output_block_size <= 64 * 96 and ceildiv(K, BK) % 2 == 0:
        k_group_size = 2
    # For very small mmas we can group more aggressively.
    if output_block_size <= 64 * 16 and ceildiv(K, BK) % 4 == 0:
        k_group_size = 4

    var min_load_volume = Int.MAX
    var optimal_block_swizzle_size = 0

    # Tile waves when there are >= 4 waves. In theory it should be >=2, but let's
    # be conservative.
    if min_num_waves >= 4:
        # Represent the load volume by
        #    BM * num_ctas_per_wave_m + MMA_N * num_ctas_per_wave_N
        # Use MMA_N because cta_group = 2, 2 ctas cover entire MMA_N. cta_group = 1
        # has BN = MMA_N.
        # Traverse the tile sizes to find min load volume per wave.
        # TODO: consider the L2 resue across waves.
        var BM = mma_mn[0] // cta_group
        for tile_size in [1, 2, 4, 8]:
            var num_ctas_m = ceildiv(M, BM)
            # When tile_size is small, it's possible that a wave has more ctas
            # then num_ctas_m * tile_size and num_ctas_per_wave_m > num_ctas_m.
            # The ctas mapping will "wrap around" and include following tile_sizes.
            var num_ctas_per_wave_m = ceildiv(num_SMs, tile_size)
            var num_ctas_per_wave_n = tile_size * ceildiv(
                num_ctas_per_wave_m, num_ctas_m
            )
            num_ctas_per_wave_m = min(num_ctas_per_wave_m, num_ctas_m)
            var load_volume_per_wave = (
                num_ctas_per_wave_m * BM + num_ctas_per_wave_n * mma_mn[1]
            )
            if load_volume_per_wave < min_load_volume:
                min_load_volume = load_volume_per_wave
                optimal_block_swizzle_size = tile_size

    # TODO: evaluate the comment's perf impact
    # var num_clc_pipeline_stages: UInt = UInt(min(min_num_waves-1, 2))
    var num_clc_pipeline_stages = 0 if min_num_waves == 1 else 2

    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        mma_shape=IndexList[3](
            mma_mn[0], mma_mn[1], Kbytes_per_mma // size_of[a_type]()
        ),
        cta_group=cta_group,
        cluster_shape=Index(cta_group, 1, 1),
        AB_swapped=swapAB,
        block_swizzle_size=optimal_block_swizzle_size,
        num_accum_pipeline_stages=min(2, min_num_waves),
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        k_group_size=k_group_size,
    )


fn build_configs[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
]() -> Set[MatmulConfig[a_type, b_type, c_type, transpose_b]]:
    comptime config_t = MatmulConfig[a_type, b_type, c_type, transpose_b]

    var set = Set[config_t]()

    for m in range(8, 128, 8):  # [8, 128]
        config = choose_config[a_type, b_type, c_type, transpose_b](m, N, K)
        if config not in set:
            set.add(config)

    for m in range(128, 8193, 64):  # [128, 8192]
        config = choose_config[a_type, b_type, c_type, transpose_b](m, N, K)
        if config not in set:
            set.add(config)

    return set^


@fieldwise_init
struct BlockScaledMatmulConfig[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool = True,
](Copyable, Equatable, Hashable, Stringable, TrivialRegisterPassable, Writable):
    """Static configuration of GPU matmul."""

    # Mandatory parameters
    var cta_group: Int
    var mma_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var AB_swapped: Bool
    var block_swizzle_size: Int
    var raster_order: RasterOrder

    comptime accum_type = get_accum_type[Self.a_type]()  # TODO: factor b_type

    comptime sf_block_atom_size = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K

    # Has default values or derivible from mandatory parameters
    var block_tile_shape: IndexList[3]
    var num_split_k: Int
    var num_pipeline_stages: Int
    var num_clc_pipeline_stages: Int
    var num_accum_pipeline_stages: Int
    var num_output_stages: Int
    var output_tile_shape: IndexList[2]
    var a_swizzle: TensorMapSwizzle
    var b_swizzle: TensorMapSwizzle
    var c_swizzle: TensorMapSwizzle
    var k_group_size: Int
    var scaling_kind: UMMAKind
    var vec_sf_size: Int
    var num_sf_k_tiles: Int

    fn __init__(
        out self,
        *,
        scaling_kind: UMMAKind,
        cta_group: Int = 2,
        mma_shape: IndexList[3] = get_mma_shape[Self.a_type, Self.accum_type](),
        cluster_shape: IndexList[3] = Index(2, 1, 1),
        AB_swapped: Bool = False,
        num_split_k: Int = 1,
        block_swizzle_size: Int = 0,
        raster_order: RasterOrder = RasterOrder.AlongM,
        k_group_size: Int = 1,
        num_pipeline_stages: Optional[Int] = None,
        num_accum_pipeline_stages: Int = 2,
        num_clc_pipeline_stages: Int = 2,
    ):
        constrained[Self.a_type == Self.b_type]()

        self.cta_group = cta_group
        self.mma_shape = mma_shape
        self.cluster_shape = cluster_shape
        self.AB_swapped = AB_swapped
        self.block_swizzle_size = block_swizzle_size
        self.raster_order = raster_order
        self.k_group_size = k_group_size

        self.block_tile_shape = _compute_block_tile_shape[Self.a_type](
            mma_shape, cta_group
        )
        self.output_tile_shape = _compute_output_tile_shape(
            mma_shape, cta_group, AB_swapped
        )

        # Scaling factors configuration (SFA, SFB)
        self.scaling_kind = scaling_kind
        self.vec_sf_size = (
            NVFP4_SF_VECTOR_SIZE if self.scaling_kind
            == UMMAKind.KIND_MXF4NVF4 else MXFP8_SF_VECTOR_SIZE
        )
        var sf_k_group_size = self.vec_sf_size * SF_ATOM_K
        self.num_sf_k_tiles = (
            (2 * self.block_tile_shape[2])
            // sf_k_group_size if self.scaling_kind
            == UMMAKind.KIND_MXF4NVF4 else self.block_tile_shape[2]
            // sf_k_group_size
        )

        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.num_output_stages = 2
        self.num_split_k = num_split_k

        var swizzles = _compute_swizzle_modes(
            self.output_tile_shape, AB_swapped
        )
        self.a_swizzle = swizzles[0]
        self.b_swizzle = swizzles[1]
        self.c_swizzle = swizzles[2]

        # Calculate scaling factor shared memory per stage
        var a_scales_smem_bytes_per_stage = (
            self.num_sf_k_tiles
            * (self.block_tile_shape[0] // SF_MN_GROUP_SIZE)
            * Self.sf_block_atom_size
            * size_of[Self.sfa_dtype]()
        )
        var b_scales_smem_bytes_per_stage = (
            self.num_sf_k_tiles
            * (
                align_up(self.mma_shape[1], SF_MN_GROUP_SIZE)
                // SF_MN_GROUP_SIZE
            )
            * Self.sf_block_atom_size
            * size_of[Self.sfb_dtype]()
        )
        var sf_smem_per_stage = (
            a_scales_smem_bytes_per_stage + b_scales_smem_bytes_per_stage
        )

        self.num_pipeline_stages = _maximize_pipeline_stages[
            Self.a_type, Self.b_type, Self.c_type
        ](
            self.block_tile_shape,
            self.output_tile_shape,
            self.num_output_stages,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            sf_smem_per_stage,
        )

        if num_pipeline_stages:
            self.num_pipeline_stages = num_pipeline_stages.value()

        # SM100 kernel only supports k grouping when num_pipeline_stages is a multiple of k_group_size.
        self.num_pipeline_stages = align_down(
            self.num_pipeline_stages, self.k_group_size
        )

    fn swap_AB_type(
        self,
    ) -> BlockScaledMatmulConfig[
        Self.b_type,
        Self.a_type,
        Self.c_type,
        Self.sfb_dtype,
        Self.sfa_dtype,
        Self.transpose_b,
    ]:
        return BlockScaledMatmulConfig[
            Self.b_type,
            Self.a_type,
            Self.c_type,
            Self.sfb_dtype,
            Self.sfa_dtype,
            Self.transpose_b,
        ](
            cta_group=self.cta_group,
            mma_shape=self.mma_shape,
            cluster_shape=self.cluster_shape,
            AB_swapped=self.AB_swapped,
            num_pipeline_stages=self.num_pipeline_stages,
            num_accum_pipeline_stages=self.num_accum_pipeline_stages,
            num_clc_pipeline_stages=self.num_clc_pipeline_stages,
            block_swizzle_size=self.block_swizzle_size,
            raster_order=self.raster_order,
            k_group_size=self.k_group_size,
            num_split_k=self.num_split_k,
            scaling_kind=self.scaling_kind,
        )

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("kernel_")
        writer.write(self.scaling_kind, "_")
        writer.write("A_vec", self.vec_sf_size, "_")
        writer.write(Self.sfa_dtype, "_")
        writer.write("B_vec", self.vec_sf_size, "_")
        writer.write(Self.sfb_dtype, "_")
        _write_common_config[W, Self.a_type, Self.c_type, Self.transpose_b](
            writer,
            self.cta_group,
            self.mma_shape,
            self.cluster_shape,
            self.num_pipeline_stages,
            self.k_group_size,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            self.num_output_stages,
            self.output_tile_shape,
            self.AB_swapped,
            self.a_swizzle,
            self.b_swizzle,
            self.c_swizzle,
            self.block_swizzle_size,
            self.raster_order,
            self.num_split_k,
        )

    fn __repr__(self) -> String:
        return String.write(self)


fn choose_block_scaled_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool = True,
](M: Int, N: Int, K: Int) -> BlockScaledMatmulConfig[
    a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
]:
    comptime assert a_type == b_type, "a_type and b_type must be the same"
    comptime assert (
        sfa_dtype == sfb_dtype
    ), "sfa_dtype and sfb_dtype must be the same"

    comptime num_SMs = B200.sm_count
    # Nvidia mma instruction process 32B in K.
    comptime Kbytes_per_mma = 32
    # We use 128B swizzle, tile size in K is 128B over element size.
    comptime BK = 128 // size_of[a_type]()

    comptime M_pivote = 32

    var cta_group = 1 if M < M_pivote else 2
    var swapAB = True if M < M_pivote else False
    var k_group_size = 1  # maybe increased for small M later

    var mma_mn = Tuple[Int, Int](256, 256)
    var min_num_waves = Int.MAX

    # Traverse possible combinations of BM x MMA_N to choose the one minimizes the
    # workload per SM. The computation per SM is the flops (ignoring 2x in 2MNK)
    # timed by max number of ctas per SM i.e. number of waves.
    # We first minimize the number of waves, then use the flops to break tie.

    # For small M, swap A and B so that the small M maps to mma_n since it supports
    # a larger range than mma_m.
    if M < M_pivote:
        for bm, mma_n in product([128], range(64, align_up(M, 64) + 1, 64)):
            num_ctas = ceildiv(M, mma_n) * ceildiv(N, bm)
            num_waves = ceildiv(num_ctas, num_SMs)
            if num_waves < min_num_waves or (
                num_waves == min_num_waves
                and bm * mma_n < mma_mn[0] * mma_mn[1]
            ):
                min_num_waves = num_waves
                mma_mn[0] = bm
                mma_mn[1] = mma_n

    # For large M, use 2xSM mma
    else:

        @parameter
        @always_inline
        fn select_mma_mn(M: Int, N: Int, _swapAB: Bool = False):
            N_alignby64 = align_up(N, 64)
            max_mma_n = min(N_alignby64, 256)
            # In pratice 64x16 mma creates too many ctas and increase L2
            # load volume, ends up hurting performance.
            min_mma_n = min(N_alignby64, 64)
            for bm in [128]:
                for mma_n in range(max_mma_n, min_mma_n - 1, -64):
                    var mma_m = bm * cta_group
                    var num_clusters = ceildiv(M, mma_m) * ceildiv(N, mma_n)
                    var num_waves = ceildiv(num_clusters, num_SMs // cta_group)
                    if num_waves > min_num_waves:
                        break
                    elif num_waves < min_num_waves or (
                        num_waves == min_num_waves
                        and mma_m * mma_n < mma_mn[0] * mma_mn[1]
                    ):
                        min_num_waves = num_waves
                        mma_mn[0] = mma_m
                        mma_mn[1] = mma_n
                        swapAB = _swapAB

        # Swap AB may work better for M = 192 and not-multiple-of-128 values.
        # Capture and update min_num_waves, mma_mn
        select_mma_mn(M, N)
        select_mma_mn(N, M, True)

    # For small mmas, we group multiple tiles per tma-mma synchronization.
    var output_block_size = (mma_mn[0] // cta_group) * mma_mn[1]
    if output_block_size <= 64 * 96 and ceildiv(K, BK) % 2 == 0:
        k_group_size = 2
    # For very small mmas we can group more aggressively.
    if output_block_size <= 64 * 16 and ceildiv(K, BK) % 4 == 0:
        k_group_size = 4

    var min_load_volume = Int.MAX
    var optimal_block_swizzle_size = 0

    # Tile waves when there are >= 4 waves. In theory it should be >=2, but let's
    # be conservative.
    if min_num_waves >= 4:
        # Represent the load volume by
        #    BM * num_ctas_per_wave_m + MMA_N * num_ctas_per_wave_N
        # Use MMA_N because cta_group = 2, 2 ctas cover entire MMA_N. cta_group = 1
        # has BN = MMA_N.
        # Traverse the tile sizes to find min load volume per wave.
        # TODO: consider the L2 resue across waves.
        var BM = mma_mn[0] // cta_group
        for tile_size in [1, 2, 4, 8]:
            var num_ctas_m = ceildiv(M, BM)
            # When tile_size is small, it's possible that a wave has more ctas
            # then num_ctas_m * tile_size and num_ctas_per_wave_m > num_ctas_m.
            # The ctas mapping will "wrap around" and include following tile_sizes.
            var num_ctas_per_wave_m = ceildiv(num_SMs, tile_size)
            var num_ctas_per_wave_n = tile_size * ceildiv(
                num_ctas_per_wave_m, num_ctas_m
            )
            num_ctas_per_wave_m = min(num_ctas_per_wave_m, num_ctas_m)
            var load_volume_per_wave = (
                num_ctas_per_wave_m * BM + num_ctas_per_wave_n * mma_mn[1]
            )
            if load_volume_per_wave < min_load_volume:
                min_load_volume = load_volume_per_wave
                optimal_block_swizzle_size = tile_size

    # TODO: evaluate the comment's perf impact
    # var num_clc_pipeline_stages: UInt = UInt(min(min_num_waves-1, 2))
    var num_clc_pipeline_stages = 0 if min_num_waves == 1 else 2

    var num_accum_pipeline_stages = 2 if mma_mn[1] <= 128 else 1

    var scaling_kind = (
        UMMAKind.KIND_MXF4NVF4 if a_type
        == DType.uint8 else UMMAKind.KIND_MXF8F6F4
    )

    return BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ](
        scaling_kind=scaling_kind,
        mma_shape=IndexList[3](
            mma_mn[0], mma_mn[1], Kbytes_per_mma // size_of[a_type]()
        ),
        cta_group=cta_group,
        cluster_shape=Index(cta_group, 1, 1),
        AB_swapped=swapAB,
        block_swizzle_size=optimal_block_swizzle_size,
        num_accum_pipeline_stages=num_accum_pipeline_stages,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        k_group_size=k_group_size,
    )


fn build_block_scaled_configs[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
]() -> Set[
    BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]
]:
    comptime config_t = BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]

    var set = Set[config_t]()

    for m in range(8, 128, 8):  # [8, 128]
        config = choose_block_scaled_config[
            a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
        ](m, N, K)
        if config not in set:
            set.add(config)

    for m in range(128, 8193, 64):  # [128, 8192]
        config = choose_block_scaled_config[
            a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
        ](m, N, K)
        if config not in set:
            set.add(config)

    return set^
