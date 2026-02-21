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

"""Register-based accumulator for blockwise FP8 matmul.

Unlike standard SM100 matmul which accumulates directly in TMEM, blockwise FP8
requires per-K-iteration scaling in CUDA cores:

    for k in K_iterations:
        partial = TMEM load (MMA result)
        scaled = partial * a_scale * b_scale
        accum += scaled  # in registers
    result = accum  # write to SMEM → GMEM
"""

from math import gcd

from gpu import WARP_SIZE, lane_id
from gpu import warp_id as get_warp_id
from gpu.memory import AddressSpace
from gpu.primitives.cluster import block_rank_in_cluster
from gpu.sync import syncwarp
from layout._coord import Coord, Idx
from layout._layout import TensorLayout, row_major
from layout._tile_tensor import TileTensor, stack_allocation
from utils.index import IndexList
from utils.static_tuple import StaticTuple

from ..structured_kernels.tile_types import (
    SMemTileArray2DRowMajor,
    static_row_major,
)
from ..structured_kernels.pipeline import ProducerConsumerPipeline
from ..structured_kernels.tile_pipeline import OutputStage, EpilogueKStage
from ..structured_kernels.tmem import TmemAddress, TmemFragments


# =============================================================================
# Accumulator size calculation
# =============================================================================


@always_inline
fn get_accumulator_dims[
    *,
    c_smem_dim1: Int,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
]() -> IndexList[2]:
    """Compute register accumulator dimensions for blockwise FP8.

    Returns (num_stages, num_elements) for the register tile shape.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    constrained[num_m_mmas == 1 and num_n_mmas == 1]()

    comptime stageN = c_smem_dim1
    comptime cg2_num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    comptime cg1_num_stages = MMA_N // stageN
    comptime num_stages = cg2_num_stages if cta_group == 2 else cg1_num_stages
    comptime data_paths = 16
    comptime bits = 256
    comptime repeats = stageN // (bits // 32)

    comptime num_elements_per_load = bits // 32
    comptime fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    comptime num_elements = repeats * fragment_size

    return IndexList[2](num_stages, num_elements)


@always_inline
fn is_lower_fragment_required[
    cta_group: Int,
    block_tile_shape: IndexList[3],
]() -> Bool:
    """Determine if lower TMEM fragment is needed based on config."""
    return not (cta_group == 1 and block_tile_shape[0] == 64)


# =============================================================================
# BlockwiseFP8Accumulator - Register-based accumulator management
# =============================================================================


struct BlockwiseFP8Accumulator[
    accum_type: DType,
    accum_num_stages: Int,
    accum_num_elements: Int,
    is_lower_required: Bool,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_size: Int,
]:
    """Register-based accumulator for blockwise FP8 matmul.

    Manages upper and lower fragment tiles in registers for per-K accumulation.
    Unlike TMEM-based accumulation, this allows scaling in CUDA cores.

    Parameters:
        accum_type: Accumulator data type (typically float32).
        accum_num_stages: Number of accumulator pipeline stages.
        accum_num_elements: Number of elements per stage.
        is_lower_required: Whether lower fragment is needed.
        block_tile_shape: Block tile dimensions (BM, BN, BK).
        mma_shape: MMA operation dimensions (MMA_M, MMA_N, MMA_K).
        cluster_size: Number of CTAs in the cluster.
    """

    # Derived tile dimensions
    comptime BM = Self.block_tile_shape[0]
    comptime BN = Self.block_tile_shape[1]
    comptime BK = Self.block_tile_shape[2]
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]

    comptime num_stages = Self.accum_num_stages
    comptime num_elements = Self.accum_num_elements

    # TileTensor register tile layout: row-major [num_stages, num_elements]
    comptime AccumLayout = static_row_major[Self.num_stages, Self.num_elements]
    comptime RegTileType = TileTensor[
        Self.accum_type,
        Self.AccumLayout,
        MutExternalOrigin,
        address_space = AddressSpace.GENERIC,
    ]

    # Fragment load parameters (match TmemFragments defaults)
    comptime data_paths = 16
    comptime bits = 256
    comptime num_elements_per_load = Self.bits // 32
    comptime fragment_size = (
        Self.data_paths * Self.num_elements_per_load
    ) // WARP_SIZE
    comptime repeats = Self.num_elements // Self.fragment_size
    comptime stageN = Self.repeats * (Self.bits // 32)
    comptime rep_frag_size = Self.repeats * Self.fragment_size

    # TmemFragments type for encapsulated upper/lower loading
    comptime Fragments = TmemFragments[
        Self.accum_type,
        Self.fragment_size,
        is_lower_required = Self.is_lower_required,
        data_paths = Self.data_paths,
        bits = Self.bits,
    ]

    var upper: Self.RegTileType
    var lower: Self.RegTileType

    @always_inline
    fn __init__(out self):
        """Create accumulator with zero-initialized register tiles."""
        var accum_layout = row_major[Self.num_stages, Self.num_elements]()
        self.upper = stack_allocation[dtype = Self.accum_type](accum_layout)
        self.lower = stack_allocation[dtype = Self.accum_type](accum_layout)
        _ = self.upper.fill(0.0)

        comptime if Self.is_lower_required:
            _ = self.lower.fill(0.0)

    @always_inline
    fn promote[
        # Parameters derived from argument types (use _ for inference)
        num_pipeline_stages: Int,
        num_accum_pipeline_stages: Int,
        stage_stride_cols: Int,
        cta_group: Int,
        num_input_stages: Int,
        # Type parameters
        b_scales_dtype: DType,
        a_scales_dtype: DType,
        # A-scales tile dimensions
        a_scales_dim0: Int,
        a_scales_dim1: Int,
    ](
        mut self,
        b_scales: TileTensor[b_scales_dtype, _, ImmutAnyOrigin],
        a_scales_tiles: SMemTileArray2DRowMajor[
            a_scales_dtype,
            a_scales_dim0,
            a_scales_dim1,
            num_pipeline_stages,
        ],
        epi_stage: EpilogueKStage[
            num_accum_pipeline_stages,
            stage_stride_cols,
            cta_group,
            num_input_stages,
        ],
        work_tile_coord: Tuple[UInt, UInt],
        k_iter: UInt,
        problem_shape: StaticTuple[Int32, 3],
    ):
        """Load partial from TMEM, apply scales, accumulate into registers.

        Core blockwise FP8 scaling: loads MMA partial from TMEM, reads A-scale
        from SMEM and B-scale from global memory, applies scaling, and
        accumulates into register tiles.

        Called within `with epi_ctx.per_k_stage(input_pipeline) as epi_stage:`.
        """
        # Type aliases for readability
        comptime a_scales_type = a_scales_dtype

        constrained[
            a_scales_dtype == b_scales_dtype
            and Self.accum_type == DType.float32,
            "Only support float32 for a_scales, b_scales, and accum_type",
        ]()

        var M = problem_shape[0]
        var N = problem_shape[1]
        var K = problem_shape[2]

        comptime load_width = 2

        var bm = work_tile_coord[0]
        var bn = work_tile_coord[1]

        # B-scale index calculation when MMA_N != BK(128)
        var b_scale_idx0 = 0
        var b_scale_next_n = 0
        var b_scale_0: Scalar[Self.accum_type]
        var b_scale_1: Scalar[Self.accum_type]

        comptime if Self.MMA_N != Self.BK:
            constrained[
                Self.stageN <= gcd(Self.MMA_N, Self.BK)
                and (gcd(Self.MMA_N, Self.BK) % Self.stageN == 0),
                "gcd(MMA_N, BK) must be divisible by stageN",
            ]()

            var global_bn_start = bn * UInt(Self.MMA_N)
            var begin_n = min(
                Int32(Self.BK) - Int32(global_bn_start % UInt(Self.BK)),
                Int32(Self.MMA_N),
            )
            var end_n = min(N - Int32(global_bn_start), Int32(Self.MMA_N))

            b_scale_idx0 = Int(global_bn_start // UInt(Self.BK))
            b_scale_next_n = Int(begin_n) if begin_n < end_n else Self.MMA_N

            b_scale_0 = rebind[Scalar[Self.accum_type]](
                b_scales.ptr.load(
                    b_scales.layout(Coord(Idx(b_scale_idx0), Idx(k_iter)))
                ).cast[Self.accum_type]()
            )
            if b_scale_next_n < Self.MMA_N:
                b_scale_1 = rebind[Scalar[Self.accum_type]](
                    b_scales.ptr.load(
                        b_scales.layout(
                            Coord(Idx(b_scale_idx0 + 1), Idx(k_iter))
                        )
                    ).cast[Self.accum_type]()
                )
            else:
                b_scale_1 = 0.0
        else:
            b_scale_0 = rebind[Scalar[Self.accum_type]](
                b_scales.ptr.load(
                    b_scales.layout(Coord(Idx(bn), Idx(k_iter)))
                ).cast[Self.accum_type]()
            )
            b_scale_1 = 0.0

        var warp_id = get_warp_id()

        # Compute row/col offset based on MMA layout
        var staged_c_row: UInt
        var staged_c_col: UInt

        comptime if Self.MMA_M == 256 or (Self.MMA_M == 128 and cta_group == 1):
            staged_c_row = warp_id * UInt(WARP_SIZE)
            staged_c_col = UInt(0)
        elif Self.MMA_M == 64 and cta_group == 1:
            staged_c_row = warp_id * UInt(WARP_SIZE // 2)
            staged_c_col = UInt(0)
        else:
            staged_c_row = (warp_id % 2) * UInt(WARP_SIZE)
            staged_c_col = UInt(Self.BN) * (warp_id // 2)

        # Thread coordinate within fragment
        comptime threads_per_row = UInt(
            Self.stageN // Self.repeats // load_width
        )
        var top_frag_upper_coord = StaticTuple[UInt32, 2](
            UInt32(lane_id() // threads_per_row),
            UInt32(lane_id() % threads_per_row * load_width),
        )
        var bottom_frag_upper_coord = StaticTuple[UInt32, 2](
            top_frag_upper_coord[0] + 8, top_frag_upper_coord[1]
        )
        var top_frag_lower_coord = StaticTuple[UInt32, 2](
            top_frag_upper_coord[0] + 16, top_frag_upper_coord[1]
        )
        var bottom_frag_lower_coord = StaticTuple[UInt32, 2](
            top_frag_lower_coord[0] + 8, top_frag_lower_coord[1]
        )

        var tmem_offset = UInt32(epi_stage.output_stage.tmem.offset())
        var tma_load_stage_index = epi_stage.input_stage_index
        var a_scales_smem = a_scales_tiles[tma_load_stage_index]

        var upper_sfa0_smem = a_scales_smem[
            0, UInt32(staged_c_row) + top_frag_upper_coord[0]
        ].cast[Self.accum_type]()
        var upper_sfa1_smem = a_scales_smem[
            0, UInt32(staged_c_row) + bottom_frag_upper_coord[0]
        ].cast[Self.accum_type]()

        var lower_sfa0_smem = Scalar[Self.accum_type]()
        var lower_sfa1_smem = Scalar[Self.accum_type]()

        comptime if Self.is_lower_required:
            lower_sfa0_smem = rebind[Scalar[Self.accum_type]](
                a_scales_smem[
                    0, UInt32(staged_c_row) + top_frag_lower_coord[0]
                ].cast[Self.accum_type]()
            )
            lower_sfa1_smem = rebind[Scalar[Self.accum_type]](
                a_scales_smem[
                    0, UInt32(staged_c_row) + bottom_frag_lower_coord[0]
                ].cast[Self.accum_type]()
            )

        # Signal input pipeline before TMEM loop
        syncwarp()
        if lane_id() < UInt(Self.cluster_size):
            epi_stage.arrive_input()
        syncwarp()

        comptime for stage in range(Self.num_stages):
            var tmem_addr = TmemAddress(
                tmem_offset + UInt32(stage * Self.stageN)
            )
            var frags = Self.Fragments.load[repeat = Self.repeats](tmem_addr)
            Self.Fragments.wait_load()

            var b_scale: Scalar[Self.accum_type]

            comptime if Self.MMA_N != Self.BK:
                b_scale = (
                    b_scale_0 if (stage * Self.stageN + Int(staged_c_col))
                    < b_scale_next_n else b_scale_1
                )
            else:
                b_scale = b_scale_0

            comptime for ld_iter in range(Self.repeats):
                comptime for j in range(Self.fragment_size // 2):
                    comptime offset = ld_iter * Self.fragment_size + j * 2

                    var upper_elems = frags.upper.slice[2, offset=offset]()
                    var lower_elems = frags.lower.slice[2, offset=offset]()

                    var upper_a_scale = (
                        upper_sfa0_smem if j == 0 else upper_sfa1_smem
                    )
                    var lower_a_scale = (
                        lower_sfa0_smem if j == 0 else lower_sfa1_smem
                    )

                    var upper_scale = upper_a_scale * b_scale
                    var lower_scale = lower_a_scale * b_scale

                    self.upper[stage, offset] += rebind[
                        Scalar[Self.accum_type]
                    ](upper_elems[0]) * rebind[Scalar[Self.accum_type]](
                        upper_scale
                    )
                    self.upper[stage, offset + 1] += rebind[
                        Scalar[Self.accum_type]
                    ](upper_elems[1]) * rebind[Scalar[Self.accum_type]](
                        upper_scale
                    )

                    comptime if Self.is_lower_required:
                        self.lower[stage, offset] += rebind[
                            Scalar[Self.accum_type]
                        ](lower_elems[0]) * rebind[Scalar[Self.accum_type]](
                            lower_scale
                        )
                        self.lower[stage, offset + 1] += rebind[
                            Scalar[Self.accum_type]
                        ](lower_elems[1]) * rebind[Scalar[Self.accum_type]](
                            lower_scale
                        )
