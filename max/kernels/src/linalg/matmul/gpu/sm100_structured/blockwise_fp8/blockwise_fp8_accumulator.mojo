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

from std.collections import Array
from std.math import ceildiv, gcd
from std.math.uutils import umod, ufloordiv

from max.gpu import WARP_SIZE, lane_id, warp_id as get_warp_id
from max.gpu.sync import syncwarp
import max.gpu.primitives.warp as warp
from layout import (
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
    stack_allocation,
)
from std.utils.static_tuple import StaticTuple

from structured_kernels.tile_types import (
    SMemTileArray2DRowMajor,
    static_row_major,
)
from ..structured_kernels.config import OutputPipelineConfig
from ..structured_kernels.tile_pipeline import EpilogueKStage
from ..structured_kernels.tmem import TmemAddress, TmemFragments


# =============================================================================
# Accumulator size calculation
# =============================================================================


@inline(.always)
def get_accumulator_dims[
    *,
    c_smem_dim1: Int,
    block_tile_shape: Tuple[Int, Int, Int],
    mma_shape: Tuple[Int, Int, Int],
    cta_group: Int,
]() -> Tuple[Int, Int]:
    """Compute register accumulator dimensions for blockwise FP8.

    Parameters:
        c_smem_dim1: N-direction width of each SMEM accumulator stage in
            elements.
        block_tile_shape: Block tile dimensions as `(BM, BN, BK)`.
        mma_shape: MMA operation dimensions as `(MMA_M, MMA_N, MMA_K)`.
        cta_group: Number of CTAs cooperating per MMA (1 or 2).

    Returns:
        `(num_stages, num_elements)` describing the register tile shape.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

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

    return (num_stages, num_elements)


@inline(.always)
def is_lower_fragment_required[
    cta_group: Int,
    block_tile_shape: Tuple[Int, Int, Int],
]() -> Bool:
    """Determine if lower TMEM fragment is needed based on config.

    Parameters:
        cta_group: Number of CTAs cooperating per MMA (1 or 2).
        block_tile_shape: Block tile dimensions as `(BM, BN, BK)`; only
            `BM` (index 0) is consulted.
    """
    return not (cta_group == 1 and block_tile_shape[0] == 64)


# =============================================================================
# BlockwiseFP8Accumulator - Register-based accumulator management
# =============================================================================


struct BlockwiseFP8Accumulator[
    accum_type: DType,
    accum_num_stages: Int,
    accum_num_elements: Int,
    is_lower_required: Bool,
    block_tile_shape: Tuple[Int, Int, Int],
    mma_shape: Tuple[Int, Int, Int],
    cluster_size: Int,
    n_scale_granularity: Int = 128,
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
        n_scale_granularity: B-scale N-direction block size (defaults to
            BK; set smaller when the matmul's N-tile spans multiple
            finer-grained scale blocks).
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
        MutUntrackedOrigin,
        # Register-backed stack tile, never DeviceBuffer-backed, so pin
        # DefaultEngine.
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
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
        is_lower_required=Self.is_lower_required,
        data_paths=Self.data_paths,
        bits=Self.bits,
    ]

    var upper: Self.RegTileType
    var lower: Self.RegTileType

    @inline(.always)
    def __init__(out self):
        """Create accumulator with zero-initialized register tiles."""
        var accum_layout = row_major[Self.num_stages, Self.num_elements]()
        self.upper = stack_allocation[dtype=Self.accum_type](accum_layout)
        self.lower = stack_allocation[dtype=Self.accum_type](accum_layout)
        _ = self.upper.fill(0.0)

        comptime if Self.is_lower_required:
            _ = self.lower.fill(0.0)

    @inline(.always)
    def promote[
        # Parameters derived from argument types (use _ for inference)
        num_pipeline_stages: Int,
        opc: OutputPipelineConfig,
        num_input_stages: Int,
        # Type parameters
        b_scales_dtype: DType,
        b_scales_layout: TensorLayout,
        b_scales_engine: TensorEngine,
        a_scales_dtype: DType,
        # A-scales tile dimensions
        a_scales_dim0: Int,
        a_scales_dim1: Int,
    ](
        mut self,
        b_scales: TileTensor[
            b_scales_dtype,
            b_scales_layout,
            ImmutAnyOrigin,
            Engine=b_scales_engine,
        ],
        a_scales_tiles: SMemTileArray2DRowMajor[
            a_scales_dtype,
            a_scales_dim0,
            a_scales_dim1,
            num_pipeline_stages,
        ],
        epi_stage: EpilogueKStage[
            opc,
            num_input_stages,
        ],
        work_tile_coord: Tuple[Int, Int],
        k_iter: Int,
        problem_shape: StaticTuple[Int32, 3],
    ):
        """Load partial from TMEM, apply scales, accumulate into registers.

        Core blockwise FP8 scaling: loads MMA partial from TMEM, reads A-scale
        from SMEM and B-scale from global memory, applies scaling, and
        accumulates into register tiles.

        Called within `with epi_ctx.per_k_stage(input_pipeline) as epi_stage:`.

        Parameters:
            num_pipeline_stages: Depth of the A-scales SMEM tile pipeline.
            opc: Output pipeline configuration; supplies `cta_group`.
            num_input_stages: Number of input stages in the epilogue K-stage.
            b_scales_dtype: Element type of the B-scales tensor; must be
                `float32`.
            b_scales_layout: Memory layout of the B-scales tensor.
            b_scales_engine: Engine of the B-scales tensor.
            a_scales_dtype: Element type of the A-scales SMEM tiles; must
                be `float32`.
            a_scales_dim0: Row count of each A-scales SMEM tile.
            a_scales_dim1: Column count of each A-scales SMEM tile.

        Args:
            b_scales: B-scale factors loaded from global memory, indexed
                by N-block and K-iteration.
            a_scales_tiles: Pipeline of A-scale factor tiles in SMEM,
                indexed by input stage.
            epi_stage: Epilogue K-stage handle providing the TMEM offset
                and input-pipeline signaling.
            work_tile_coord: `(bm, bn)` coordinates of this tile in the
                M-N problem grid.
            k_iter: Current K-direction iteration index.
            problem_shape: `(M, N, K)` dimensions of the full matmul
                problem.
        """
        # Type aliases for readability
        comptime a_scales_type = a_scales_dtype

        comptime assert (
            a_scales_dtype == b_scales_dtype and Self.accum_type == .float32
        ), "Only support float32 for a_scales, b_scales, and accum_type"

        var M = problem_shape[0]
        var N = problem_shape[1]
        var K = problem_shape[2]

        comptime load_width = 2

        var bm = work_tile_coord[0]
        var bn = work_tile_coord[1]

        # B-scale N-direction block size is independent of the kernel's
        # K-tile size; defaults to BK but can be smaller when the matmul
        # spans multiple finer-grained scale blocks per MMA tile.
        comptime NScaleBlock = Self.n_scale_granularity

        # An MMA tile starts at a multiple of gcd(MMA_N, NScaleBlock), so it
        # can touch up to this many scale blocks. This is not limited to two:
        # a 256-wide tile over 64-wide scales spans four, which is why the
        # dispatcher may pick MMA_N independently of n_scale_granularity. A
        # nonzero in-block offset (e.g. MMA_N=192, NScaleBlock=128) is only
        # covered by the tests that pass such a config explicitly.
        comptime num_b_scales = ceildiv(
            Self.MMA_N + NScaleBlock - gcd(Self.MMA_N, NScaleBlock),
            NScaleBlock,
        )
        # Scale i applies from tile column b_scale_start[i] on; MMA_N marks a
        # block that starts past the end of N.
        var b_scale_vals = Array[Scalar[Self.accum_type], num_b_scales](
            fill=0.0
        )
        var b_scale_start = Array[Int, num_b_scales](fill=Self.MMA_N)
        var b_scale_idx0 = bn

        comptime if Self.MMA_N != NScaleBlock:
            comptime assert Self.stageN <= gcd(Self.MMA_N, NScaleBlock) and (
                gcd(Self.MMA_N, NScaleBlock) % Self.stageN == 0
            ), "gcd(MMA_N, n_scale_granularity) must be divisible by stageN"

            var global_bn_start = bn * Self.MMA_N
            var end_n = min(Int(N) - global_bn_start, Self.MMA_N)
            b_scale_idx0 = Int(ufloordiv(global_bn_start, NScaleBlock))
            var offset_in_block = Int(umod(global_bn_start, NScaleBlock))

            comptime for i in range(1, num_b_scales):
                var start = i * NScaleBlock - offset_in_block
                if start < end_n:
                    b_scale_start[i] = start
                    b_scale_vals[i] = rebind[Scalar[Self.accum_type]](
                        b_scales.load((b_scale_idx0 + i, k_iter)).cast[
                            Self.accum_type
                        ]()
                    )

        b_scale_vals[0] = rebind[Scalar[Self.accum_type]](
            b_scales.load((b_scale_idx0, k_iter)).cast[Self.accum_type]()
        )

        var warp_id = get_warp_id()

        # Compute row/col offset based on MMA layout
        var staged_c_row: UInt32
        var staged_c_col: Int

        comptime if Self.MMA_M == 256 or (
            Self.MMA_M == 128 and opc.cta_group == 1
        ):
            staged_c_row = UInt32(warp_id * WARP_SIZE)
            staged_c_col = 0
        elif Self.MMA_M == 64 and opc.cta_group == 1:
            staged_c_row = UInt32(warp_id * (WARP_SIZE // 2))
            staged_c_col = 0
        else:
            staged_c_row = UInt32(umod(warp_id, 2) * WARP_SIZE)
            staged_c_col = Self.BN * ufloordiv(warp_id, 2)
            # The b_scale_start sentinel (MMA_N) relies on every column this
            # warp visits staying below MMA_N.
            comptime assert (
                Self.BN + Self.num_stages * Self.stageN <= Self.MMA_N
            ), "staged columns must stay inside the MMA tile"

        # Thread coordinate within fragment
        comptime threads_per_row = Self.stageN // Self.repeats // load_width
        var top_frag_upper_coord = StaticTuple[UInt32, 2](
            UInt32(ufloordiv(lane_id(), threads_per_row)),
            UInt32(umod(lane_id(), threads_per_row) * load_width),
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
            0, staged_c_row + top_frag_upper_coord[0]
        ].cast[Self.accum_type]()
        var upper_sfa1_smem = a_scales_smem[
            0, staged_c_row + bottom_frag_upper_coord[0]
        ].cast[Self.accum_type]()

        var lower_sfa0_smem = Scalar[Self.accum_type]()
        var lower_sfa1_smem = Scalar[Self.accum_type]()

        comptime if Self.is_lower_required:
            lower_sfa0_smem = a_scales_smem[
                0, staged_c_row + top_frag_lower_coord[0]
            ].cast[Self.accum_type]()
            lower_sfa1_smem = a_scales_smem[
                0, staged_c_row + bottom_frag_lower_coord[0]
            ].cast[Self.accum_type]()

        # Signal input pipeline before TMEM loop
        syncwarp()
        if lane_id() < Self.cluster_size:
            epi_stage.arrive_input()
        syncwarp()

        comptime for stage in range(Self.num_stages):
            var tmem_addr = TmemAddress(
                tmem_offset + UInt32(stage * Self.stageN)
            )
            var frags = Self.Fragments.load[repeat=Self.repeats](tmem_addr)
            Self.Fragments.wait_load()

            var b_scale = b_scale_vals[0]

            comptime for i in range(1, num_b_scales):
                if stage * Self.stageN + staged_c_col >= b_scale_start[i]:
                    b_scale = b_scale_vals[i]

            comptime for ld_iter in range(Self.repeats):
                comptime for j in range(Self.fragment_size // 2):
                    comptime offset = ld_iter * Self.fragment_size + j * 2

                    var upper_a_scale = (
                        upper_sfa0_smem if j == 0 else upper_sfa1_smem
                    )
                    var lower_a_scale = (
                        lower_sfa0_smem if j == 0 else lower_sfa1_smem
                    )

                    var upper_scale = upper_a_scale * b_scale
                    var lower_scale = lower_a_scale * b_scale

                    self.upper[stage, offset] += (
                        frags.upper[offset] * upper_scale
                    )
                    self.upper[stage, offset + 1] += (
                        frags.upper[offset + 1] * upper_scale
                    )

                    comptime if Self.is_lower_required:
                        self.lower[stage, offset] += (
                            frags.lower[offset] * lower_scale
                        )
                        self.lower[stage, offset + 1] += (
                            frags.lower[offset + 1] * lower_scale
                        )


# =============================================================================
# BlockwiseFP8BScales - a tile's B scales held across K blocks
# =============================================================================


struct BlockwiseFP8BScales[
    accum_type: DType,
    MMA_N: Int,
    n_scale_granularity: Int,
](TrivialRegisterPassable):
    """A work tile's B scales for 32 K blocks, one per lane of each warp.

    Loading each K block's B scale from global memory after the MMA-done
    wait puts a dependent memory round trip in every K block of the
    epilogue. This loads a chunk of 32 K blocks per lane up front (and the
    next chunk one chunk ahead) and hands each K block its scale by warp
    shuffle. The values are the same loads, so results are unchanged bit
    for bit. The tile must lie inside one B-scale block along N.

    Parameters:
        accum_type: Accumulator data type (float32).
        MMA_N: Rows of B covered by the tile (the swap-AB tile's MMA M).
        n_scale_granularity: B-scale N-direction block size.
    """

    comptime chunk = WARP_SIZE

    var idx: Int
    var num_iters: Int
    var cur: Scalar[Self.accum_type]
    var nxt: Scalar[Self.accum_type]

    @inline(.always)
    def __init__[
        b_scales_dtype: DType,
        b_scales_layout: TensorLayout,
        b_scales_engine: TensorEngine,
    ](
        out self,
        b_scales: TileTensor[
            b_scales_dtype,
            b_scales_layout,
            ImmutAnyOrigin,
            Engine=b_scales_engine,
        ],
        bn: Int,
        num_iters: Int,
    ):
        """Locate the tile's B-scale row and load the first two chunks.

        Args:
            b_scales: B-scale factors, indexed by (N block, K block).
            bn: Index of the tile along B's rows (N).
            num_iters: Number of K blocks.
        """
        comptime assert (
            Self.n_scale_granularity % Self.MMA_N == 0
        ), "the tile must lie inside one B-scale block"
        self.idx = bn * Self.MMA_N // Self.n_scale_granularity
        self.num_iters = num_iters
        self.cur = 0.0
        self.nxt = 0.0
        self._load(b_scales, 0)
        self.cur = self.nxt
        if num_iters > Self.chunk:
            self._load(b_scales, Self.chunk)

    @inline(.always)
    def _load[
        b_scales_dtype: DType,
        b_scales_layout: TensorLayout,
        b_scales_engine: TensorEngine,
    ](
        mut self,
        b_scales: TileTensor[
            b_scales_dtype,
            b_scales_layout,
            ImmutAnyOrigin,
            Engine=b_scales_engine,
        ],
        k_base: Int,
    ):
        var k = k_base + Int(lane_id())
        if k < self.num_iters:
            self.nxt = rebind[Scalar[Self.accum_type]](
                b_scales.load((self.idx, k)).cast[Self.accum_type]()
            )

    @inline(.always)
    def advance[
        b_scales_dtype: DType,
        b_scales_layout: TensorLayout,
        b_scales_engine: TensorEngine,
    ](
        mut self,
        b_scales: TileTensor[
            b_scales_dtype,
            b_scales_layout,
            ImmutAnyOrigin,
            Engine=b_scales_engine,
        ],
        k_iter: Int,
    ):
        """At a chunk boundary, rotate in the prefetched chunk and fetch the next.

        Args:
            b_scales: B-scale factors, indexed by (N block, K block).
            k_iter: Current K block.
        """
        if k_iter > 0 and umod(k_iter, Self.chunk) == 0:
            self.cur = self.nxt
            if k_iter + Self.chunk < self.num_iters:
                self._load(b_scales, k_iter + Self.chunk)

    @inline(.always)
    def get(self, k_iter: Int) -> Scalar[Self.accum_type]:
        """Return the B scale of K block `k_iter`.

        Args:
            k_iter: Current K block.
        """
        return warp.shuffle_idx(self.cur, UInt32(umod(k_iter, Self.chunk)))


# =============================================================================
# BlockwiseFP8SwapAccumulator - register accumulator for the swap-AB tile
# =============================================================================


struct BlockwiseFP8SwapAccumulator[
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    cluster_size: Int,
](TrivialRegisterPassable):
    """Register accumulator for a MMA_M x MMA_N swap-AB tile (C^T).

    The MMA's A operand is MMA_M weight rows (TMEM lanes; each epilogue warp
    owns MMA_M / 4 of them in 16-lane halves) and its B operand `MMA_N`
    activation rows (TMEM columns). Per K tile each element gets
    `partial * (a_scale[column] * b_scale)` added in K order, the same
    expression as the non-swapped accumulator, with the activation scale now
    per column and one weight-block scale per tile.

    Parameters:
        accum_type: Accumulator data type (float32).
        MMA_M: Weight rows per tile (UMMA M), 64 or 128.
        MMA_N: Activation rows per tile (UMMA N), 8 or 16.
        cluster_size: CTAs per cluster (arrivals on the input stage).
    """

    comptime reps = Self.MMA_N // 8
    # 16-lane halves per warp: M=64 puts 16 rows in each warp's lanes, M=128
    # puts 32.
    comptime halves = Self.MMA_M // 64
    # Per 8-column repeat a thread holds rows (r, r + 8) x columns (c, c + 1).
    comptime per_rep = 4
    comptime half_size = Self.reps * Self.per_rep
    comptime size = Self.halves * Self.half_size

    var acc: SIMD[Self.accum_type, Self.size]

    @inline(.always)
    def __init__(out self):
        self.acc = SIMD[Self.accum_type, Self.size](0.0)

    @inline(.always)
    def promote[
        group: Int,
        num_pipeline_stages: Int,
        opc: OutputPipelineConfig,
        num_input_stages: Int,
        a_scales_dtype: DType,
        a_scales_dim0: Int,
        a_scales_dim1: Int,
    ](
        mut self,
        b_scale: SIMD[Self.accum_type, group],
        a_scales_tiles: SMemTileArray2DRowMajor[
            a_scales_dtype,
            a_scales_dim0,
            a_scales_dim1,
            num_pipeline_stages,
        ],
        epi_stage: EpilogueKStage[
            opc,
            num_input_stages,
        ],
    ):
        """Scale and add a group of K-tile partials, in K order.

        Parameters:
            group: K tiles per pipeline stage.
            num_pipeline_stages: Number of A-scale SMEM tiles.
            opc: Output pipeline configuration.
            num_input_stages: Number of input pipeline stages.
            a_scales_dtype: A-scales element type.
            a_scales_dim0: A-scales tile rows.
            a_scales_dim1: A-scales tile columns.

        Args:
            b_scale: Per K tile, the tile's weight-block scale.
            a_scales_tiles: Pipeline of activation-scale tiles in SMEM.
            epi_stage: Epilogue K-stage handle.
        """
        comptime assert (
            a_scales_dtype == Self.accum_type and Self.accum_type == .float32
        ), "Only support float32 scales and accumulators"
        var col0 = UInt32(umod(lane_id(), 4)) * 2
        var first_tile = epi_stage.input_stage_index * UInt32(group)

        # Activation scale per (K tile, repeat, column), read before the
        # input stage is released.
        var a_sc = SIMD[Self.accum_type, group * Self.reps * 2]()
        comptime for g in range(group):
            var a_scales_smem = a_scales_tiles[first_tile + UInt32(g)]
            comptime for r in range(Self.reps):
                comptime for i in range(2):
                    a_sc[(g * Self.reps + r) * 2 + i] = a_scales_smem[
                        0, UInt32(r * 8 + i) + col0
                    ].cast[Self.accum_type]()

        syncwarp()
        if lane_id() < Self.cluster_size:
            epi_stage.arrive_input()
        syncwarp()

        var tmem = TmemAddress(UInt32(epi_stage.output_stage.tmem.offset()))
        var upper = tmem.load_upper[
            Self.accum_type,
            group * Self.half_size,
            16,
            256,
            group * Self.reps,
        ]()
        var lower = upper.copy()
        comptime if Self.halves == 2:
            lower = tmem.load_lower[
                Self.accum_type,
                group * Self.half_size,
                16,
                256,
                group * Self.reps,
            ]()
        TmemFragments[Self.accum_type, Self.half_size].wait_load()

        comptime for g in range(group):
            comptime for h in range(Self.halves):
                comptime for r in range(Self.reps):
                    comptime for j in range(2):
                        comptime for i in range(2):
                            comptime e = r * Self.per_rep + j * 2 + i
                            var scale = (
                                a_sc[(g * Self.reps + r) * 2 + i] * b_scale[g]
                            )
                            var p = (
                                upper[g * Self.half_size + e] if h
                                == 0 else lower[g * Self.half_size + e]
                            )
                            self.acc[h * Self.half_size + e] += p * scale

    @inline(.always)
    def store[
        c_type: DType
    ](
        self,
        c_ptr: MutPointer[Scalar[c_type], MutUntrackedOrigin],
        problem_shape: StaticTuple[Int32, 3],
        n0: Int,
        m0: Int,
    ):
        """Write the tile to row-major C (M x N), transposing C^T.

        Args:
            c_ptr: Base of C.
            problem_shape: `(M, N, K)` of the matmul.
            n0: First weight row (C column) of the tile.
            m0: First activation row (C row) of the tile.
        """
        var M = Int(problem_shape[0])
        var N = Int(problem_shape[1])
        var n_base = (
            n0
            + Int(get_warp_id()) * 16 * Self.halves
            + Int(ufloordiv(lane_id(), 4))
        )
        var m_base = m0 + Int(umod(lane_id(), 4)) * 2
        comptime for h in range(Self.halves):
            comptime for r in range(Self.reps):
                comptime for j in range(2):
                    comptime for i in range(2):
                        var n = n_base + h * 16 + j * 8
                        var m = m_base + r * 8 + i
                        if m < M and n < N:
                            c_ptr[m * N + n] = self.acc[
                                h * Self.half_size
                                + r * Self.per_rep
                                + j * 2
                                + i
                            ].cast[c_type]()
