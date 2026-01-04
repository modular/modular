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

"""TileWriter components for SM100 matrix multiplication epilogue.

This module provides modular components for the output pipeline:

1. **TMAStoreWriter**: TMA async store from shared memory to global memory
2. **StMatrixWriter**: Register to shared memory via st.matrix instructions
3. **EpilogueApplier**: Apply element-wise operations on fragments
4. **load_tmem_fragments**: Load accumulator data from TMEM (uses TmemAddress)

The SM100 epilogue pipeline flows as:
    TMEM (accumulators) → Registers → SMEM → GMEM (via TMA)

TMEM operations use TmemAddress from tmem.mojo for load/store abstraction.

Usage:
    # TMA store from shared memory to global memory
    var tma_writer = TMAStoreWriter[...](c_tma_op)
    tma_writer.store_tile(c_smem_tile, (n_coord, m_coord))
"""

from sys import align_of, simd_width_of

from gpu import WARP_SIZE, lane_id
from gpu import warp_id as get_warp_id
from gpu.memory import fence_async_view_proxy
from gpu.host.nvidia.tma import TensorMapSwizzle
from .barriers import WarpGroupBarrier
from layout import Layout, RuntimeLayout, UNKNOWN_VALUE, RuntimeTuple
from layout.int_tuple import IntTuple
from layout.layout import blocked_product, zipped_divide, upcast
from layout.runtime_tuple import idx2crd
from layout.swizzle import Swizzle, make_swizzle as _make_swizzle
from layout.tma_async import TMATensorTile
from linalg.structuring import SMemTileArrayType, SMemTileType
from linalg.utils import elementwise_compute_lambda_type
from utils.fast_div import FastDiv


# =============================================================================
# Helper type comptime for runtime layouts with 32-bit indices
# =============================================================================

comptime RLayout32Bits[layout: Layout] = RuntimeLayout[
    layout, element_type = DType.uint32, linear_idx_type = DType.uint32
]

# Import architecture-agnostic writers from SM90
# TMA writes and threadwise SMEM→GMEM work identically on H100 and B200
from linalg.matmul.gpu.sm90.tile_writer import (
    TileWriterTMA,
    TileWriterThreadwise,
    SMemTileWriter,  # Trait for SMEM→GMEM writers
)

# Aliases for SM100 naming convention
comptime TMAStoreWriter = TileWriterTMA
comptime ThreadwiseStoreWriter = TileWriterThreadwise


# =============================================================================
# tma_wait_pipelined - Pipelined TMA wait helper
# =============================================================================


@always_inline
fn tma_wait_pipelined[
    c_type: DType,
    c_layout: Layout,
    c_desc_layout: Layout,
    is_last_stage: Bool,
](c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],):
    """Wait for TMA stores with pipelining.

    For SM100 output pipeline:
    - Non-last stages: Keep 1 store in flight for pipelining
    - Last stage: Wait for all stores to complete

    Template Parameters:
        c_type: Output data type.
        c_layout: Global memory layout for C.
        c_desc_layout: TMA descriptor layout for C.
        is_last_stage: If True, wait for all; else keep 1 in flight.

    Args:
        c_tma_op: TMA tensor tile descriptor.
    """

    @parameter
    if is_last_stage:
        c_tma_op.wait_group[0]()  # Wait for all stores
    else:
        c_tma_op.wait_group[1]()  # Keep 1 store in flight


@always_inline
fn tma_store_with_pipeline[
    c_type: DType,
    c_layout: Layout,
    c_desc_layout: Layout,
    is_last_stage: Bool,
](
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    src: SMemTileType[c_type, _, alignment=128, **_],
    coords: Tuple[UInt, UInt],
):
    """Perform TMA store with pipelined commit and wait.

    Encapsulates the common SM100 output pattern:
    1. fence_async_view_proxy()
    2. async_store()
    3. commit_group()
    4. wait_group() with pipelining

    Template Parameters:
        c_type: Output data type.
        c_layout: Global memory layout for C.
        c_desc_layout: TMA descriptor layout for C.
        is_last_stage: If True, wait for all; else keep 1 in flight.

    Args:
        c_tma_op: TMA tensor tile descriptor.
        src: Source shared memory tile.
        coords: Destination coordinates in global memory.
    """
    fence_async_view_proxy()
    c_tma_op.async_store(src, coords)
    c_tma_op.commit_group()
    tma_wait_pipelined[c_type, c_layout, c_desc_layout, is_last_stage](c_tma_op)


# =============================================================================
# StMatrixCoords - Coordinate computation for st.matrix operations
# =============================================================================


@register_passable("trivial")
struct StMatrixCoords[
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    cta_group: Int,
    transpose_c: Bool,
]:
    """Compute coordinates for st.matrix operations.

    Encapsulates the complex coordinate calculations needed for storing
    accumulator fragments to shared memory.

    Template Parameters:
        MMA_M: MMA M dimension.
        MMA_N: MMA N dimension.
        stageN: Stage N dimension (width of each output tile).
        cta_group: Number of CTAs cooperating (1 or 2).
        transpose_c: Whether output is transposed.
    """

    var warp_id: UInt32
    var lane_id: UInt32
    var c_row: UInt32
    var c_col: UInt32

    @always_inline
    fn __init__(
        out self,
        warp_id: UInt32,
        lane_id: UInt32,
        c_row: UInt32,
        c_col: UInt32,
    ):
        """Initialize coordinate calculator.

        Args:
            warp_id: Warp ID within the CTA.
            lane_id: Lane ID within the warp.
            c_row: Base row coordinate in global memory.
            c_col: Base column coordinate in global memory.
        """
        self.warp_id = warp_id
        self.lane_id = lane_id
        self.c_row = c_row
        self.c_col = c_col

    @always_inline
    fn staged_row(self, stage: UInt32, num_stages: UInt32) -> UInt32:
        """Compute the staged row coordinate.

        Args:
            stage: Current stage index.
            num_stages: Total number of stages.

        Returns:
            Row coordinate for the current stage.
        """
        var staged_c_row = self.c_row

        @parameter
        if Self.MMA_M == 256 or (Self.MMA_M == 128 and Self.cta_group == 1):
            # Layout A/D
            staged_c_row += self.warp_id * 32
        elif Self.MMA_M == 64 and Self.cta_group == 1:
            # Layout F
            staged_c_row += self.warp_id * 16
        else:
            # Layout B (cta_group == 2)
            staged_c_row += (self.warp_id % 2) * 32

        return staged_c_row

    @always_inline
    fn staged_col(self, stage: UInt32, num_stages: UInt32) -> UInt32:
        """Compute the staged column coordinate.

        Args:
            stage: Current stage index.
            num_stages: Total number of stages.

        Returns:
            Column coordinate for the current stage.
        """
        var staged_c_col = self.c_col + stage * UInt32(Self.stageN)

        @parameter
        if not (
            Self.MMA_M == 256
            or (Self.MMA_M == 128 and Self.cta_group == 1)
            or (Self.MMA_M == 64 and Self.cta_group == 1)
        ):
            # Layout B (cta_group == 2) has column offset based on warp
            staged_c_col += (
                (self.warp_id // 2) * num_stages * UInt32(Self.stageN)
            )

        return staged_c_col

    @always_inline
    fn smem_coord_m(self) -> UInt32:
        """Compute shared memory M coordinate for TMA store.

        Returns:
            M coordinate in shared memory tile.
        """

        @parameter
        if Self.cta_group == 2 and Self.MMA_M != 256:
            return self.warp_id // 2
        else:
            return UInt32(0)


# =============================================================================
# TMEMFragment - Accumulator fragment loaded from tensor memory
# =============================================================================


@register_passable("trivial")
struct TMEMFragment[
    accum_type: DType,
    epilogue_type: DType,
    frag_size: Int,
]:
    """Accumulator fragment pair from tensor memory.

    SM100 TMEM stores data in upper/lower fragment pairs due to the
    physical layout of tensor memory datapaths.

    Template Parameters:
        accum_type: Accumulator data type (e.g., float32).
        epilogue_type: Epilogue data type after casting (e.g., bfloat16).
        frag_size: Number of elements per fragment.
    """

    var upper: SIMD[Self.accum_type, Self.frag_size]
    var lower: SIMD[Self.accum_type, Self.frag_size]
    var has_lower: Bool

    @always_inline
    fn __init__(out self, has_lower: Bool = True):
        """Initialize empty fragments.

        Args:
            has_lower: Whether lower fragment is needed (based on MMA config).
        """
        self.upper = SIMD[Self.accum_type, Self.frag_size]()
        self.lower = SIMD[Self.accum_type, Self.frag_size]()
        self.has_lower = has_lower

    @always_inline
    fn cast_upper(self) -> SIMD[Self.epilogue_type, Self.frag_size]:
        """Cast upper fragment to epilogue type.

        Returns:
            Upper fragment cast to epilogue_type.
        """
        return self.upper.cast[Self.epilogue_type]()

    @always_inline
    fn cast_lower(self) -> SIMD[Self.epilogue_type, Self.frag_size]:
        """Cast lower fragment to epilogue type.

        Returns:
            Lower fragment cast to epilogue_type.
        """
        return self.lower.cast[Self.epilogue_type]()


# =============================================================================
# AccumTile - Accumulator tile (upper + lower fragments) for writing
# =============================================================================


@register_passable("trivial")
struct AccumTile[dtype: DType, size: Int]:
    """Accumulator tile holding upper and lower fragment data.

    SM100 accumulators in TMEM are stored as two halves (upper 16 rows,
    lower 16 rows). This struct represents the complete tile being written.

    This is the SM100 equivalent of SM90's RegTileType - the data being
    written by the tile writer.

    Template Parameters:
        dtype: Data type of the fragments (typically epilogue_dtype).
        size: Number of elements per fragment.
    """

    var upper: SIMD[Self.dtype, Self.size]
    var lower: SIMD[Self.dtype, Self.size]

    @always_inline
    fn __init__(
        out self,
        upper: SIMD[Self.dtype, Self.size],
        lower: SIMD[Self.dtype, Self.size],
    ):
        """Create an accumulator tile from upper and lower fragments."""
        self.upper = upper
        self.lower = lower


# =============================================================================
# AccumBarrier - Accumulator pipeline barrier helper
# =============================================================================


@register_passable("trivial")
struct AccumBarrier[cta_group: Int]:
    """Helper for accumulator pipeline barrier operations.

    Handles the different arrival patterns for single-CTA vs 2-CTA groups.

    Template Parameters:
        cta_group: Number of CTAs cooperating (1 or 2).
    """

    @staticmethod
    @always_inline
    fn arrive(
        pipeline: ProducerConsumerPipeline,
        stage: UInt32,
    ):
        """Signal accumulator arrival.

        Args:
            pipeline: The MMA output pipeline.
            stage: Current pipeline stage.
        """

        @parameter
        if Self.cta_group == 1:
            from gpu.sync import mbarrier_arrive

            _ = mbarrier_arrive(pipeline.consumer_mbar(stage))
        else:
            from gpu.sync import umma_arrive_leader_cta

            umma_arrive_leader_cta(pipeline.consumer_mbar(stage))


# Import for AccumBarrier
from .pipeline import ProducerConsumerPipeline


# =============================================================================
# StMatrixConfig - Configuration for st.matrix operations
# =============================================================================


@register_passable("trivial")
struct StMatrixConfig[
    c_type: DType,
    stageN: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_c: Bool = False,
]:
    """Configuration for st.matrix store operations.

    Computes the various constants needed for st.matrix operations
    based on the output tile configuration.

    Template Parameters:
        c_type: Output data type (e.g., bfloat16).
        stageN: Stage width in elements.
        c_swizzle: TMA swizzle mode.
        transpose_c: Whether output is transposed.
    """

    # Number of elements per st.matrix row (32B for stsmx4, 16B for stsmx2)
    comptime stsmx_row_size = 32 // size_of[
        Self.c_type
    ]() if Self.stageN % 16 == 0 else 16 // size_of[Self.c_type]()

    # Number of elements per lane (each lane handles 16B)
    comptime stsmx_lane_size = 16 // size_of[Self.c_type]()

    # SIMD width for st_matrix instruction
    comptime stmtx_simd_width = 4 if Self.stageN % 16 == 0 else 2

    # Swizzle width in elements
    comptime swizzle_width = Self.c_swizzle.bytes() // size_of[Self.c_type]()

    @staticmethod
    @always_inline
    fn make_swizzle() -> Swizzle:
        """Create the swizzle pattern for st.matrix operations.

        Returns:
            Swizzle instance for the configured swizzle mode.
        """
        from layout.swizzle import make_swizzle as _make_swizzle

        return _make_swizzle[Self.c_type, Self.c_swizzle]()


# =============================================================================
# StMatrixWriter - Write register fragments to shared memory
# =============================================================================


@register_passable("trivial")
struct StMatrixWriter[
    c_type: DType,
    c_smem_layout: Layout,
    stageN: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_c: Bool = False,
]:
    """Write register fragments to shared memory using st.matrix.

    Handles the complex swizzling and addressing required for efficient
    shared memory writes from WGMMA accumulator fragments.

    Template Parameters:
        c_type: Output data type.
        c_smem_layout: Shared memory tile layout.
        stageN: Stage width in elements.
        c_swizzle: TMA swizzle mode.
        transpose_c: Whether output is transposed.
    """

    comptime Config = StMatrixConfig[
        Self.c_type, Self.stageN, Self.c_swizzle, Self.transpose_c
    ]

    # Computed layout parameters
    comptime stride0 = Self.c_smem_layout.stride[0].value()
    comptime stride1 = Self.c_smem_layout.stride[1].value()
    comptime shape0 = Self.c_smem_layout.shape[
        1
    ].value() if not Self.transpose_c else Self.c_smem_layout.shape[0].value()

    comptime stsmx_tile_offset = (
        Self.stride0 if Self.transpose_c else Self.stride1
    ) * Self.Config.stsmx_row_size

    var swizzle: Swizzle
    var lane_id: UInt32

    @always_inline
    fn __init__(out self, lane_id: UInt32):
        """Initialize the st.matrix writer.

        Args:
            lane_id: Lane ID within the warp.
        """
        self.swizzle = Self.Config.make_swizzle()
        self.lane_id = lane_id

    @always_inline
    fn compute_lane_offset(self) -> UInt32:
        """Compute the base lane offset for st.matrix.

        Returns:
            Lane offset in shared memory.
        """

        @parameter
        if Self.transpose_c:
            # Transposed layout uses different addressing
            from layout.int_tuple import IntTuple
            from layout.layout import Layout

            comptime trans_layout = Layout(
                IntTuple(8, 2, 2),
                IntTuple(Self.stride0, 8 * Self.stride1, 8 * Self.stride0),
            )
            return UInt32(RLayout32Bits[trans_layout]()(Int(self.lane_id)))
        else:
            return (self.lane_id & 15) * UInt32(Self.stride0) + (
                self.lane_id >> 4
            ) * 8

    @always_inline
    fn write_fragment[
        frag_size: Int
    ](
        self,
        frag: SIMD[_, frag_size],
        dst: SMemTileType[Self.c_type, Self.c_smem_layout, alignment=128],
        warp_offset: UInt32 = 0,
    ):
        """Write a fragment to shared memory using st.matrix.

        Args:
            frag: Source fragment (typically from TMEM load).
            dst: Destination shared memory tile.
            warp_offset: Additional warp-based offset for transpose mode.
        """
        from gpu.mma import st_matrix
        from memory import bitcast

        var lane_offset = self.compute_lane_offset()

        # Helper to slice SIMD vector (LLVM extract generates bad code on GPU)
        @always_inline
        fn slice[offset: Int, size: Int](v: SIMD) -> SIMD[v.dtype, size]:
            var tmp = SIMD[v.dtype, size]()

            @parameter
            for i in range(size):
                tmp[i] = v[i + offset]
            return tmp

        @parameter
        for i in range(Self.shape0 // Self.Config.stsmx_row_size):
            comptime n_offset = i * Self.stsmx_tile_offset
            var offset: UInt32

            @parameter
            if Self.transpose_c:
                offset = (
                    self.swizzle(lane_offset + n_offset + warp_offset)
                    - warp_offset
                )
            else:
                offset = self.swizzle(lane_offset + n_offset)

            var v = slice[
                i * Self.Config.stsmx_lane_size,
                2 * Self.Config.stmtx_simd_width,
            ](frag).cast[Self.c_type]()

            st_matrix[
                simd_width = Self.Config.stmtx_simd_width,
                transpose = Self.transpose_c,
            ](
                dst.ptr + offset,
                bitcast[DType.float32, Self.Config.stmtx_simd_width](v),
            )


# =============================================================================
# store_fragment_to_smem - Static helper matching stsm_helper interface
# =============================================================================


@always_inline
fn store_fragment_to_smem[
    swizzle: Swizzle,
    stageN: Int,
    transpose_c: Bool = False,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](vec: SIMD, dst: SMemTileType, warp_offset: UInt32 = 0,):
    """Store a fragment to shared memory using st.matrix.

    This function provides a static interface compatible with stsm_helper,
    delegating to the underlying st.matrix operations.

    Template Parameters:
        swizzle: Pre-computed swizzle pattern.
        stageN: Stage width in elements.
        transpose_c: Whether output is transposed.
        c_swizzle: TMA swizzle mode (for configuration).

    Args:
        vec: Source SIMD fragment.
        dst: Destination shared memory tile.
        warp_offset: Additional warp-based offset for transpose mode.
    """
    from gpu.mma import st_matrix
    from memory import bitcast
    from gpu import lane_id as get_lane_id

    comptime c_type = dst.dtype

    # Configuration constants
    comptime stsmx_row_size = 32 // size_of[
        c_type
    ]() if stageN % 16 == 0 else 16 // size_of[c_type]()
    comptime stsmx_lane_size = 16 // size_of[c_type]()
    comptime stmtx_simd_width = 4 if stageN % 16 == 0 else 2

    # Layout parameters
    comptime stride0 = dst.layout.stride[0].value()
    comptime stride1 = dst.layout.stride[1].value()
    comptime shape0 = dst.layout.shape[
        1
    ].value() if not transpose_c else dst.layout.shape[0].value()
    comptime stsmx_tile_offset = (
        stride0 if transpose_c else stride1
    ) * stsmx_row_size

    var lane = get_lane_id()

    # Compute lane offset
    var stsm_lane_offset: UInt32

    @parameter
    if transpose_c:
        from layout.int_tuple import IntTuple

        comptime trans_layout = Layout(
            IntTuple(8, 2, 2), IntTuple(stride0, 8 * stride1, 8 * stride0)
        )
        stsm_lane_offset = UInt32(RLayout32Bits[trans_layout]()(Int(lane)))
    else:
        stsm_lane_offset = (lane & 15) * UInt32(stride0) + (lane >> 4) * 8

    # Helper to slice SIMD vector
    @always_inline
    fn slice[offset: Int, size: Int](v: SIMD) -> SIMD[v.dtype, size]:
        var tmp = SIMD[v.dtype, size]()

        @parameter
        for i in range(size):
            tmp[i] = v[i + offset]
        return tmp

    # Store each portion of the fragment
    @parameter
    for i in range(shape0 // stsmx_row_size):
        comptime n_offset = i * stsmx_tile_offset
        var offset: UInt32

        @parameter
        if transpose_c:
            offset = (
                swizzle(stsm_lane_offset + n_offset + warp_offset) - warp_offset
            )
        else:
            offset = swizzle(stsm_lane_offset + n_offset)

        var v = slice[i * stsmx_lane_size, 2 * stmtx_simd_width](vec).cast[
            c_type
        ]()

        st_matrix[simd_width=stmtx_simd_width, transpose=transpose_c](
            dst.ptr + offset, bitcast[DType.float32, stmtx_simd_width](v)
        )


# =============================================================================
# load_tmem_fragments - Load and cast TMEM fragments
# =============================================================================


@always_inline
fn load_tmem_fragments[
    accum_type: DType,
    epilogue_type: DType,
    frag_size: Int,
    is_lower_required: Bool,
    data_paths: Int = 16,
    bits: Int = 256,
    repeat: Int = 1,
](
    tmem_addr: UInt32,
) -> Tuple[
    SIMD[epilogue_type, frag_size * repeat],
    SIMD[epilogue_type, frag_size * repeat],
]:
    """Load upper and lower fragments from TMEM and cast to epilogue type.

    This encapsulates the common pattern of loading accumulator data from
    tensor memory, waiting for completion, and casting to output type.

    Template Parameters:
        accum_type: Accumulator data type (e.g., float32).
        epilogue_type: Output data type after casting (e.g., bfloat16).
        frag_size: Base fragment size per warp.
        is_lower_required: Whether lower fragment is needed.
        data_paths: TMEM data paths (default 16).
        bits: TMEM bits width (default 256).
        repeat: Repeat factor for larger fragments.

    Args:
        tmem_addr: Tensor memory address for this stage.

    Returns:
        Tuple of (upper_casted, lower_casted) SIMD fragments.
    """
    from .tmem import TmemAddress

    comptime width = frag_size * repeat
    var tmem = TmemAddress(tmem_addr)

    # Load fragments using TmemAddress abstraction
    var upper_frag = tmem.load_upper[
        accum_type, width, data_paths, bits, repeat
    ]()

    var lower_frag = SIMD[accum_type, width]()

    @parameter
    if is_lower_required:
        lower_frag = tmem.load_lower[
            accum_type, width, data_paths, bits, repeat
        ]()

    TmemAddress.wait_load()

    # Cast and return
    var upper_casted = upper_frag.cast[epilogue_type]()
    var lower_casted = SIMD[epilogue_type, width]()

    @parameter
    if is_lower_required:
        lower_casted = lower_frag.cast[epilogue_type]()

    return (upper_casted, lower_casted)


# =============================================================================
# EpilogueConfig - Configuration for epilogue operations
# =============================================================================


@register_passable("trivial")
struct EpilogueConfig[
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    cta_group: Int,
    transpose_c: Bool,
]:
    """Configuration for epilogue stage computations.

    Computes the number of stages and other parameters needed for
    the output epilogue based on MMA and CTA configuration.

    Template Parameters:
        MMA_M: MMA M dimension.
        MMA_N: MMA N dimension.
        stageN: Stage width in elements.
        cta_group: Number of CTAs cooperating (1 or 2).
        transpose_c: Whether output is transposed.
    """

    # Whether lower fragment is needed (not for cta_group=1, MMA_M=64)
    comptime is_lower_frag_required = not (
        Self.cta_group == 1 and Self.MMA_M == 64
    )

    # Number of stages for output
    comptime cg2_num_stages = Self.MMA_N // Self.stageN if Self.MMA_M == 256 else Self.MMA_N // Self.stageN // 2
    comptime cg1_num_stages = Self.MMA_N // Self.stageN
    comptime num_stages = Self.cg2_num_stages if Self.cta_group == 2 else Self.cg1_num_stages

    # Fragment configuration
    comptime data_paths = 16
    comptime bits = 256
    comptime fragment_size = (
        Self.data_paths * (Self.bits // 32)
    ) // 32  # Per warp


# =============================================================================
# TMAStoreCoords - Compute coordinates for TMA store operations
# =============================================================================


@register_passable("trivial")
struct TMAStoreCoords[
    BM: Int,
    BN: Int,
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    cta_group: Int,
    c_smem_shape0: Int,  # c_smem_tile.layout.shape[0].value()
    stage: Int,  # Current output stage index (compile-time)
]:
    """Compute TMA store coordinates and warp election for SM100 epilogue.

    Encapsulates the complex coordinate computation logic for TMA stores,
    including cta_group-specific branching and warp election.

    Template Parameters:
        BM: Block M dimension.
        BN: Block N dimension.
        MMA_M: MMA M dimension.
        MMA_N: MMA N dimension.
        stageN: Stage width in elements.
        cta_group: Number of CTAs cooperating (1 or 2).
        c_smem_shape0: Shape[0] of shared memory tile layout.
        stage: Current output stage index.
    """

    # Compute TMA_BM at compile time
    comptime CG2_TMA_BM = Self.c_smem_shape0 if Self.MMA_M == 256 else Self.BM
    comptime CG1_TMA_BM = Self.c_smem_shape0
    comptime TMA_BM = Self.CG2_TMA_BM if Self.cta_group == 2 else Self.CG1_TMA_BM

    # Compile-time stage offset for N coordinate
    comptime stage_n_offset = Self.stage * Self.stageN

    # Runtime values
    var coord_m: UInt
    var coord_n: UInt
    var elect_one_warp: Bool
    var c_smem_coord_m: UInt

    @always_inline
    fn __init__(
        out self,
        c_coord: Tuple[UInt32, UInt32],
        warp_id: UInt32,
    ):
        """Compute all TMA store coordinates.

        Args:
            c_coord: Output tile coordinates (m_tile, n_tile).
            warp_id: Current warp ID.
        """
        # Warp election logic
        var cg2_elect_one_warp = (
            warp_id == 0 if Self.MMA_M == 256 else warp_id % 2 == 0
        )
        var cg1_elect_one_warp = warp_id == 0
        self.elect_one_warp = (
            cg2_elect_one_warp if Self.cta_group == 2 else cg1_elect_one_warp
        )

        # N coordinate computation (stage offset is compile-time)
        var coord_n_mma_m256 = c_coord[1] * UInt(Self.MMA_N) + UInt(
            Self.stage_n_offset
        )
        var coord_n_mma_m128 = (
            c_coord[1] * UInt(Self.MMA_N)
            + UInt(Self.stage_n_offset)
            + UInt(Self.BN * Int(warp_id // 2))
        )

        var cg2_coord_n = (
            coord_n_mma_m256 if Self.MMA_M == 256 else coord_n_mma_m128
        )
        var cg1_coord_n = coord_n_mma_m256
        self.coord_n = UInt(cg2_coord_n if Self.cta_group == 2 else cg1_coord_n)

        # M coordinate
        self.coord_m = UInt(c_coord[0]) * UInt(Self.BM)

        # SMEM coordinate for tile selection (non-transpose path)
        var cg2_c_smem_coord_m = UInt(
            0 if Self.MMA_M == 256 else Int(warp_id // 2)
        )
        var cg1_c_smem_coord_m = UInt(0)
        self.c_smem_coord_m = (
            cg2_c_smem_coord_m if Self.cta_group == 2 else cg1_c_smem_coord_m
        )


# =============================================================================
# TMAStoreExecutor - Execute TMA stores with proper SMEM tiling
# =============================================================================


@register_passable("trivial")
struct TMAStoreExecutor[
    c_type: DType,
    c_smem_layout: Layout,
    BM: Int,
    BN: Int,
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    stage_contiguous_size: Int,
    cta_group: Int,
    c_swizzle: TensorMapSwizzle,
    transpose_c: Bool,
    is_lower_frag_required: Bool,
]:
    """Execute TMA store from shared memory to global memory with proper tiling.

    Encapsulates all the complex SMEM tiling/reshaping logic for TMA stores.
    Handles 3 distinct paths based on transpose_c, cta_group, and MMA_M:

    1. transpose_c + cta_group==2 + MMA_M==128: Split reshape
    2. transpose_c + other: Loop over swizzle-width tiles
    3. non-transpose: Simple tile selection

    Template Parameters:
        c_type: Output data type.
        c_smem_layout: Shared memory layout for C tile.
        BM: Block M dimension.
        BN: Block N dimension.
        MMA_M: MMA M dimension.
        MMA_N: MMA N dimension.
        stageN: Stage width in elements.
        stage_contiguous_size: Contiguous size in SMEM layout.
        cta_group: Number of CTAs cooperating (1 or 2).
        c_swizzle: TensorMap swizzle mode.
        transpose_c: Whether output is transposed.
        is_lower_frag_required: Whether lower fragment is used.
    """

    # Compile-time computed values
    comptime swizzle_width = Self.c_swizzle.bytes() // size_of[Self.c_type]()

    # Number of tiles for transpose path B
    comptime num_c_smem_tiles = 128 // Self.swizzle_width // (
        1 if Self.is_lower_frag_required else 2
    )

    # TMA_BM from TMAStoreCoords logic
    comptime c_smem_shape0 = Self.c_smem_layout.shape[0].value()
    comptime CG2_TMA_BM = Self.c_smem_shape0 if Self.MMA_M == 256 else Self.BM
    comptime CG1_TMA_BM = Self.c_smem_shape0
    comptime TMA_BM = Self.CG2_TMA_BM if Self.cta_group == 2 else Self.CG1_TMA_BM

    @staticmethod
    @always_inline
    fn execute[
        c_layout: Layout,
        c_desc_layout: Layout,
    ](
        c_smem_tile: SMemTileType[
            Self.c_type, Self.c_smem_layout, alignment=128
        ],
        store_coords: TMAStoreCoords[
            Self.BM,
            Self.BN,
            Self.MMA_M,
            Self.MMA_N,
            Self.stageN,
            Self.cta_group,
            Self.c_smem_shape0,
            _,  # stage - any value works since we only use coord fields
        ],
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Execute TMA store with appropriate tiling for the configuration.

        Args:
            c_smem_tile: Source shared memory tile.
            store_coords: Precomputed TMA store coordinates.
            c_tma_op: TMA tensor tile for async store operations.
            warp_id: Current warp ID.
            lane: Current lane ID within warp.
        """
        if store_coords.elect_one_warp and lane == 0:
            fence_async_view_proxy()

            @parameter
            if Self.transpose_c:
                Self._store_transpose[c_layout, c_desc_layout](
                    c_smem_tile, store_coords, c_tma_op, warp_id
                )
            else:
                Self._store_non_transpose[c_layout, c_desc_layout](
                    c_smem_tile, store_coords, c_tma_op
                )

            c_tma_op.commit_group()

    @staticmethod
    @always_inline
    fn _store_transpose[
        c_layout: Layout,
        c_desc_layout: Layout,
    ](
        c_smem_tile: SMemTileType[
            Self.c_type, Self.c_smem_layout, alignment=128
        ],
        store_coords: TMAStoreCoords[
            Self.BM,
            Self.BN,
            Self.MMA_M,
            Self.MMA_N,
            Self.stageN,
            Self.cta_group,
            Self.c_smem_shape0,
            _,
        ],
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
        warp_id: UInt32,
    ):
        """Handle transpose_c TMA store paths."""

        @parameter
        if Self.cta_group == 2 and Self.MMA_M == 128:
            # Path A: cta_group==2 with MMA_M==128
            # Reshape to (2*stageN, stage_contiguous_size//2), tile by warp
            var c_smem_reshaped = c_smem_tile.reshape[
                Layout.row_major(
                    2 * Self.stageN, Self.stage_contiguous_size // 2
                )
            ]()
            var c_smem_split = c_smem_reshaped.tile[
                Self.stageN, Self.stage_contiguous_size // 2
            ](Int(warp_id // 2), 0)

            c_tma_op.async_store(
                c_smem_split,
                (store_coords.coord_m, store_coords.coord_n),
            )
        else:
            # Path B: Other transpose cases - loop over swizzle tiles
            @parameter
            for i in range(Self.num_c_smem_tiles):
                var c_smem_warp_tile = c_smem_tile.tile[
                    Self.stageN
                    * Self.swizzle_width
                    // Self.stage_contiguous_size,
                    Self.stage_contiguous_size,
                ](i, 0).reshape[
                    Layout.row_major(Self.stageN, Self.swizzle_width)
                ]()

                c_tma_op.async_store(
                    c_smem_warp_tile,
                    (
                        store_coords.coord_m + UInt(i * Self.swizzle_width),
                        store_coords.coord_n,
                    ),
                )

    @staticmethod
    @always_inline
    fn _store_non_transpose[
        c_layout: Layout,
        c_desc_layout: Layout,
    ](
        c_smem_tile: SMemTileType[
            Self.c_type, Self.c_smem_layout, alignment=128
        ],
        store_coords: TMAStoreCoords[
            Self.BM,
            Self.BN,
            Self.MMA_M,
            Self.MMA_N,
            Self.stageN,
            Self.cta_group,
            Self.c_smem_shape0,
            _,
        ],
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
    ):
        """Handle non-transpose TMA store path."""
        # Path C: Simple tile selection by TMA_BM
        # Note: coords are (coord_n, coord_m) - swapped for non-transpose!
        var c_smem_split = c_smem_tile.tile[Self.TMA_BM, Self.stageN](
            Int(store_coords.c_smem_coord_m), 0
        )
        c_tma_op.async_store(
            c_smem_split,
            (store_coords.coord_n, store_coords.coord_m),
        )


# =============================================================================
# FragmentCoords - Coordinate tracking for fragment elements
# =============================================================================


@register_passable("trivial")
struct FragmentCoords[
    stageN: Int,
    repeats: Int,
]:
    """Compute coordinates for fragment elements in tensor memory layout.

    Based on tcgen05 matrix fragment layout (16x256b):
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b

    Template Parameters:
        stageN: Stage width in elements.
        repeats: Number of repetitions for wider loads.
    """

    comptime load_width = 2
    comptime threads_per_row = Self.stageN // Self.repeats // Self.load_width

    var top_upper: StaticTuple[UInt32, 2]
    var bottom_upper: StaticTuple[UInt32, 2]
    var top_lower: StaticTuple[UInt32, 2]
    var bottom_lower: StaticTuple[UInt32, 2]

    @always_inline
    fn __init__(out self, lane_id: UInt32):
        """Initialize fragment coordinates based on lane ID.

        Args:
            lane_id: Lane ID within the warp.
        """
        # Top-left coordinate based on lane position
        var row = lane_id // UInt32(Self.threads_per_row)
        var col = (lane_id % UInt32(Self.threads_per_row)) * UInt32(
            Self.load_width
        )

        self.top_upper = StaticTuple[UInt32, 2](row, col)
        # Bottom is 8 rows below top
        self.bottom_upper = StaticTuple[UInt32, 2](row + 8, col)
        # Lower fragment is 16 rows below upper
        self.top_lower = StaticTuple[UInt32, 2](row + 16, col)
        self.bottom_lower = StaticTuple[UInt32, 2](row + 24, col)


# =============================================================================
# EpilogueApplier - Apply element-wise operations on fragments
# =============================================================================


@register_passable("trivial")
struct EpilogueApplier[
    MMA_M: Int,
    stageN: Int,
    num_stages: Int,
    repeats: Int,
    cta_group: Int,
    transpose_c: Bool,
]:
    """Apply element-wise epilogue operations on register fragments.

    Computes global coordinates for each element and applies a lambda function.
    Handles different MMA layouts (A/B/D/F) and transpose modes.

    Template Parameters:
        MMA_M: MMA M dimension.
        stageN: Stage width in elements.
        num_stages: Number of output stages.
        repeats: Number of repetitions per load.
        cta_group: Number of CTAs cooperating (1 or 2).
        transpose_c: Whether output is transposed.
    """

    comptime Coords = FragmentCoords[Self.stageN, Self.repeats]

    var coords: Self.Coords
    var warp_id: UInt32
    var lane_id: UInt32

    @always_inline
    fn __init__(out self, warp_id: UInt32, lane_id: UInt32):
        """Initialize the epilogue applier.

        Args:
            warp_id: Warp ID within the CTA.
            lane_id: Lane ID within the warp.
        """
        self.coords = Self.Coords(lane_id)
        self.warp_id = warp_id
        self.lane_id = lane_id

    @always_inline
    fn compute_staged_coords(
        self, stage: UInt32, c_row: UInt32, c_col: UInt32
    ) -> Tuple[UInt32, UInt32]:
        """Compute staged row and column coordinates.

        Args:
            stage: Current stage index.
            c_row: Base row coordinate.
            c_col: Base column coordinate.

        Returns:
            Tuple of (staged_row, staged_col).
        """
        var staged_c_col = c_col + stage * UInt32(Self.stageN)
        var staged_c_row = c_row

        @parameter
        if Self.MMA_M == 256 or (Self.MMA_M == 128 and Self.cta_group == 1):
            # Layout A/D
            staged_c_row += self.warp_id * 32
        elif Self.MMA_M == 64 and Self.cta_group == 1:
            # Layout F
            staged_c_row += self.warp_id * 16
        else:
            # Layout B (cta_group == 2)
            staged_c_row += (self.warp_id % 2) * 32
            staged_c_col += (self.warp_id // 2) * UInt32(
                Self.num_stages * Self.stageN
            )

        return (staged_c_row, staged_c_col)

    @always_inline
    fn apply_to_fragment[
        epilogue_dtype: DType,
        frag_size: Int,
        compute_lambda_fn: elementwise_compute_lambda_type,
    ](
        self,
        mut frag: SIMD[epilogue_dtype, frag_size],
        staged_row: UInt32,
        staged_col: UInt32,
        is_upper: Bool,
    ):
        """Apply epilogue lambda to a fragment.

        Args:
            frag: Fragment to apply epilogue to (modified in place).
            staged_row: Staged row coordinate.
            staged_col: Staged column coordinate.
            is_upper: Whether this is the upper or lower fragment.
        """
        var top_coord = (
            self.coords.top_upper if is_upper else self.coords.top_lower
        )
        var bottom_coord = (
            self.coords.bottom_upper if is_upper else self.coords.bottom_lower
        )

        @parameter
        for rep in range(Self.repeats):
            comptime inc = rep * 8
            comptime offset = rep * 4

            # Compute global coordinates
            var top_row = staged_row + top_coord[0]
            var top_col = staged_col + top_coord[1] + UInt32(inc)
            var bottom_row = staged_row + bottom_coord[0]
            var bottom_col = staged_col + bottom_coord[1] + UInt32(inc)

            # Extract the 4 elements for this repeat
            var elem0 = frag[offset]
            var elem1 = frag[offset + 1]
            var elem2 = frag[offset + 2]
            var elem3 = frag[offset + 3]

            # Apply lambda to each element with proper coordinates
            # Lambda is instantiated with width=1 (scalar)
            @parameter
            if Self.transpose_c:
                elem0 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(top_col), Int(top_row)), elem0
                )
                elem1 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(top_col + 1), Int(top_row)), elem1
                )
                elem2 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(bottom_col), Int(bottom_row)), elem2
                )
                elem3 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(bottom_col + 1), Int(bottom_row)), elem3
                )
            else:
                elem0 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(top_row), Int(top_col)), elem0
                )
                elem1 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(top_row), Int(top_col + 1)), elem1
                )
                elem2 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(bottom_row), Int(bottom_col)), elem2
                )
                elem3 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(bottom_row), Int(bottom_col + 1)), elem3
                )

            # Store back
            frag[offset] = elem0
            frag[offset + 1] = elem1
            frag[offset + 2] = elem2
            frag[offset + 3] = elem3

    @always_inline
    fn apply_to_both_fragments[
        epilogue_dtype: DType,
        frag_size: Int,
        compute_lambda_fn: elementwise_compute_lambda_type,
        is_lower_frag_required: Bool,
    ](
        self,
        mut upper_frag: SIMD[epilogue_dtype, frag_size],
        mut lower_frag: SIMD[epilogue_dtype, frag_size],
        stage: UInt32,
        c_row: UInt32,
        c_col: UInt32,
    ) -> Tuple[
        SIMD[epilogue_dtype, frag_size], SIMD[epilogue_dtype, frag_size]
    ]:
        """Apply epilogue lambda to both upper and lower fragments.

        This is the main entry point for register-based epilogue, replacing
        the standalone register_epilogue function.

        Args:
            upper_frag: Upper fragment to apply epilogue to.
            lower_frag: Lower fragment to apply epilogue to.
            stage: Current stage index.
            c_row: Base row coordinate.
            c_col: Base column coordinate.

        Returns:
            Tuple of (modified upper_frag, modified lower_frag).
        """
        var staged_row, staged_col = self.compute_staged_coords(
            stage, c_row, c_col
        )

        # Apply to upper fragment
        self.apply_to_fragment[epilogue_dtype, frag_size, compute_lambda_fn](
            upper_frag, staged_row, staged_col, is_upper=True
        )

        # Apply to lower fragment if required
        @parameter
        if is_lower_frag_required:
            self.apply_to_fragment[
                epilogue_dtype, frag_size, compute_lambda_fn
            ](lower_frag, staged_row, staged_col, is_upper=False)

        return (upper_frag, lower_frag)


# =============================================================================
# OutputStageWriter - Orchestrate a single output stage
# =============================================================================


@register_passable("trivial")
struct OutputStageWriter[
    c_type: DType,
    c_smem_layout: Layout,
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    cta_group: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_c: Bool = False,
]:
    """Orchestrate writing a single output stage.

    Coordinates TMEM read, optional epilogue, st.matrix to SMEM, and TMA store.

    Template Parameters:
        c_type: Output data type.
        c_smem_layout: Shared memory tile layout.
        MMA_M: MMA M dimension.
        MMA_N: MMA N dimension.
        stageN: Stage width in elements.
        cta_group: Number of CTAs cooperating.
        c_swizzle: TMA swizzle mode.
        transpose_c: Whether output is transposed.
    """

    comptime Config = EpilogueConfig[
        Self.MMA_M, Self.MMA_N, Self.stageN, Self.cta_group, Self.transpose_c
    ]
    comptime StWriter = StMatrixWriter[
        Self.c_type,
        Self.c_smem_layout,
        Self.stageN,
        Self.c_swizzle,
        Self.transpose_c,
    ]

    var st_writer: Self.StWriter
    var warp_id: UInt32
    var lane_id: UInt32

    @always_inline
    fn __init__(out self, warp_id: UInt32, lane_id: UInt32):
        """Initialize the output stage writer.

        Args:
            warp_id: Warp ID within the CTA.
            lane_id: Lane ID within the warp.
        """
        self.st_writer = Self.StWriter(lane_id)
        self.warp_id = warp_id
        self.lane_id = lane_id

    @always_inline
    fn write_upper_fragment[
        frag_size: Int, epilogue_dtype: DType
    ](
        self,
        frag: SIMD[epilogue_dtype, frag_size],
        dst: SMemTileType[Self.c_type, Self.c_smem_layout, alignment=128],
        warp_offset: UInt32 = 0,
    ):
        """Write the upper fragment to shared memory.

        Args:
            frag: Upper fragment (already cast to epilogue_dtype).
            dst: Destination shared memory tile.
            warp_offset: Additional warp-based offset for transpose mode.
        """
        self.st_writer.write_fragment(frag, dst, warp_offset)

    @always_inline
    fn write_lower_fragment[
        frag_size: Int, epilogue_dtype: DType
    ](
        self,
        frag: SIMD[epilogue_dtype, frag_size],
        dst: SMemTileType[Self.c_type, Self.c_smem_layout, alignment=128],
        warp_offset: UInt32 = 0,
    ):
        """Write the lower fragment to shared memory.

        Args:
            frag: Lower fragment (already cast to epilogue_dtype).
            dst: Destination shared memory tile.
            warp_offset: Additional warp-based offset for transpose mode.
        """
        self.st_writer.write_fragment(frag, dst, warp_offset)


# =============================================================================
# TMEMToSMemWriter - Write TMEM accumulators to shared memory (SM100-specific)
# =============================================================================


@register_passable("trivial")
struct TMEMToSMemWriter[
    c_type: DType,
    accum_type: DType,
    c_smem_layout: Layout,
    BM: Int,
    BN: Int,
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    cta_group: Int,
    num_output_warps: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_c: Bool = False,
]:
    """Write TMEM accumulator fragments to shared memory for SM100.

    This is the SM100-specific equivalent of SM90's FragmentToSMemWriter.
    Key difference: SM100 accumulators live in Tensor Memory (TMEM),
    not registers, so we need tcgen05_ld to load them first.

    Handles three tile reshaping cases:
    1. transpose_c + is_lower_frag_required: 2 warps share swizzle blocks
    2. transpose_c + !is_lower_frag_required: 4 warps, upper only
    3. !transpose_c: Simple row-major tiling

    Template Parameters:
        c_type: Output data type (e.g., bfloat16).
        accum_type: Accumulator data type (e.g., float32).
        c_smem_layout: Shared memory tile layout.
        BM: Block M dimension.
        BN: Block N dimension.
        MMA_M: MMA M dimension.
        MMA_N: MMA N dimension.
        stageN: Stage N dimension.
        cta_group: Number of CTAs cooperating (1 or 2).
        num_output_warps: Number of warps participating in output.
        c_swizzle: TMA swizzle mode.
        transpose_c: Whether output is transposed.
    """

    comptime Config = EpilogueConfig[
        Self.MMA_M, Self.MMA_N, Self.stageN, Self.cta_group, Self.transpose_c
    ]

    # Computed layout parameters
    comptime swizzle_width = Self.c_swizzle.bytes() // size_of[Self.c_type]()
    comptime stage_contiguous_size = Self.c_smem_layout.shape[1].value()
    comptime data_paths = 16

    # Compile-time swizzle pattern
    comptime swizzle = _make_swizzle[Self.c_type, Self.c_swizzle]()

    var warp_id: UInt32
    var lane_id: UInt32

    @always_inline
    fn __init__(out self, warp_id: UInt32, lane_id: UInt32):
        """Initialize the TMEM to SMEM writer.

        Args:
            warp_id: Warp ID within the CTA.
            lane_id: Lane ID within the warp.
        """
        self.warp_id = warp_id
        self.lane_id = lane_id

    @always_inline
    fn write_stage[
        repeat: Int,
        bits: Int = 256,
    ](
        self,
        tmem_addr: UInt32,
        stage: Int,
        c_smem_tile: SMemTileType[
            Self.c_type, Self.c_smem_layout, alignment=128
        ],
    ):
        """Write a single stage from TMEM to shared memory with tile reshaping.

        Automatically handles the correct tile reshaping based on transpose_c
        and is_lower_frag_required configuration.

        Template Parameters:
            repeat: Repeat factor for fragment loading.
            bits: TMEM bits width (default 256).

        Args:
            tmem_addr: Base tensor memory address.
            stage: Current stage index.
            c_smem_tile: Base shared memory tile (will be reshaped internally).
        """
        from .tmem import TmemAddress

        comptime frag_size = Self.Config.fragment_size
        comptime is_lower_required = Self.Config.is_lower_frag_required
        comptime width = frag_size * repeat

        # Compute stage TMEM address using TmemAddress abstraction
        var tmem = TmemAddress(tmem_addr + UInt32(stage * Self.stageN))

        # Load fragments
        var upper_frag = tmem.load_upper[
            Self.accum_type, width, Self.data_paths, bits, repeat
        ]()

        var lower_frag = SIMD[Self.accum_type, width]()

        @parameter
        if is_lower_required:
            lower_frag = tmem.load_lower[
                Self.accum_type, width, Self.data_paths, bits, repeat
            ]()

        TmemAddress.wait_load()

        # Cast and write fragments
        self.write_fragments[repeat](
            upper_frag.cast[Self.c_type](),
            lower_frag.cast[Self.c_type](),
            c_smem_tile,
        )

    @always_inline
    fn write_fragments[
        repeat: Int,
    ](
        self,
        upper_frag: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        lower_frag: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        c_smem_tile: SMemTileType[
            Self.c_type, Self.c_smem_layout, alignment=128
        ],
    ):
        """Write pre-loaded fragments to shared memory with tile reshaping.

        Use this when fragments are loaded separately (e.g., with load_tmem_fragments)
        and need to be written after applying register-based epilogue.

        Template Parameters:
            repeat: Repeat factor matching the fragment size.

        Args:
            upper_frag: Upper fragment (already casted to c_type).
            lower_frag: Lower fragment (already casted to c_type, ignored if not needed).
            c_smem_tile: Base shared memory tile (will be reshaped internally).
        """
        comptime is_lower_required = Self.Config.is_lower_frag_required

        @parameter
        if Self.transpose_c:
            self._write_transpose[repeat, is_lower_required](
                upper_frag,
                lower_frag,
                c_smem_tile,
            )
        else:
            self._write_non_transpose[repeat, is_lower_required](
                upper_frag,
                lower_frag,
                c_smem_tile,
            )

    @always_inline
    fn _write_transpose[
        repeat: Int,
        is_lower_required: Bool,
    ](
        self,
        upper_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        lower_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        c_smem_tile: SMemTileType[
            Self.c_type, Self.c_smem_layout, alignment=128
        ],
    ):
        """Handle transposed output with tile reshaping."""

        @parameter
        if is_lower_required:
            # Case 1: transpose_c + is_lower_frag_required
            # Layout: row_major(stageN, 2, tile_width=32)
            comptime tile_width = 32
            comptime smem_swblock_layout = Layout.row_major(
                Self.stageN, 2, tile_width
            )
            comptime num_swblocks = Self.stage_contiguous_size // Self.swizzle_width
            comptime smem_logical_layout = Layout(
                flatten([num_swblocks, smem_swblock_layout.shape]),
                flatten(
                    [
                        Self.stageN * Self.swizzle_width,
                        smem_swblock_layout.stride,
                    ]
                ),
            )

            var new_smem = SMemTileType[
                Self.c_type,
                smem_logical_layout,
                alignment = c_smem_tile.alignment,
            ](c_smem_tile.ptr)

            warp_j, warp_i = divmod(Int(self.warp_id), 2)
            var _c_smem_warp_tile = new_smem.tile[
                1, Self.stageN, 1, tile_width
            ](warp_j, 0, warp_i, 0)
            var c_smem_warp_tile = _c_smem_warp_tile.reshape[
                coalesce(_c_smem_warp_tile.layout)
            ]()

            var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                Self.stageN, Self.data_paths
            ](0, 0)
            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                Self.stageN, Self.data_paths
            ](0, 1)

            var warp_offset = warp_i * tile_width
            store_fragment_to_smem[
                Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
            ](upper_casted, c_smem_warp_tile_upper, UInt32(warp_offset))

            warp_offset += tile_width // 2
            store_fragment_to_smem[
                Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
            ](lower_casted, c_smem_warp_tile_lower, UInt32(warp_offset))
        else:
            # Case 2: transpose_c + !is_lower_frag_required
            # Layout: row_major(stageN, 4, tile_width=16)
            comptime tile_width = 16
            comptime smem_logical_layout = Layout.row_major(
                Self.stageN, 4, tile_width
            )

            var new_smem = SMemTileType[
                Self.c_type,
                smem_logical_layout,
                alignment = c_smem_tile.alignment,
            ](c_smem_tile.ptr)
            var _c_smem_warp_tile = new_smem.tile[Self.stageN, 1, tile_width](
                0, Int(self.warp_id), 0
            )
            var c_smem_warp_tile = _c_smem_warp_tile.reshape[
                coalesce(_c_smem_warp_tile.layout)
            ]()

            var warp_offset = Int(self.warp_id) * tile_width
            store_fragment_to_smem[
                Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
            ](upper_casted, c_smem_warp_tile, UInt32(warp_offset))

    @always_inline
    fn _write_non_transpose[
        repeat: Int,
        is_lower_required: Bool,
    ](
        self,
        upper_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        lower_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        c_smem_tile: SMemTileType[
            Self.c_type, Self.c_smem_layout, alignment=128
        ],
    ):
        """Handle non-transposed output with simple row tiling."""
        # Case 3: !transpose_c - Simple row-major tiling
        comptime c_smem_tile_m = 32 if Self.cta_group == 2 else Self.BM // Self.num_output_warps
        var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, Self.stageN](
            Int(self.warp_id), 0
        )

        var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
            Self.data_paths, Self.stageN
        ](0, 0)
        store_fragment_to_smem[
            Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
        ](upper_casted, c_smem_warp_tile_upper)

        @parameter
        if is_lower_required:
            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                Self.data_paths, Self.stageN
            ](1, 0)
            store_fragment_to_smem[
                Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
            ](lower_casted, c_smem_warp_tile_lower)


# =============================================================================
# Imports for IndexList
# =============================================================================
from utils.index import IndexList
from utils.static_tuple import StaticTuple
from layout.layout import coalesce, flatten


# =============================================================================
# SMemEpilogueWriter - SMEM-based epilogue writer with element-wise compute
# =============================================================================


@register_passable("trivial")
struct SMemEpilogueWriter[
    # Infer-only: deduced from c_tiles argument type
    c_type: DType,
    c_smem_layout: Layout,
    num_output_stages: Int,
    //,
    # Configuration parameters (must be explicit)
    epilogue_dtype: DType,
    BM: Int,
    BN: Int,
    MMA_M: Int,
    MMA_N: Int,
    cta_group: Int,
    num_output_warps: Int,
    c_swizzle: TensorMapSwizzle,
    transpose_c: Bool,
    is_lower_frag_required: Bool,
    num_stages: Int,
    simd_size: Int,
    stage: Int,
    rep_frag_size: Int,
    compute_lambda_fn: elementwise_compute_lambda_type,
]:
    """Write accumulator tile to SMEM and apply element-wise epilogue lambda.

    This writer handles the SMEM-based epilogue path when register_based_epilogue=False.
    Inferred from c_tiles: c_type, c_smem_layout, num_output_stages.
    Derived from layout: stageN, stage_contiguous_size.
    """

    # Derived from c_smem_layout - no need to pass as parameters
    comptime N_dim = 0 if Self.transpose_c else 1
    comptime stageN = Self.c_smem_layout.shape[Self.N_dim].value()
    comptime stage_contiguous_size = Self.c_smem_layout.shape[1].value()

    comptime swizzle = _make_swizzle[Self.c_type, Self.c_swizzle]()
    comptime swizzle_width = Self.c_swizzle.bytes() // size_of[Self.c_type]()
    comptime data_paths = 16
    comptime barrier_threads = Self.num_output_warps * WARP_SIZE
    comptime OutputSyncBarrier = WarpGroupBarrier[Self.barrier_threads]
    comptime Tile = AccumTile[Self.epilogue_dtype, Self.rep_frag_size]
    comptime CTileArray = SMemTileArrayType[
        Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
    ]

    var warp_id: UInt32
    var c_tiles: Self.CTileArray
    var M: UInt32
    var N: UInt32
    var c_row: UInt32
    var c_col: UInt32

    @always_inline
    fn __init__(
        out self,
        warp_id: UInt32,
        c_tiles: SMemTileArrayType[
            Self.c_type,
            Self.c_smem_layout,
            Self.num_output_stages,
            alignment=128,
        ],
        c_shape: Tuple[UInt32, UInt32],
        c_coord: Tuple[UInt32, UInt32],
    ):
        """Initialize the SMEM epilogue writer."""
        self.warp_id = warp_id
        self.c_tiles = c_tiles
        self.M = c_shape[0]
        self.N = c_shape[1]
        self.c_row = c_coord[0] * UInt32(Self.BM)
        self.c_col = c_coord[1] * UInt32(Self.MMA_N)

    @always_inline
    fn write_tile(self, tile: Self.Tile):
        """Write accumulator tile to SMEM and apply epilogue lambda."""
        # Double-buffer tile selection
        var c_smem_tile = self.c_tiles[Self.stage % Self.num_output_stages]

        @parameter
        if Self.transpose_c:
            self._write_transpose(tile.upper, tile.lower, c_smem_tile)
        else:
            self._write_non_transpose(tile.upper, tile.lower, c_smem_tile)

    @always_inline
    fn _write_transpose(
        self,
        upper_frag: SIMD[Self.epilogue_dtype, Self.rep_frag_size],
        lower_frag: SIMD[Self.epilogue_dtype, Self.rep_frag_size],
        c_smem_tile: SMemTileType[
            Self.c_type, Self.c_smem_layout, alignment=128
        ],
    ):
        """Transpose path: reshape tiles and apply epilogue."""

        @parameter
        if Self.is_lower_frag_required:
            # cta_group=2 path with both upper and lower fragments
            comptime tile_width = 32
            comptime smem_swblock_layout = Layout.row_major(
                Self.stageN, 2, tile_width
            )
            comptime num_swblocks = Self.stage_contiguous_size // Self.swizzle_width
            comptime smem_logical_layout = Layout(
                flatten([num_swblocks, smem_swblock_layout.shape]),
                flatten(
                    [
                        Self.stageN * Self.swizzle_width,
                        smem_swblock_layout.stride,
                    ]
                ),
            )

            var new_smem = SMemTileType[
                Self.c_type, smem_logical_layout, alignment=128
            ](c_smem_tile.ptr)
            warp_j, warp_i = divmod(Int(self.warp_id), 2)
            var _c_smem_warp_tile = new_smem.tile[
                1, Self.stageN, 1, tile_width
            ](warp_j, 0, warp_i, 0)
            var c_smem_warp_tile = _c_smem_warp_tile.reshape[
                coalesce(_c_smem_warp_tile.layout)
            ]()

            var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                Self.stageN, Self.data_paths
            ](0, 0)
            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                Self.stageN, Self.data_paths
            ](0, 1)

            warp_offset = warp_i * tile_width
            store_fragment_to_smem[Self.swizzle, Self.stageN, Self.transpose_c](
                upper_frag, c_smem_warp_tile_upper, warp_offset
            )
            warp_offset += tile_width // 2
            store_fragment_to_smem[Self.swizzle, Self.stageN, Self.transpose_c](
                lower_frag, c_smem_warp_tile_lower, warp_offset
            )

            Self.OutputSyncBarrier.sync()

            shared_memory_epilogue_transpose[
                UInt(Self.stage),
                UInt(Self.stageN),
                new_smem.dtype,
                new_smem.layout,
                Self.swizzle,
                Self.compute_lambda_fn,
                UInt(Self.num_output_warps),
                2,  # warp_dim
                Self.MMA_M,
                Self.BN,
                Self.cta_group,
            ](
                self.M,
                self.N,
                UInt(self.c_col),
                UInt(self.c_row),
                new_smem,
                UInt(warp_i),
                UInt(warp_j),
            )
        else:
            # cta_group=1 path with only upper fragment
            comptime tile_width = 16
            comptime smem_logical_layout = Layout.row_major(
                Self.stageN, 4, tile_width
            )

            var new_smem = SMemTileType[
                Self.c_type, smem_logical_layout, alignment=128
            ](c_smem_tile.ptr)
            var _c_smem_warp_tile = new_smem.tile[Self.stageN, 1, tile_width](
                0, Int(self.warp_id), 0
            )
            var c_smem_warp_tile = _c_smem_warp_tile.reshape[
                coalesce(_c_smem_warp_tile.layout)
            ]()

            var c_smem_warp_tile_upper = c_smem_warp_tile
            warp_offset = Int(self.warp_id) * tile_width
            store_fragment_to_smem[
                Self.swizzle, Int(Self.stageN), Self.transpose_c
            ](upper_frag, c_smem_warp_tile_upper, warp_offset)

            Self.OutputSyncBarrier.sync()

            shared_memory_epilogue_transpose[
                UInt(Self.stage),
                UInt(Self.stageN),
                new_smem.dtype,
                new_smem.layout,
                Self.swizzle,
                Self.compute_lambda_fn,
                UInt(Self.num_output_warps),
                1,  # warp_dim
                Self.MMA_M,
                Self.BN,
                Self.cta_group,
            ](
                self.M,
                self.N,
                UInt(self.c_col),
                UInt(self.c_row),
                new_smem,
                UInt(self.warp_id),
                UInt(0),
            )

    @always_inline
    fn _write_non_transpose(
        self,
        upper_frag: SIMD[Self.epilogue_dtype, Self.rep_frag_size],
        lower_frag: SIMD[Self.epilogue_dtype, Self.rep_frag_size],
        c_smem_tile: SMemTileType[
            Self.c_type, Self.c_smem_layout, alignment=128
        ],
    ):
        """Non-transpose path: tile per warp and apply epilogue."""
        comptime c_smem_tile_m = 32 if Self.cta_group == 2 else Self.BM // Self.num_output_warps
        var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, Self.stageN](
            Int(self.warp_id), 0
        )

        var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
            Self.data_paths, Self.stageN
        ](0, 0)
        store_fragment_to_smem[Self.swizzle, Self.stageN, Self.transpose_c](
            upper_frag, c_smem_warp_tile_upper
        )

        var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
            Self.data_paths, Self.stageN
        ](1, 0)

        @parameter
        if Self.is_lower_frag_required:
            store_fragment_to_smem[Self.swizzle, Self.stageN, Self.transpose_c](
                lower_frag, c_smem_warp_tile_lower
            )

        Self.OutputSyncBarrier.sync()

        shared_memory_epilogue[
            UInt(Self.MMA_M),
            Self.data_paths,
            UInt(Self.num_stages),
            UInt(Self.stage),
            UInt(Self.stageN),
            c_smem_warp_tile_upper.dtype,
            UInt(c_smem_tile.shape[1]()),
            UInt(Self.simd_size),
            c_smem_warp_tile_upper.layout,
            c_smem_warp_tile_lower.layout,
            Self.swizzle,
            Self.compute_lambda_fn,
            UInt(Self.num_output_warps),
        ](
            self.M,
            self.N,
            UInt(self.c_col),
            UInt(self.c_row),
            c_smem_warp_tile_upper,
            c_smem_warp_tile_lower,
        )


# =============================================================================
# Shared Memory Epilogue Functions
# =============================================================================
# These functions apply element-wise compute lambdas to data in shared memory.
# Used when register_based_epilogue=False.


@always_inline
fn shared_memory_epilogue_transpose[
    stage: UInt,
    stageN: UInt,
    c_type: DType,
    c_smem_layout: Layout,
    swizzle: Swizzle,
    compute_lambda_fn: elementwise_compute_lambda_type,
    num_output_warps: UInt,
    warp_dim: UInt,
    MMA_M: Int,
    BN: Int,
    cta_group: Int,
](
    M: UInt32,
    N: UInt32,
    c_col: UInt,
    c_row: UInt,
    c_smem: SMemTileType[c_type, c_smem_layout, alignment=128],
    warp_i: UInt,
    warp_j: UInt,
):
    """Apply element-wise epilogue to transposed shared memory tile.

    Handles the transpose_c case for SMEM-based epilogue. Supports two warp
    configurations based on warp_dim parameter.

    Template Parameters:
        stage: Current output stage index.
        stageN: Stage width in elements.
        c_type: Output data type.
        c_smem_layout: Shared memory tile layout.
        swizzle: Swizzle pattern for SMEM access.
        compute_lambda_fn: Element-wise compute function.
        num_output_warps: Number of warps participating.
        warp_dim: Warp dimension configuration (1 or 2).
        MMA_M: MMA M dimension.
        BN: Block N dimension.
        cta_group: Number of CTAs cooperating.

    Args:
        M: Output M dimension.
        N: Output N dimension.
        c_col: Base column coordinate.
        c_row: Base row coordinate.
        c_smem: Shared memory tile.
        warp_i: Warp index i.
        warp_j: Warp index j.
    """
    var c_i = c_col + stage * stageN
    var c_j = c_row
    # this function write the shared memory tile to global memory starting at
    # (c_i, c_j). When `warp_dim` is 2, the layout modes are:
    # (warp_j, stageN, warp_i, UL),
    # else, `warp_dim` is 1, the layout modes are:
    # (stageN, warp_i, U), where U denotes upper and L denotes lower.
    comptime simd_size = simd_width_of[c_type]()
    comptime alignment = align_of[SIMD[c_type, simd_size]]()
    comptime swizzle_dim = 64

    @parameter
    if warp_dim == 2:
        comptime layout_3d = Layout.row_major(2, Int(stageN), swizzle_dim)
        var rt_layout_3d = RLayout32Bits[layout_3d]()
        constrained[c_smem_layout.rank() == 4, "c_smem_layout must be 4D"]()
        comptime thread_layout = Layout.row_major(1, 8, 1, 4)
        comptime result = zipped_divide(
            upcast(c_smem_layout, simd_size), thread_layout
        )
        var rt_thread_layout = RLayout32Bits[thread_layout]()
        var lane = lane_id()
        var crd = idx2crd(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE), element_type = DType.uint32](
                Int(lane)
            ),
            rt_thread_layout.shape,
            rt_thread_layout.stride,
        )
        comptime thread_shape = IntTuple(0, UNKNOWN_VALUE, 0, UNKNOWN_VALUE)

        @parameter
        for iter_i in range(result.shape[1][3].value()):

            @parameter
            for iter_j in range(result.shape[1][1].value()):
                comptime rest_shape = IntTuple(
                    UNKNOWN_VALUE, iter_j, UNKNOWN_VALUE, iter_i
                )
                var coord = RuntimeTuple[
                    [thread_shape, rest_shape], element_type = DType.uint32
                ](
                    Int(0),
                    Int(crd[1].get_int()),
                    Int(0),
                    Int(crd[3].get_int()),
                    Int(warp_j),
                    Int(iter_j),
                    Int(warp_i),
                    Int(iter_i),
                )
                var offset = simd_size * RLayout32Bits[result]()(coord)
                var logical_crd = idx2crd(
                    RuntimeTuple[
                        IntTuple(UNKNOWN_VALUE), element_type = DType.uint32
                    ](Int(offset)),
                    rt_layout_3d.shape,
                    rt_layout_3d.stride,
                )
                var local_i: UInt32
                var local_j: UInt32

                var ci = logical_crd[0].get_int()
                var cj = logical_crd[1].get_int()
                var ck = logical_crd[2].get_int()

                @parameter
                if cta_group == 2 and MMA_M == 128:
                    # logical shared memory -> global layout Layout B:
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
                    local_i = cj + ci * BN
                    local_j = ck
                else:
                    # logical shared memory -> global layout Layout A:
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-a
                    local_i = cj
                    local_j = ci * swizzle_dim + ck

                # undo swizzle to get logical `c_smem[logical_crd]` value.
                var ptr = (
                    c_smem.ptr
                    + swizzle(cj * swizzle_dim + ck)
                    + ci * swizzle_dim * Int(stageN)
                )
                var global_i = local_i + c_i
                var global_j = local_j + c_j
                if global_i < Int(M) and global_j < Int(N):
                    var val = ptr.load[width=simd_size, alignment=alignment]()
                    var reg_val = compute_lambda_fn[alignment=alignment](
                        (Int(global_i), Int(global_j)),
                        val,
                    )
                    ptr.store[width=simd_size, alignment=alignment](reg_val)
    else:
        # Layout F: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-f
        constrained[c_smem_layout.rank() == 3, "c_smem_layout must be 3D"]()
        comptime thread_layout = Layout.row_major(min(16, Int(stageN)), 1, 2)
        comptime thread_bound = UInt(thread_layout.cosize())
        var lane = lane_id()
        if lane < thread_bound:
            comptime result = zipped_divide(
                upcast(c_smem_layout, simd_size), thread_layout
            )
            var rt_thread_layout = RLayout32Bits[thread_layout]()
            var crd = idx2crd(
                RuntimeTuple[
                    IntTuple(UNKNOWN_VALUE), element_type = DType.uint32
                ](Int(lane)),
                rt_thread_layout.shape,
                rt_thread_layout.stride,
            )
            comptime thread_shape = IntTuple(UNKNOWN_VALUE, 0, UNKNOWN_VALUE)
            comptime layout_2d = Layout.row_major(Int(stageN), swizzle_dim)
            var rt_layout_2d = RLayout32Bits[layout_2d]()

            @parameter
            for iter_i in range(result.shape[1][2].value()):

                @parameter
                for iter_j in range(result.shape[1][0].value()):
                    comptime rest_shape = IntTuple(
                        iter_j,
                        UNKNOWN_VALUE,
                        iter_i,
                    )
                    var coord = RuntimeTuple[
                        [thread_shape, rest_shape], element_type = DType.uint32
                    ](
                        Int(crd[0].get_int()),
                        Int(0),
                        Int(crd[2].get_int()),
                        Int(iter_j),
                        Int(warp_i),
                        Int(iter_i),
                    )
                    var offset = simd_size * RLayout32Bits[result]()(coord)
                    var logical_crd = idx2crd(
                        RuntimeTuple[
                            IntTuple(UNKNOWN_VALUE), element_type = DType.uint32
                        ](Int(offset)),
                        rt_layout_2d.shape,
                        rt_layout_2d.stride,
                    )

                    var local_i = logical_crd[0].get_int()
                    var local_j = logical_crd[1].get_int()

                    # undo swizzle to get logical `c_smem[logical_crd]` value.
                    var ptr = c_smem.ptr + swizzle(offset)
                    var global_i = local_i + c_i
                    var global_j = local_j + c_j
                    if global_i < Int(M) and global_j < Int(N):
                        var val = ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        var reg_val = compute_lambda_fn[alignment=alignment](
                            (Int(global_i), Int(global_j)),
                            val,
                        )
                        ptr.store[width=simd_size, alignment=alignment](reg_val)

    WarpGroupBarrier[Int(num_output_warps) * WARP_SIZE].sync()


@always_inline
fn shared_memory_epilogue[
    MMA_M: UInt,
    data_paths: UInt,
    num_stages: UInt,
    stage: UInt,
    stageN: UInt,
    c_type: DType,
    shared_n: UInt,
    simd_size: UInt,
    c_smem_upper_layout: Layout,
    c_smem_lower_layout: Layout,
    swizzle: Swizzle,
    compute_lambda_fn: elementwise_compute_lambda_type,
    num_output_warps: UInt,
](
    M: UInt32,
    N: UInt32,
    c_col: UInt,
    c_row: UInt,
    c_smem_warp_tile_upper: SMemTileType[c_type, c_smem_upper_layout, *_, **_],
    c_smem_warp_tile_lower: SMemTileType[c_type, c_smem_lower_layout, *_, **_],
):
    """Apply element-wise epilogue to non-transposed shared memory tile.

    Handles the non-transpose case for SMEM-based epilogue. Processes upper
    and lower fragments separately with proper coordinate mapping.

    Template Parameters:
        MMA_M: MMA M dimension.
        data_paths: Number of data paths (typically 16).
        num_stages: Total number of output stages.
        stage: Current output stage index.
        stageN: Stage width in elements.
        c_type: Output data type.
        shared_n: Shared memory N dimension.
        simd_size: SIMD width for vectorized access.
        c_smem_upper_layout: Layout for upper fragment tile.
        c_smem_lower_layout: Layout for lower fragment tile.
        swizzle: Swizzle pattern for SMEM access.
        compute_lambda_fn: Element-wise compute function.
        num_output_warps: Number of warps participating.

    Args:
        M: Output M dimension.
        N: Output N dimension.
        c_col: Base column coordinate.
        c_row: Base row coordinate.
        c_smem_warp_tile_upper: Upper fragment shared memory tile.
        c_smem_warp_tile_lower: Lower fragment shared memory tile.
    """
    # Here we start keeping track of the index / indices this thread is
    # responsible for in shared memory. This is represented with shared_memory_row
    # and shared_memory_column and the children of these values shared_memory_row_upper_half
    # shared_memory_row_lower_half. We also need to update the global memory column c_col by
    # stageN since we are sliding through the overall compute block.

    var staged_c_col = c_col + stage * stageN

    var warp_id = get_warp_id()
    var shared_memory_row = warp_id * 32

    var shared_memory_row_upper_half = shared_memory_row
    var shared_memory_row_lower_half = shared_memory_row + 16

    # This distribute layout allocates vectors to corresponding threads. If stageN is 32, 8 x 4 is used since each row of
    # 4 threads can access 8 elements (8 x 4 = 32). If stageN is 16 then 16 x 2 is used. Since each fragment contains 16 rows,
    # there will be 2 chunks created when using 8x4.

    comptime distribute_cols = stageN // simd_size
    comptime distribute_rows = WARP_SIZE // Int(distribute_cols)

    comptime distribute_layout = Layout.row_major(
        distribute_rows, Int(distribute_cols)
    )
    var c_smem_upper_frag = c_smem_warp_tile_upper.vectorize[
        1, Int(simd_size)
    ]().distribute[distribute_layout, swizzle=swizzle](lane_id())

    var c_smem_lower_frag = c_smem_warp_tile_lower.vectorize[
        1, Int(simd_size)
    ]().distribute[distribute_layout, swizzle=swizzle](lane_id())

    comptime fragment_size = c_smem_upper_frag.layout.size()

    var local_row, local_col = divmod(lane_id(), distribute_cols)

    var shared_memory_col = local_col * simd_size
    shared_memory_row_lower_half += local_row
    shared_memory_row_upper_half += local_row

    @parameter
    for i in range(fragment_size):
        comptime alignment = align_of[SIMD[c_type, Int(simd_size)]]()

        # these offsets are swizzled so to retrieve the corresponding gmem offset we need to remove the swizzle
        # luckily removing the swizzle is as simple as swizzling a second time
        var swz_offset_upper = (
            shared_memory_row_upper_half * shared_n + shared_memory_col
        )
        var swz_offset_lower = (
            shared_memory_row_lower_half * shared_n + shared_memory_col
        )

        var offset_upper = swizzle(Int(swz_offset_upper))
        var offset_lower = swizzle(Int(swz_offset_lower))

        var shared_upper_row: Int64
        var shared_upper_col: Int64
        var shared_lower_row: Int64
        var shared_lower_col: Int64

        # Now that we have the true index we, need to add the global tile index to find the corresponding
        # index, in gmem. However the data will be stored in tensor memory differently depending on
        # MMA_M size, we take that into account here.

        @parameter
        if MMA_M != 256:
            comptime blocked_m_128_layout = blocked_product(
                Layout.row_major(Int(data_paths * 2), Int(stageN)),
                Layout.col_major(2, 2),
                coalesce_output=True,
            )

            var upper_coord = idx2crd(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](offset_upper),
                RuntimeTuple[
                    blocked_m_128_layout.shape,
                    element_type = DType.int64,
                ](),
                RuntimeTuple[
                    blocked_m_128_layout.stride,
                    element_type = DType.int64,
                ](),
            )

            var lower_coord = idx2crd(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](offset_lower),
                RuntimeTuple[
                    blocked_m_128_layout.shape,
                    element_type = DType.int64,
                ](),
                RuntimeTuple[
                    blocked_m_128_layout.stride,
                    element_type = DType.int64,
                ](),
            )

            shared_upper_row = upper_coord[0].get_int()
            shared_lower_row = lower_coord[0].get_int()

            var section_offset_upper = upper_coord[1][1].get_int()
            var col_offset_upper = upper_coord[1][0].get_int()

            var section_offset_lower = lower_coord[1][1].get_int()
            var col_offset_lower = lower_coord[1][0].get_int()

            shared_upper_col = (
                section_offset_upper * (num_stages * stageN) + col_offset_upper
            )
            shared_lower_col = (
                section_offset_lower * (num_stages * stageN) + col_offset_lower
            )

        else:
            # can't cast to uint64 as it's not supported yet
            # this will cost us slightly in performance
            comptime fast_div = FastDiv[DType.uint32](Int(shared_n))

            shared_upper_row = (
                Scalar[DType.int](offset_upper).cast[fast_div.uint_type]()
                / fast_div
            ).cast[DType.int64]()
            shared_upper_col = offset_upper % Int(shared_n)

            shared_lower_row = (
                Scalar[DType.int](offset_lower).cast[fast_div.uint_type]()
                / fast_div
            ).cast[DType.int64]()
            shared_lower_col = offset_lower % Int(shared_n)

        # now we need to add the global tile offset
        var global_upper_row = shared_upper_row + c_row
        var global_upper_col = shared_upper_col + staged_c_col
        var global_lower_row = shared_lower_row + c_row
        var global_lower_col = shared_lower_col + staged_c_col

        if global_upper_row < Int(M) and global_upper_col < Int(N):
            var reg_val = compute_lambda_fn[alignment=alignment](
                (Int(global_upper_row), Int(global_upper_col)),
                c_smem_upper_frag[i, 0],
            )
            c_smem_upper_frag[i, 0] = reg_val

        if global_lower_row < Int(M) and global_lower_col < Int(N):
            var reg_val = compute_lambda_fn[alignment=alignment](
                (Int(global_lower_row), Int(global_lower_col)),
                c_smem_lower_frag[i, 0],
            )
            c_smem_lower_frag[i, 0] = reg_val

        # If more than one chunk is created (happens when 8x4 is used)
        # they will be spaced 8 rows away from each other

        shared_memory_row_upper_half += UInt(distribute_rows)
        shared_memory_row_lower_half += UInt(distribute_rows)

    WarpGroupBarrier[Int(num_output_warps) * WARP_SIZE].sync()
