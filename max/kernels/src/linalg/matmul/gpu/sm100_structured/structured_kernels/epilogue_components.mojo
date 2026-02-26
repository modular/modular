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

"""Low-level epilogue components for SM100 matrix multiplication.

This module provides modular building blocks for the output pipeline:

1. **store_fragment_to_smem**: Register to shared memory via st.matrix instructions
2. **TMEMToSMemWriter**: Write TMEM accumulators to shared memory
3. **TMAStoreExecutor**: Execute TMA stores with proper SMEM tiling
4. **EpilogueApplier**: Apply element-wise operations on fragments

The SM100 epilogue pipeline flows as:
    TMEM (accumulators) → Registers → SMEM → GMEM (via TMA)
"""

from sys import align_of, simd_width_of

from gpu import WARP_SIZE, lane_id, warp_id
from gpu.memory import fence_async_view_proxy
from gpu.host.nvidia.tma import TensorMapSwizzle
from .barriers import WarpGroupBarrier
from layout import Layout, RuntimeLayout, UNKNOWN_VALUE, RuntimeTuple
from layout.int_tuple import IntTuple
from layout._layout import Layout as InternalLayout, TensorLayout, row_major
from layout.layout import blocked_product, zipped_divide, upcast
from layout.runtime_tuple import idx2crd, crd2idx as rt_crd2idx
from layout.swizzle import Swizzle, make_swizzle as _make_swizzle
from layout._tile_tensor import TileTensor
from layout.tma_async import TMATensorTile
from linalg.structuring import SMemTileArray, SMemTile
from linalg.utils import elementwise_compute_lambda_type
from utils.fast_div import FastDiv
from utils.static_tuple import StaticTuple

# TileTensor-based types - use TileSMemTile to avoid name collision with old SMemTile
from .tile_types import (
    SMemTile as TileSMemTile,
    SMemTileStride,
    SMemTileShape,
    SMemTileArray2D,
    SMemTileArray2DRowMajor,
    SMemTileArrayWithLayout,
)


# =============================================================================
# tma_wait_pipelined - Pipelined TMA wait helper
# =============================================================================


@always_inline
fn tma_wait_pipelined[
    c_type: DType,
    c_layout: Layout,
    c_desc_layout: Layout,
    is_last_stage: Bool,
](c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout]):
    """Wait for TMA stores with pipelining.

    For SM100 output pipeline:
    - Non-last stages: Keep 1 store in flight for pipelining
    - Last stage: Wait for all stores to complete
    """

    comptime if is_last_stage:
        c_tma_op.wait_group[0]()  # Wait for all stores
    else:
        c_tma_op.wait_group[1]()  # Keep 1 store in flight


# =============================================================================
# AccumTile - Accumulator tile (upper + lower fragments) for writing
# =============================================================================


struct AccumTile[dtype: DType, size: Int](TrivialRegisterPassable):
    """Upper + lower TMEM fragments (16 rows each) for SM100 output."""

    var upper: SIMD[Self.dtype, Self.size]
    var lower: SIMD[Self.dtype, Self.size]

    @always_inline
    fn __init__(
        out self,
        upper: SIMD[Self.dtype, Self.size],
        lower: SIMD[Self.dtype, Self.size],
    ):
        self.upper = upper
        self.lower = lower


# =============================================================================
# AccumBarrier - Accumulator pipeline barrier helper
# =============================================================================


struct AccumBarrier[cta_group: Int](TrivialRegisterPassable):
    """Pipeline barrier helper for single-CTA vs 2-CTA arrival patterns."""

    @staticmethod
    @always_inline
    fn arrive(pipeline: ProducerConsumerPipeline, stage: UInt32):
        """Signal accumulator arrival on pipeline barrier."""

        comptime if Self.cta_group == 1:
            from gpu.sync import mbarrier_arrive

            _ = mbarrier_arrive(pipeline.consumer_mbar(stage))
        else:
            from gpu.sync import umma_arrive_leader_cta

            umma_arrive_leader_cta(pipeline.consumer_mbar(stage))


# Import for AccumBarrier
from .pipeline import ProducerConsumerPipeline


# =============================================================================
# store_fragment_to_smem - Static helper for st.matrix operations
# =============================================================================


@always_inline
fn store_fragment_to_smem[
    swizzle: Swizzle,
    stageN: Int,
    transpose_c: Bool = False,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](vec: SIMD, dst: SMemTile, warp_offset: UInt32 = 0):
    """Store fragment to SMEM via st.matrix instruction."""
    from gpu.compute.mma import st_matrix
    from memory import bitcast

    comptime c_type = dst.dtype
    comptime stsmx_row_size = 32 // size_of[
        c_type
    ]() if stageN % 16 == 0 else 16 // size_of[c_type]()
    comptime stsmx_lane_size = 16 // size_of[c_type]()
    comptime stmtx_simd_width = 4 if stageN % 16 == 0 else 2
    comptime stride0 = dst.layout.stride[0].value()
    comptime stride1 = dst.layout.stride[1].value()
    comptime shape0 = dst.layout.shape[
        1
    ].value() if not transpose_c else dst.layout.shape[0].value()
    comptime stsmx_tile_offset = (
        stride0 if transpose_c else stride1
    ) * stsmx_row_size

    var lane_id = lane_id()
    var stsm_lane_offset: UInt32

    comptime if transpose_c:
        # Use new Layout directly instead of RuntimeLayout wrapper
        from layout._layout import Layout as NewLayout
        from layout._coord import Coord, Idx

        comptime trans_layout = NewLayout(
            Coord(Idx[8](), Idx[2](), Idx[2]()),
            Coord(Idx[stride0](), Idx[8 * stride1](), Idx[8 * stride0]()),
        )
        stsm_lane_offset = UInt32(trans_layout(Idx(Int(lane_id))))
    else:
        stsm_lane_offset = (
            UInt32(lane_id & 15) * UInt32(stride0) + UInt32(lane_id >> 4) * 8
        )

    @always_inline
    fn slice[offset: Int, size: Int](v: SIMD) -> SIMD[v.dtype, size]:
        var tmp = SIMD[v.dtype, size]()

        comptime for i in range(size):
            tmp[i] = v[i + offset]
        return tmp

    comptime for i in range(shape0 // stsmx_row_size):
        comptime n_offset = i * stsmx_tile_offset
        var offset: UInt32

        comptime if transpose_c:
            offset = (
                swizzle(stsm_lane_offset + UInt32(n_offset) + warp_offset)
                - warp_offset
            )
        else:
            offset = swizzle(stsm_lane_offset + UInt32(n_offset))

        var v = slice[i * stsmx_lane_size, 2 * stmtx_simd_width](vec).cast[
            c_type
        ]()

        st_matrix[simd_width=stmtx_simd_width, transpose=transpose_c](
            dst.ptr + offset, bitcast[DType.float32, stmtx_simd_width](v)
        )


@always_inline
fn store_fragment_to_smem[
    swizzle: Swizzle,
    stageN: Int,
    transpose_c: Bool = False,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    vec: SIMD,
    dst: TileTensor[address_space = AddressSpace.SHARED, ...],
    warp_offset: UInt32 = 0,
):
    """Store fragment to SMEM via st.matrix."""
    from gpu.compute.mma import st_matrix
    from memory import bitcast
    from layout._coord import Coord, Idx

    comptime c_type = dst.dtype
    comptime stsmx_row_size = 32 // size_of[
        c_type
    ]() if stageN % 16 == 0 else 16 // size_of[c_type]()
    comptime stsmx_lane_size = 16 // size_of[c_type]()
    comptime stmtx_simd_width = 4 if stageN % 16 == 0 else 2
    comptime stride0 = dst.static_stride[0]
    comptime stride1 = dst.static_stride[1]
    comptime shape0 = dst.static_shape[
        1
    ] if not transpose_c else dst.static_shape[0]
    comptime stsmx_tile_offset = (
        stride0 if transpose_c else stride1
    ) * stsmx_row_size

    var lane_id = lane_id()
    var stsm_lane_offset: UInt32

    comptime if transpose_c:
        comptime trans_layout = InternalLayout(
            Coord(Idx[8](), Idx[2](), Idx[2]()),
            Coord(Idx[stride0](), Idx[8 * stride1](), Idx[8 * stride0]()),
        )
        stsm_lane_offset = UInt32(trans_layout(Idx(Int(lane_id))))
    else:
        stsm_lane_offset = (
            UInt32(lane_id & 15) * UInt32(stride0) + UInt32(lane_id >> 4) * 8
        )

    @always_inline
    fn slice[offset: Int, size: Int](v: SIMD) -> SIMD[v.dtype, size]:
        var tmp = SIMD[v.dtype, size]()

        comptime for i in range(size):
            tmp[i] = v[i + offset]
        return tmp

    comptime for i in range(shape0 // stsmx_row_size):
        comptime n_offset = i * stsmx_tile_offset
        var offset: UInt32

        comptime if transpose_c:
            offset = (
                swizzle(stsm_lane_offset + UInt32(n_offset) + warp_offset)
                - warp_offset
            )
        else:
            offset = swizzle(stsm_lane_offset + UInt32(n_offset))

        var v = slice[i * stsmx_lane_size, 2 * stmtx_simd_width](vec).cast[
            c_type
        ]()

        st_matrix[simd_width=stmtx_simd_width, transpose=transpose_c](
            dst.ptr.mut_cast[True]() + offset,
            bitcast[DType.float32, stmtx_simd_width](v),
        )


# =============================================================================
# EpilogueConfig - Configuration for epilogue operations
# =============================================================================


struct EpilogueConfig[
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    cta_group: Int,
    transpose_c: Bool,
](TrivialRegisterPassable):
    """Computed epilogue parameters based on MMA and CTA configuration."""

    # Lower fragment needed except for cta_group=1, MMA_M=64
    comptime is_lower_frag_required = not (
        Self.cta_group == 1 and Self.MMA_M == 64
    )

    comptime cg2_num_stages = Self.MMA_N // Self.stageN if Self.MMA_M == 256 else Self.MMA_N // Self.stageN // 2
    comptime cg1_num_stages = Self.MMA_N // Self.stageN
    comptime num_stages = Self.cg2_num_stages if Self.cta_group == 2 else Self.cg1_num_stages

    comptime data_paths = 16
    comptime bits = 256
    comptime fragment_size = (Self.data_paths * (Self.bits // 32)) // 32


# =============================================================================
# TMAStoreCoords - Compute coordinates for TMA store operations
# =============================================================================


struct TMAStoreCoords[
    BM: Int,
    BN: Int,
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    cta_group: Int,
    c_smem_shape0: Int,
    stage: Int,
    batched: Bool = False,
](TrivialRegisterPassable):
    """TMA store coordinates and warp election for SM100 epilogue.

    When batched=True, includes a batch coordinate for 3D TMA stores.
    """

    comptime CG2_TMA_BM = Self.c_smem_shape0 if Self.MMA_M == 256 else Self.BM
    comptime CG1_TMA_BM = Self.c_smem_shape0
    comptime TMA_BM = Self.CG2_TMA_BM if Self.cta_group == 2 else Self.CG1_TMA_BM
    comptime stage_n_offset = Self.stage * Self.stageN

    var coord_m: UInt
    var coord_n: UInt
    var coord_b: UInt  # Batch coordinate (only used when batched=True)
    var elect_one_warp: Bool
    var c_smem_coord_m: UInt

    @always_inline
    fn __init__(out self, c_coord: Tuple[UInt32, UInt32], warp_id: UInt32):
        """Compute TMA store coordinates from 2D tile coords and warp ID."""
        # Warp election
        var cg2_elect = warp_id == 0 if Self.MMA_M == 256 else warp_id % 2 == 0
        var cg1_elect = warp_id == 0
        self.elect_one_warp = cg2_elect if Self.cta_group == 2 else cg1_elect

        # N coordinate
        var n_base = c_coord[1] * UInt32(Self.MMA_N) + UInt32(
            Self.stage_n_offset
        )
        var n_mma128 = n_base + UInt32(Self.BN * Int(warp_id // 2))
        var cg2_n = n_base if Self.MMA_M == 256 else n_mma128
        self.coord_n = UInt(cg2_n if Self.cta_group == 2 else n_base)

        # M coordinate
        self.coord_m = UInt(c_coord[0]) * UInt(Self.BM)

        # Batch coordinate (default 0 for 2D)
        self.coord_b = UInt(0)

        # SMEM tile offset
        var cg2_smem_m: UInt

        comptime if Self.MMA_M == 256:
            cg2_smem_m = 0
        else:
            cg2_smem_m = UInt(warp_id // 2)
        self.c_smem_coord_m = cg2_smem_m if Self.cta_group == 2 else UInt(0)

    @always_inline
    fn __init__(
        out self, c_coord: Tuple[UInt32, UInt32, UInt32], warp_id: UInt32
    ):
        """Compute TMA store coordinates from 3D tile coords and warp ID."""
        # Warp election
        var cg2_elect = warp_id == 0 if Self.MMA_M == 256 else warp_id % 2 == 0
        var cg1_elect = warp_id == 0
        self.elect_one_warp = cg2_elect if Self.cta_group == 2 else cg1_elect

        # N coordinate
        var n_base = c_coord[1] * UInt32(Self.MMA_N) + UInt32(
            Self.stage_n_offset
        )
        var n_mma128 = n_base + UInt32(Self.BN * Int(warp_id // 2))
        var cg2_n = n_base if Self.MMA_M == 256 else n_mma128
        self.coord_n = UInt(cg2_n if Self.cta_group == 2 else n_base)

        # M coordinate
        self.coord_m = UInt(c_coord[0]) * UInt(Self.BM)

        # Batch coordinate
        self.coord_b = UInt(c_coord[2])

        # SMEM tile offset
        var cg2_smem_m: UInt

        comptime if Self.MMA_M == 256:
            cg2_smem_m = 0
        else:
            cg2_smem_m = UInt(warp_id // 2)
        self.c_smem_coord_m = cg2_smem_m if Self.cta_group == 2 else UInt(0)


# =============================================================================
# TMAStoreExecutor - Execute TMA stores with proper SMEM tiling
# =============================================================================


struct TMAStoreExecutor[
    c_type: DType,
    c_smem_dim0: Int,  # Replaces c_smem_layout.shape[0]
    c_smem_dim1: Int,  # Replaces c_smem_layout.shape[1]
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
    batched: Bool = False,
](TrivialRegisterPassable):
    """Execute TMA store from SMEM to GMEM with proper tiling.

    Handles 3 paths: transpose+cta_group2+MMA128, transpose+other, non-transpose.
    When batched=True, uses 3D coordinates (M, N, Batch) for TMA stores.
    """

    # Create layout from dimensions (eliminates old Layout from interface)
    comptime c_smem_layout = Layout.row_major(
        Self.c_smem_dim0, Self.c_smem_dim1
    )

    comptime swizzle_width = Self.c_swizzle.bytes() // size_of[Self.c_type]()
    comptime num_c_smem_tiles = 128 // Self.swizzle_width // (
        1 if Self.is_lower_frag_required else 2
    )
    comptime c_smem_shape0 = Self.c_smem_dim0  # Now direct access
    comptime CG2_TMA_BM = Self.c_smem_shape0 if Self.MMA_M == 256 else Self.BM
    comptime CG1_TMA_BM = Self.c_smem_shape0
    comptime TMA_BM = Self.CG2_TMA_BM if Self.cta_group == 2 else Self.CG1_TMA_BM

    @staticmethod
    @always_inline
    fn execute[
        c_layout: Layout,
        c_desc_layout: Layout,
    ](
        c_smem_tile: SMemTile[Self.c_type, Self.c_smem_layout, alignment=128],
        store_coords: TMAStoreCoords[
            Self.BM,
            Self.BN,
            Self.MMA_M,
            Self.MMA_N,
            Self.stageN,
            Self.cta_group,
            Self.c_smem_shape0,
            _,
            Self.batched,
        ],
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Execute TMA store (LayoutTensor path)."""
        if store_coords.elect_one_warp and lane == 0:
            fence_async_view_proxy()

            comptime if Self.transpose_c:
                Self._store_transpose_lt[c_layout, c_desc_layout](
                    c_smem_tile, store_coords, c_tma_op, warp_id
                )
            else:
                Self._store_non_transpose(c_smem_tile, store_coords, c_tma_op)

            c_tma_op.commit_group()

    @staticmethod
    @always_inline
    fn _store_transpose_lt[
        c_layout: Layout,
        c_desc_layout: Layout,
    ](
        c_smem_tile: SMemTile[Self.c_type, Self.c_smem_layout, alignment=128],
        store_coords: TMAStoreCoords[
            Self.BM,
            Self.BN,
            Self.MMA_M,
            Self.MMA_N,
            Self.stageN,
            Self.cta_group,
            Self.c_smem_shape0,
            _,
            Self.batched,
        ],
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
        warp_id: UInt32,
    ):
        """Handle transpose_c TMA store paths."""

        comptime if Self.cta_group == 2 and Self.MMA_M == 128:
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

            comptime if Self.batched:
                c_tma_op.async_store(
                    c_smem_split,
                    StaticTuple[UInt32, 3](
                        UInt32(store_coords.coord_m),
                        UInt32(store_coords.coord_n),
                        UInt32(store_coords.coord_b),
                    ),
                )
            else:
                c_tma_op.async_store(
                    c_smem_split,
                    (store_coords.coord_m, store_coords.coord_n),
                )
        else:
            # Path B: Other transpose cases - loop over swizzle tiles
            comptime for i in range(Self.num_c_smem_tiles):
                var c_smem_warp_tile = c_smem_tile.tile[
                    Self.stageN
                    * Self.swizzle_width
                    // Self.stage_contiguous_size,
                    Self.stage_contiguous_size,
                ](i, 0).reshape[
                    Layout.row_major(Self.stageN, Self.swizzle_width)
                ]()

                comptime if Self.batched:
                    c_tma_op.async_store(
                        c_smem_warp_tile,
                        StaticTuple[UInt32, 3](
                            UInt32(
                                store_coords.coord_m
                                + UInt(i * Self.swizzle_width)
                            ),
                            UInt32(store_coords.coord_n),
                            UInt32(store_coords.coord_b),
                        ),
                    )
                else:
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
        //,
    ](
        c_smem_tile: SMemTile[Self.c_type, Self.c_smem_layout, alignment=128],
        store_coords: TMAStoreCoords[
            Self.BM,
            Self.BN,
            Self.MMA_M,
            Self.MMA_N,
            Self.stageN,
            Self.cta_group,
            Self.c_smem_shape0,
            _,
            Self.batched,
        ],
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
    ):
        """Handle non-transpose TMA store path."""
        # Path C: Simple tile selection by TMA_BM
        # Note: coords are (coord_n, coord_m) - swapped for non-transpose!
        var c_smem_split = c_smem_tile.tile[Self.TMA_BM, Self.stageN](
            Int(store_coords.c_smem_coord_m), 0
        )

        comptime if Self.batched:
            c_tma_op.async_store(
                c_smem_split,
                StaticTuple[UInt32, 3](
                    UInt32(store_coords.coord_n),
                    UInt32(store_coords.coord_m),
                    UInt32(store_coords.coord_b),
                ),
            )
        else:
            c_tma_op.async_store(
                c_smem_split,
                (store_coords.coord_n, store_coords.coord_m),
            )

    @staticmethod
    @always_inline
    fn execute[
        c_layout: Layout,
        c_desc_layout: Layout,
    ](
        c_smem_tile: TileTensor[address_space = AddressSpace.SHARED, ...],
        store_coords: TMAStoreCoords[
            Self.BM,
            Self.BN,
            Self.MMA_M,
            Self.MMA_N,
            Self.stageN,
            Self.cta_group,
            Self.c_smem_shape0,
            _,
            Self.batched,
        ],
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Execute TMA store."""
        if store_coords.elect_one_warp and lane == 0:
            fence_async_view_proxy()

            comptime if Self.transpose_c:
                Self._store_transpose[c_layout, c_desc_layout](
                    c_smem_tile, store_coords, c_tma_op, warp_id
                )
            else:
                # Non-transpose: wrap as SMemTile for _store_non_transpose
                from memory import LegacyUnsafePointer

                comptime SMemPtrType = LegacyUnsafePointer[
                    Scalar[Self.c_type],
                    address_space = AddressSpace.SHARED,
                    origin=MutAnyOrigin,
                ]
                var c_lt = SMemTile[
                    Self.c_type, Self.c_smem_layout, alignment=128
                ](rebind[SMemPtrType](c_smem_tile.ptr.mut_cast[True]()))
                Self._store_non_transpose(c_lt, store_coords, c_tma_op)

            c_tma_op.commit_group()

    @staticmethod
    @always_inline
    fn _store_transpose[
        c_layout: Layout,
        c_desc_layout: Layout,
    ](
        c_smem_tile: TileTensor[address_space = AddressSpace.SHARED, ...],
        store_coords: TMAStoreCoords[
            Self.BM,
            Self.BN,
            Self.MMA_M,
            Self.MMA_N,
            Self.stageN,
            Self.cta_group,
            Self.c_smem_shape0,
            _,
            Self.batched,
        ],
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
        warp_id: UInt32,
    ):
        """Transpose TMA store using reshape."""
        from layout._coord import Coord, Idx

        # Convert to LayoutTensor at TMA async_store boundary
        from memory import LegacyUnsafePointer

        comptime if Self.cta_group == 2 and Self.MMA_M == 128:
            # Path A: reshape to (2*stageN, sc_size//2), tile by warp
            comptime reshaped = row_major[
                2 * Self.stageN, Self.stage_contiguous_size // 2
            ]()
            var c_reshaped = c_smem_tile.reshape(reshaped)
            var c_split = c_reshaped.tile[
                Self.stageN, Self.stage_contiguous_size // 2
            ](Coord(Idx(Int(warp_id // 2)), Idx(0)))

            # Convert to LayoutTensor for TMA async_store
            comptime split_layout = Layout.row_major(
                Self.stageN, Self.stage_contiguous_size // 2
            )
            comptime SMemPtrType = LegacyUnsafePointer[
                Scalar[Self.c_type],
                address_space = AddressSpace.SHARED,
                origin=MutAnyOrigin,
            ]
            var c_split_lt = SMemTile[Self.c_type, split_layout, alignment=128](
                rebind[SMemPtrType](c_split.ptr.mut_cast[True]())
            )

            comptime if Self.batched:
                c_tma_op.async_store(
                    c_split_lt,
                    StaticTuple[UInt32, 3](
                        UInt32(store_coords.coord_m),
                        UInt32(store_coords.coord_n),
                        UInt32(store_coords.coord_b),
                    ),
                )
            else:
                c_tma_op.async_store(
                    c_split_lt,
                    (store_coords.coord_m, store_coords.coord_n),
                )
        else:
            # Path B: loop over swizzle tiles
            comptime tile_dim0 = (
                Self.stageN * Self.swizzle_width // Self.stage_contiguous_size
            )

            comptime for i in range(Self.num_c_smem_tiles):
                var tiled = c_smem_tile.tile[
                    tile_dim0, Self.stage_contiguous_size
                ](Coord(Idx(i), Idx(0)))

                comptime reshaped = row_major[Self.stageN, Self.swizzle_width]()
                var c_warp_tile = tiled.reshape(reshaped)

                # Convert to LayoutTensor for TMA async_store
                comptime warp_layout = Layout.row_major(
                    Self.stageN, Self.swizzle_width
                )
                comptime SMemPtrType = LegacyUnsafePointer[
                    Scalar[Self.c_type],
                    address_space = AddressSpace.SHARED,
                    origin=MutAnyOrigin,
                ]
                var c_warp_lt = SMemTile[
                    Self.c_type, warp_layout, alignment=128
                ](rebind[SMemPtrType](c_warp_tile.ptr.mut_cast[True]()))

                comptime if Self.batched:
                    c_tma_op.async_store(
                        c_warp_lt,
                        StaticTuple[UInt32, 3](
                            UInt32(
                                store_coords.coord_m
                                + UInt(i * Self.swizzle_width)
                            ),
                            UInt32(store_coords.coord_n),
                            UInt32(store_coords.coord_b),
                        ),
                    )
                else:
                    c_tma_op.async_store(
                        c_warp_lt,
                        (
                            store_coords.coord_m + UInt(i * Self.swizzle_width),
                            store_coords.coord_n,
                        ),
                    )


# =============================================================================
# FragmentCoords - Coordinate tracking for fragment elements
# =============================================================================


struct FragmentCoords[stageN: Int, repeats: Int](TrivialRegisterPassable):
    """Fragment element coordinates for tcgen05 16x256b matrix layout."""

    comptime load_width = 2
    comptime threads_per_row = Self.stageN // Self.repeats // Self.load_width

    var top_upper: StaticTuple[UInt32, 2]
    var bottom_upper: StaticTuple[UInt32, 2]
    var top_lower: StaticTuple[UInt32, 2]
    var bottom_lower: StaticTuple[UInt32, 2]

    @always_inline
    fn __init__(out self, lane_id: UInt32):
        """Compute (row, col) for each fragment position from lane ID."""
        var row = lane_id // UInt32(Self.threads_per_row)
        var col = (lane_id % UInt32(Self.threads_per_row)) * UInt32(
            Self.load_width
        )

        self.top_upper = StaticTuple[UInt32, 2](row, col)
        self.bottom_upper = StaticTuple[UInt32, 2](row + 8, col)
        self.top_lower = StaticTuple[UInt32, 2](row + 16, col)
        self.bottom_lower = StaticTuple[UInt32, 2](row + 24, col)


# =============================================================================
# EpilogueApplier - Apply element-wise operations on fragments
# =============================================================================


struct EpilogueApplier[
    MMA_M: Int,
    stageN: Int,
    num_stages: Int,
    repeats: Int,
    cta_group: Int,
    transpose_c: Bool,
](TrivialRegisterPassable):
    """Apply element-wise epilogue lambda to register fragments."""

    comptime Coords = FragmentCoords[Self.stageN, Self.repeats]

    var coords: Self.Coords
    var warp_id: UInt32
    var lane_id: UInt32

    @always_inline
    fn __init__(out self, warp_id: UInt32, lane_id: UInt32):
        self.coords = Self.Coords(lane_id)
        self.warp_id = warp_id
        self.lane_id = lane_id

    @always_inline
    fn compute_staged_coords(
        self, stage: UInt32, c_row: UInt32, c_col: UInt32
    ) -> Tuple[UInt32, UInt32]:
        """Compute global coords with warp and stage offsets (layout-dependent).
        """
        var staged_col = c_col + stage * UInt32(Self.stageN)
        var staged_row = c_row

        comptime if Self.MMA_M == 256 or (
            Self.MMA_M == 128 and Self.cta_group == 1
        ):
            staged_row += self.warp_id * 32  # Layout A/D
        elif Self.MMA_M == 64 and Self.cta_group == 1:
            staged_row += self.warp_id * 16  # Layout F
        else:
            staged_row += (self.warp_id % 2) * 32  # Layout B
            staged_col += (self.warp_id // 2) * UInt32(
                Self.num_stages * Self.stageN
            )

        return (staged_row, staged_col)

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
        """Apply epilogue lambda to fragment elements with global coords."""
        var top = self.coords.top_upper if is_upper else self.coords.top_lower
        var bot = (
            self.coords.bottom_upper if is_upper else self.coords.bottom_lower
        )

        comptime for rep in range(Self.repeats):
            comptime inc = rep * 8
            comptime offset = rep * 4

            var top_row = staged_row + top[0]
            var top_col = staged_col + top[1] + UInt32(inc)
            var bot_row = staged_row + bot[0]
            var bot_col = staged_col + bot[1] + UInt32(inc)

            var elem0 = frag[offset]
            var elem1 = frag[offset + 1]
            var elem2 = frag[offset + 2]
            var elem3 = frag[offset + 3]

            comptime if Self.transpose_c:
                elem0 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(top_col), Int(top_row)), elem0
                )
                elem1 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(top_col + 1), Int(top_row)), elem1
                )
                elem2 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(bot_col), Int(bot_row)), elem2
                )
                elem3 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(bot_col + 1), Int(bot_row)), elem3
                )
            else:
                elem0 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(top_row), Int(top_col)), elem0
                )
                elem1 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(top_row), Int(top_col + 1)), elem1
                )
                elem2 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(bot_row), Int(bot_col)), elem2
                )
                elem3 = compute_lambda_fn[epilogue_dtype, 1](
                    IndexList[2](Int(bot_row), Int(bot_col + 1)), elem3
                )

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
        """Apply epilogue to both fragments (main entry point)."""
        var staged_row, staged_col = self.compute_staged_coords(
            stage, c_row, c_col
        )

        self.apply_to_fragment[epilogue_dtype, frag_size, compute_lambda_fn](
            upper_frag, staged_row, staged_col, is_upper=True
        )

        comptime if is_lower_frag_required:
            self.apply_to_fragment[
                epilogue_dtype, frag_size, compute_lambda_fn
            ](lower_frag, staged_row, staged_col, is_upper=False)

        return (upper_frag, lower_frag)

    # =========================================================================
    # Residual Add - Load C from SMEM and add beta*C to fragment registers
    # =========================================================================

    @always_inline
    fn add_residual_to_fragment[
        epilogue_dtype: DType,
        frag_size: Int,
        c_type: DType,
        c_smem_stride: Int,
        swizzle: Swizzle,
    ](
        self,
        mut frag: SIMD[epilogue_dtype, frag_size],
        local_row: UInt32,
        local_col: UInt32,
        is_upper: Bool,
        src_ptr: UnsafePointer[
            Scalar[c_type], address_space = AddressSpace.SHARED
        ],
        beta: Scalar[epilogue_dtype],
    ):
        """Add beta * C to fragment elements by loading C from swizzled SMEM.

        Uses the same per-lane coordinate mapping as apply_to_fragment, but
        instead of applying a lambda, loads source C values from SMEM at the
        matching swizzled addresses and adds beta * C to each element.

        Args:
            frag: Fragment register values to modify in-place.
            local_row: Tile-local row offset (warp offset within tile).
            local_col: Tile-local column offset (stage offset within tile).
            is_upper: Whether this is the upper (rows 0-15) or lower (16-31)
                fragment half.
            src_ptr: Pointer to source C SMEM tile (same TMA swizzle as output).
            beta: Residual scale factor.
        """
        from gpu.memory import AddressSpace

        var top = self.coords.top_upper if is_upper else self.coords.top_lower
        var bot = (
            self.coords.bottom_upper if is_upper else self.coords.bottom_lower
        )

        comptime for rep in range(Self.repeats):
            comptime inc = rep * 8
            comptime offset = rep * 4

            var top_row = local_row + top[0]
            var top_col = local_col + top[1] + UInt32(inc)
            var bot_row = local_row + bot[0]
            var bot_col = local_col + bot[1] + UInt32(inc)

            # Load C from swizzled SMEM at matching fragment coordinates
            var c0 = src_ptr.load(
                Int(swizzle(top_row * UInt32(c_smem_stride) + top_col))
            ).cast[epilogue_dtype]()
            var c1 = src_ptr.load(
                Int(swizzle(top_row * UInt32(c_smem_stride) + top_col + 1))
            ).cast[epilogue_dtype]()
            var c2 = src_ptr.load(
                Int(swizzle(bot_row * UInt32(c_smem_stride) + bot_col))
            ).cast[epilogue_dtype]()
            var c3 = src_ptr.load(
                Int(swizzle(bot_row * UInt32(c_smem_stride) + bot_col + 1))
            ).cast[epilogue_dtype]()

            # FMA: frag += beta * C
            frag[offset] += beta * c0
            frag[offset + 1] += beta * c1
            frag[offset + 2] += beta * c2
            frag[offset + 3] += beta * c3

    @always_inline
    fn add_residual_to_both_fragments[
        epilogue_dtype: DType,
        frag_size: Int,
        is_lower_frag_required: Bool,
        c_type: DType,
        c_smem_stride: Int,
        swizzle: Swizzle,
    ](
        self,
        mut upper_frag: SIMD[epilogue_dtype, frag_size],
        mut lower_frag: SIMD[epilogue_dtype, frag_size],
        stage: UInt32,
        src_ptr: UnsafePointer[
            Scalar[c_type], address_space = AddressSpace.SHARED
        ],
        beta: Scalar[epilogue_dtype],
    ) -> Tuple[
        SIMD[epilogue_dtype, frag_size], SIMD[epilogue_dtype, frag_size]
    ]:
        """Add beta * C to both fragment halves from swizzled SMEM.

        Computes tile-local coordinates from stage and warp ID, then loads
        source C from SMEM and adds beta * C to each fragment element.

        Args:
            upper_frag: Upper fragment (rows 0-15 within warp tile).
            lower_frag: Lower fragment (rows 16-31 within warp tile).
            stage: Output stage index (for column offset computation).
            src_ptr: Pointer to source C SMEM tile.
            beta: Residual scale factor.

        Returns:
            Updated (upper_frag, lower_frag) tuple.
        """
        # Tile-local coords: pass (0, 0) as origin since we're indexing
        # within the SMEM tile, not in global coordinates.
        var local_row, local_col = self.compute_staged_coords(stage, 0, 0)

        self.add_residual_to_fragment[
            epilogue_dtype, frag_size, c_type, c_smem_stride, swizzle
        ](upper_frag, local_row, local_col, True, src_ptr, beta)

        comptime if is_lower_frag_required:
            self.add_residual_to_fragment[
                epilogue_dtype, frag_size, c_type, c_smem_stride, swizzle
            ](lower_frag, local_row, local_col, False, src_ptr, beta)

        return (upper_frag, lower_frag)


# =============================================================================
# TMEMToSMemWriter - Write TMEM accumulators to shared memory (SM100-specific)
# =============================================================================


struct TMEMToSMemWriter[
    c_type: DType,
    accum_type: DType,
    c_smem_dim0: Int,  # Replaces c_smem_layout.shape[0]
    c_smem_dim1: Int,  # Replaces c_smem_layout.shape[1]
    BM: Int,
    BN: Int,
    MMA_M: Int,
    MMA_N: Int,
    stageN: Int,
    cta_group: Int,
    num_output_warps: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_c: Bool = False,
](TrivialRegisterPassable):
    """Write TMEM accumulators to SMEM via st.matrix (SM100-specific)."""

    # Create internal layout from dimensions
    comptime c_smem_layout = Layout.row_major(
        Self.c_smem_dim0, Self.c_smem_dim1
    )

    comptime Config = EpilogueConfig[
        Self.MMA_M, Self.MMA_N, Self.stageN, Self.cta_group, Self.transpose_c
    ]

    comptime swizzle_width = Self.c_swizzle.bytes() // size_of[Self.c_type]()
    comptime stage_contiguous_size = Self.c_smem_dim1  # Now direct access
    comptime data_paths = 16
    comptime swizzle = _make_swizzle[Self.c_type, Self.c_swizzle]()

    var warp_id: UInt32
    var lane_id: UInt32

    @always_inline
    fn __init__(out self, warp_id: UInt32, lane_id: UInt32):
        self.warp_id = warp_id
        self.lane_id = lane_id

    @always_inline
    fn write_fragments[
        repeat: Int
    ](
        self,
        upper_frag: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        lower_frag: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        c_smem_tile: SMemTile[Self.c_type, Self.c_smem_layout, alignment=128],
    ):
        """Write pre-loaded fragments to SMEM (LayoutTensor path)."""
        comptime is_lower_required = Self.Config.is_lower_frag_required

        comptime if Self.transpose_c:
            self._write_transpose_lt[repeat, is_lower_required](
                upper_frag, lower_frag, c_smem_tile
            )
        else:
            self._write_non_transpose_lt[repeat, is_lower_required](
                upper_frag, lower_frag, c_smem_tile
            )

    @always_inline
    fn _write_transpose_lt[
        repeat: Int, is_lower_required: Bool
    ](
        self,
        upper_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        lower_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        c_smem_tile: SMemTile[Self.c_type, Self.c_smem_layout, alignment=128],
    ):
        """Transposed output (LayoutTensor path).

        Two paths based on swizzle width:
        - Large swizzle (SWIZZLE_128B): multiple warps share a swizzle block,
          requiring warp_offset for correct swizzle/deswizzle addressing.
        - Small swizzle (SWIZZLE_32B): each warp fragment fits within its own
          swizzle block, using simple tiles_per_frag tiling.
        """
        # SWIZZLE_128B: swizzle_width=64 > data_paths=16 → True
        # SWIZZLE_32B:  swizzle_width=16 == data_paths=16 → False
        comptime _large_swizzle = Self.swizzle_width > Self.data_paths

        comptime if _large_swizzle:
            comptime if is_lower_required:
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

                var new_smem = SMemTile[
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
                comptime tile_width = 16
                comptime smem_logical_layout = Layout.row_major(
                    Self.stageN, 4, tile_width
                )

                var new_smem = SMemTile[
                    Self.c_type,
                    smem_logical_layout,
                    alignment = c_smem_tile.alignment,
                ](c_smem_tile.ptr)
                var _c_smem_warp_tile = new_smem.tile[
                    Self.stageN, 1, tile_width
                ](0, Int(self.warp_id), 0)
                var c_smem_warp_tile = _c_smem_warp_tile.reshape[
                    coalesce(_c_smem_warp_tile.layout)
                ]()

                var warp_offset = Int(self.warp_id) * tile_width
                store_fragment_to_smem[
                    Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
                ](upper_casted, c_smem_warp_tile, UInt32(warp_offset))
        else:
            comptime tiles_per_frag = (
                Self.stageN * Self.swizzle_width // Self.stage_contiguous_size
            )

            comptime if is_lower_required:
                var c_smem_warp_tile_upper = c_smem_tile.tile[
                    tiles_per_frag, Self.stage_contiguous_size
                ](2 * Int(self.warp_id), 0).reshape[
                    Layout.row_major(Self.stageN, Self.swizzle_width)
                ]()
                var c_smem_warp_tile_lower = c_smem_tile.tile[
                    tiles_per_frag, Self.stage_contiguous_size
                ](2 * Int(self.warp_id) + 1, 0).reshape[
                    Layout.row_major(Self.stageN, Self.swizzle_width)
                ]()

                store_fragment_to_smem[
                    Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
                ](upper_casted, c_smem_warp_tile_upper)
                store_fragment_to_smem[
                    Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
                ](lower_casted, c_smem_warp_tile_lower)
            else:
                var c_smem_warp_tile_upper = c_smem_tile.tile[
                    tiles_per_frag, Self.stage_contiguous_size
                ](Int(self.warp_id), 0).reshape[
                    Layout.row_major(Self.stageN, Self.swizzle_width)
                ]()

                store_fragment_to_smem[
                    Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
                ](upper_casted, c_smem_warp_tile_upper)

    @always_inline
    fn _write_non_transpose_lt[
        repeat: Int, is_lower_required: Bool
    ](
        self,
        upper_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        lower_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        c_smem_tile: SMemTile[Self.c_type, Self.c_smem_layout, alignment=128],
    ):
        """Non-transposed output (LayoutTensor path)."""
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

        comptime if is_lower_required:
            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                Self.data_paths, Self.stageN
            ](1, 0)
            store_fragment_to_smem[
                Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
            ](lower_casted, c_smem_warp_tile_lower)

    @always_inline
    fn write_fragments[
        repeat: Int
    ](
        self,
        upper_frag: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        lower_frag: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        c_smem_tile: TileTensor[address_space = AddressSpace.SHARED, ...],
    ):
        """Write pre-loaded fragments to SMEM."""
        comptime is_lower_required = Self.Config.is_lower_frag_required

        comptime if Self.transpose_c:
            self._write_transpose[repeat, is_lower_required](
                upper_frag, lower_frag, c_smem_tile
            )
        else:
            self._write_non_transpose[repeat, is_lower_required](
                upper_frag, lower_frag, c_smem_tile
            )

    @always_inline
    fn _write_transpose[
        repeat: Int, is_lower_required: Bool
    ](
        self,
        upper_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        lower_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        c_smem_tile: TileTensor[address_space = AddressSpace.SHARED, ...],
    ):
        """Transposed output using reshape.

        Two paths based on swizzle width:
        - Large swizzle (SWIZZLE_128B): multiple warps share a swizzle block,
          requiring warp_offset for correct swizzle/deswizzle addressing.
        - Small swizzle (SWIZZLE_32B): each warp fragment fits within its own
          swizzle block, using simple tiles_per_frag tiling.
        """
        from layout._coord import Coord, Idx

        # SWIZZLE_128B: swizzle_width=64 > data_paths=16 → True
        # SWIZZLE_32B:  swizzle_width=16 == data_paths=16 → False
        comptime _large_swizzle = Self.swizzle_width > Self.data_paths

        comptime if _large_swizzle:
            comptime if is_lower_required:
                comptime tile_width = 32
                comptime num_swblocks = Self.stage_contiguous_size // Self.swizzle_width

                # 4D logical layout: (num_swblocks, stageN, 2, tile_width)
                comptime logical_layout = InternalLayout(
                    Coord(
                        Idx[num_swblocks](),
                        Idx[Self.stageN](),
                        Idx[2](),
                        Idx[tile_width](),
                    ),
                    Coord(
                        Idx[Self.stageN * Self.swizzle_width](),
                        Idx[2 * tile_width](),
                        Idx[tile_width](),
                        Idx[1](),
                    ),
                )
                var new_smem = c_smem_tile.reshape(logical_layout)

                warp_j, warp_i = divmod(Int(self.warp_id), 2)
                var tiled = new_smem.tile[1, Self.stageN, 1, tile_width](
                    Coord(Idx(warp_j), Idx(0), Idx(warp_i), Idx(0))
                )

                # Coalesce: (1, stageN, 1, 32) -> (stageN, 32)
                comptime coalesced = InternalLayout(
                    Coord(Idx[Self.stageN](), Idx[tile_width]()),
                    Coord(Idx[2 * tile_width](), Idx[1]()),
                )
                var c_smem_warp_tile = tiled.reshape(coalesced)

                var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                    Self.stageN, Self.data_paths
                ](Coord(Idx(0), Idx(0)))
                var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                    Self.stageN, Self.data_paths
                ](Coord(Idx(0), Idx(1)))

                var warp_offset = warp_i * tile_width
                store_fragment_to_smem[
                    Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
                ](upper_casted, c_smem_warp_tile_upper, UInt32(warp_offset))

                warp_offset += tile_width // 2
                store_fragment_to_smem[
                    Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
                ](lower_casted, c_smem_warp_tile_lower, UInt32(warp_offset))
            else:
                comptime tile_width = 16
                comptime logical = row_major[Self.stageN, 4, tile_width]()
                var new_smem = c_smem_tile.reshape(logical)

                var tiled = new_smem.tile[Self.stageN, 1, tile_width](
                    Coord(Idx(0), Idx(Int(self.warp_id)), Idx(0))
                )

                # Coalesce: (stageN, 1, 16) -> (stageN, 16)
                comptime coalesced = InternalLayout(
                    Coord(Idx[Self.stageN](), Idx[tile_width]()),
                    Coord(Idx[4 * tile_width](), Idx[1]()),
                )
                var c_smem_warp_tile = tiled.reshape(coalesced)

                var warp_offset = Int(self.warp_id) * tile_width
                store_fragment_to_smem[
                    Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
                ](upper_casted, c_smem_warp_tile, UInt32(warp_offset))
        else:
            comptime tiles_per_frag = (
                Self.stageN * Self.swizzle_width // Self.stage_contiguous_size
            )
            comptime reshaped = row_major[Self.stageN, Self.swizzle_width]()

            comptime if is_lower_required:
                var c_smem_warp_tile_upper = c_smem_tile.tile[
                    tiles_per_frag, Self.stage_contiguous_size
                ](Coord(Idx(2 * Int(self.warp_id)), Idx(0))).reshape(reshaped)
                var c_smem_warp_tile_lower = c_smem_tile.tile[
                    tiles_per_frag, Self.stage_contiguous_size
                ](Coord(Idx(2 * Int(self.warp_id) + 1), Idx(0))).reshape(
                    reshaped
                )

                store_fragment_to_smem[
                    Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
                ](upper_casted, c_smem_warp_tile_upper)
                store_fragment_to_smem[
                    Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
                ](lower_casted, c_smem_warp_tile_lower)
            else:
                var c_smem_warp_tile_upper = c_smem_tile.tile[
                    tiles_per_frag, Self.stage_contiguous_size
                ](Coord(Idx(Int(self.warp_id)), Idx(0))).reshape(reshaped)

                store_fragment_to_smem[
                    Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
                ](upper_casted, c_smem_warp_tile_upper)

    @always_inline
    fn _write_non_transpose[
        repeat: Int, is_lower_required: Bool
    ](
        self,
        upper_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        lower_casted: SIMD[Self.c_type, Self.Config.fragment_size * repeat],
        c_smem_tile: TileTensor[address_space = AddressSpace.SHARED, ...],
    ):
        """Non-transposed output."""
        from layout._coord import Coord, Idx

        comptime c_smem_tile_m = 32 if Self.cta_group == 2 else Self.BM // Self.num_output_warps
        var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, Self.stageN](
            Coord(Idx(Int(self.warp_id)), Idx(0))
        )

        var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
            Self.data_paths, Self.stageN
        ](Coord(Idx(0), Idx(0)))
        store_fragment_to_smem[
            Self.swizzle, Self.stageN, Self.transpose_c, Self.c_swizzle
        ](upper_casted, c_smem_warp_tile_upper)

        comptime if is_lower_required:
            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                Self.data_paths, Self.stageN
            ](Coord(Idx(1), Idx(0)))
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


struct SMemEpilogueWriter[
    # Infer-only: deduced from c_tiles argument type
    c_type: DType,
    num_output_stages: Int,
    //,
    # Configuration parameters (must be explicit)
    c_smem_dim0: Int,
    c_smem_dim1: Int,
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
](TrivialRegisterPassable):
    """SMEM-based epilogue: write accumulators and apply lambda in SMEM."""

    # Create layout from dimensions
    comptime c_smem_layout = Layout.row_major(
        Self.c_smem_dim0, Self.c_smem_dim1
    )

    comptime stageN = Self.c_smem_dim0 if Self.transpose_c else Self.c_smem_dim1
    comptime stage_contiguous_size = Self.c_smem_dim1

    comptime swizzle = _make_swizzle[Self.c_type, Self.c_swizzle]()
    comptime swizzle_width = Self.c_swizzle.bytes() // size_of[Self.c_type]()
    comptime data_paths = 16
    comptime barrier_threads = Self.num_output_warps * WARP_SIZE
    comptime OutputSyncBarrier = WarpGroupBarrier[Self.barrier_threads]
    comptime Tile = AccumTile[Self.epilogue_dtype, Self.rep_frag_size]
    comptime CTileArrayLT = SMemTileArray[
        Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
    ]

    var warp_id: UInt32
    var c_tiles: Self.CTileArrayLT
    var M: UInt32
    var N: UInt32
    var c_row: UInt32
    var c_col: UInt32

    @always_inline
    fn __init__(
        out self,
        warp_id: UInt32,
        c_tiles: SMemTileArray[
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
    fn __init__(
        out self,
        warp_id: UInt32,
        c_tiles: SMemTileArray2DRowMajor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.num_output_stages,
            128,
        ],
        c_shape: Tuple[UInt32, UInt32],
        c_coord: Tuple[UInt32, UInt32],
    ):
        """Initialize from TileTensor array (converts internally)."""
        self.warp_id = warp_id
        self.c_tiles = Self.CTileArrayLT(c_tiles.ptr)
        self.M = c_shape[0]
        self.N = c_shape[1]
        self.c_row = c_coord[0] * UInt32(Self.BM)
        self.c_col = c_coord[1] * UInt32(Self.MMA_N)

    @always_inline
    fn write_tile(self, tile: Self.Tile):
        """Write accumulator tile to SMEM and apply epilogue lambda."""
        # Double-buffer tile selection
        var c_smem_tile = self.c_tiles[Self.stage % Self.num_output_stages]

        comptime if Self.transpose_c:
            self._write_transpose(tile.upper, tile.lower, c_smem_tile)
        else:
            self._write_non_transpose(tile.upper, tile.lower, c_smem_tile)

    @always_inline
    fn _write_transpose(
        self,
        upper_frag: SIMD[Self.epilogue_dtype, Self.rep_frag_size],
        lower_frag: SIMD[Self.epilogue_dtype, Self.rep_frag_size],
        c_smem_tile: SMemTile[Self.c_type, Self.c_smem_layout, alignment=128],
    ):
        """Transpose path: reshape tiles and apply epilogue."""

        comptime if Self.is_lower_frag_required:
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

            var new_smem = SMemTile[
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
                upper_frag, c_smem_warp_tile_upper, UInt32(warp_offset)
            )
            warp_offset += tile_width // 2
            store_fragment_to_smem[Self.swizzle, Self.stageN, Self.transpose_c](
                lower_frag, c_smem_warp_tile_lower, UInt32(warp_offset)
            )

            Self.OutputSyncBarrier.sync()

            shared_memory_epilogue_transpose[
                UInt(Self.stage),
                UInt(Self.stageN),
                new_smem.dtype,
                new_smem.layout,
                Self.swizzle,
                Self.compute_lambda_fn,
                Self.num_output_warps,
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

            var new_smem = SMemTile[
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
            store_fragment_to_smem[Self.swizzle, Self.stageN, Self.transpose_c](
                upper_frag, c_smem_warp_tile_upper, UInt32(warp_offset)
            )

            Self.OutputSyncBarrier.sync()

            shared_memory_epilogue_transpose[
                UInt(Self.stage),
                UInt(Self.stageN),
                new_smem.dtype,
                new_smem.layout,
                Self.swizzle,
                Self.compute_lambda_fn,
                Self.num_output_warps,
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
        c_smem_tile: SMemTile[Self.c_type, Self.c_smem_layout, alignment=128],
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

        comptime if Self.is_lower_frag_required:
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
            Self.num_output_warps,
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
    num_output_warps: Int,
    warp_dim: Int,
    MMA_M: Int,
    BN: Int,
    cta_group: Int,
](
    M: UInt32,
    N: UInt32,
    c_col: UInt,
    c_row: UInt,
    c_smem: SMemTile[c_type, c_smem_layout, alignment=128],
    warp_i: UInt,
    warp_j: UInt,
):
    """Apply element-wise epilogue to transposed SMEM tile.

    Supports warp_dim=1 (stageN, warp_i, U) or warp_dim=2 (warp_j, stageN, warp_i, UL).
    """
    var gmem_col = c_col + stage * stageN
    var gmem_row = c_row

    comptime simd_size = simd_width_of[c_type]()
    comptime alignment = align_of[SIMD[c_type, simd_size]]()
    comptime swizzle_dim = 64

    comptime if warp_dim == 2:
        # Use new Layout for idx2crd operations
        comptime layout_3d = row_major[2, Int(stageN), swizzle_dim]()
        comptime assert c_smem_layout.rank() == 4, "c_smem_layout must be 4D"
        comptime thread_layout = Layout.row_major(1, 8, 1, 4)
        comptime result = zipped_divide(
            upcast(c_smem_layout, simd_size), thread_layout
        )
        comptime thread_layout_new = row_major[1, 8, 1, 4]()
        var lane = lane_id()
        var crd = thread_layout_new.idx2crd[out_dtype = DType.uint32](Int(lane))
        comptime thread_shape = IntTuple(0, UNKNOWN_VALUE, 0, UNKNOWN_VALUE)

        comptime for iter_i in range(result.shape[1][3].value()):
            comptime for iter_j in range(result.shape[1][1].value()):
                comptime rest_shape = IntTuple(
                    UNKNOWN_VALUE, iter_j, UNKNOWN_VALUE, iter_i
                )
                var coord = RuntimeTuple[
                    [thread_shape, rest_shape], element_type = DType.uint32
                ](
                    Int(0),
                    crd[1].value(),
                    Int(0),
                    crd[3].value(),
                    Int(warp_j),
                    iter_j,
                    Int(warp_i),
                    iter_i,
                )
                var offset = UInt32(simd_size) * rt_crd2idx[
                    [thread_shape, rest_shape],
                    result.shape,
                    result.stride,
                    DType.uint32,
                ](
                    coord,
                    RuntimeTuple[result.shape](),
                    RuntimeTuple[result.stride](),
                )
                var logical_crd = layout_3d.idx2crd[out_dtype = DType.uint32](
                    Int(offset)
                )
                var local_i: UInt32
                var local_j: UInt32

                var ci = logical_crd[0].value()
                var cj = logical_crd[1].value()
                var ck = logical_crd[2].value()

                comptime if cta_group == 2 and MMA_M == 128:
                    # logical shared memory -> global layout Layout B:
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
                    local_i = UInt32(cj) + UInt32(ci) * UInt32(BN)
                    local_j = UInt32(ck)
                else:
                    # logical shared memory -> global layout Layout A:
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-a
                    local_i = UInt32(cj)
                    local_j = UInt32(ci * swizzle_dim + ck)

                # undo swizzle to get logical `c_smem[logical_crd]` value.
                var ptr = (
                    c_smem.ptr
                    + swizzle(cj * swizzle_dim + ck)
                    + UInt32(ci * swizzle_dim) * UInt32(Int(stageN))
                )
                var row = local_i + UInt32(gmem_col)
                var col = local_j + UInt32(gmem_row)
                if row < UInt32(Int(M)) and col < UInt32(Int(N)):
                    var val = ptr.load[width=simd_size, alignment=alignment]()
                    ptr.store[width=simd_size, alignment=alignment](
                        compute_lambda_fn[alignment=alignment](
                            (Int(row), Int(col)), val
                        )
                    )
    else:
        # Layout F: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-f
        comptime assert c_smem_layout.rank() == 3, "c_smem_layout must be 3D"
        comptime thread_layout = Layout.row_major(min(16, Int(stageN)), 1, 2)
        comptime thread_bound = UInt(thread_layout.cosize())
        var lane = lane_id()
        if lane < thread_bound:
            comptime result = zipped_divide(
                upcast(c_smem_layout, simd_size), thread_layout
            )
            # Use new Layout for idx2crd operations
            comptime thread_layout_new = row_major[min(16, Int(stageN)), 1, 2]()
            var crd = thread_layout_new.idx2crd[out_dtype = DType.uint32](
                Int(lane)
            )
            comptime thread_shape = IntTuple(UNKNOWN_VALUE, 0, UNKNOWN_VALUE)
            comptime layout_2d_new = row_major[Int(stageN), swizzle_dim]()

            comptime for iter_i in range(result.shape[1][2].value()):
                comptime for iter_j in range(result.shape[1][0].value()):
                    comptime rest_shape = IntTuple(
                        iter_j,
                        UNKNOWN_VALUE,
                        iter_i,
                    )
                    var coord = RuntimeTuple[
                        [thread_shape, rest_shape], element_type = DType.uint32
                    ](
                        crd[0].value(),
                        Int(0),
                        crd[2].value(),
                        iter_j,
                        Int(warp_i),
                        iter_i,
                    )
                    var offset = UInt32(simd_size) * rt_crd2idx[
                        [thread_shape, rest_shape],
                        result.shape,
                        result.stride,
                        DType.uint32,
                    ](
                        coord,
                        RuntimeTuple[result.shape](),
                        RuntimeTuple[result.stride](),
                    )
                    var logical_crd = layout_2d_new.idx2crd[
                        out_dtype = DType.uint32
                    ](Int(offset))

                    var local_i = logical_crd[0].value()
                    var local_j = logical_crd[1].value()

                    # Undo swizzle to get logical value
                    var ptr = c_smem.ptr + swizzle(offset)
                    var row = UInt32(local_i) + UInt32(gmem_col)
                    var col = UInt32(local_j) + UInt32(gmem_row)
                    if row < UInt32(Int(M)) and col < UInt32(Int(N)):
                        var val = ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        ptr.store[width=simd_size, alignment=alignment](
                            compute_lambda_fn[alignment=alignment](
                                (Int(row), Int(col)), val
                            )
                        )

    WarpGroupBarrier[num_output_warps * WARP_SIZE].sync()


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
    num_output_warps: Int,
](
    M: UInt32,
    N: UInt32,
    c_col: UInt,
    c_row: UInt,
    c_smem_warp_tile_upper: SMemTile[c_type, c_smem_upper_layout, ...],
    c_smem_warp_tile_lower: SMemTile[c_type, c_smem_lower_layout, ...],
):
    """Apply element-wise epilogue to non-transposed SMEM tile.

    Each warp processes upper (rows 0-15) and lower (rows 16-31) fragments.
    Uses distribute layout to map SIMD vectors to threads within each warp.
    """
    # Global column with stage offset
    var gmem_col = c_col + stage * stageN

    # Each warp owns 32 rows: upper half (0-15) and lower half (16-31)
    var warp_base_row = warp_id() * 32
    var upper_row = warp_base_row
    var lower_row = warp_base_row + 16

    # Distribute layout: maps stageN elements across warp threads
    # e.g., stageN=32 → 8x4 (4 threads × 8 elements), stageN=16 → 16x2
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

    var lane_row, lane_col = divmod(lane_id(), distribute_cols)
    var col = lane_col * simd_size
    upper_row += lane_row
    lower_row += lane_row

    comptime for i in range(fragment_size):
        comptime alignment = align_of[SIMD[c_type, Int(simd_size)]]()

        # Compute swizzled SMEM offsets, then un-swizzle to get logical coords
        var swz_offset_upper = upper_row * shared_n + col
        var swz_offset_lower = lower_row * shared_n + col

        var offset_upper = swizzle(Int(swz_offset_upper))
        var offset_lower = swizzle(Int(swz_offset_lower))

        var local_upper_row: Int64
        var local_upper_col: Int64
        var local_lower_row: Int64
        var local_lower_col: Int64

        # Convert SMEM offset to logical (row, col) - layout differs by MMA_M size
        comptime if MMA_M != 256:
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

            local_upper_row = upper_coord[0].get_int()
            local_lower_row = lower_coord[0].get_int()

            var section_offset_upper = upper_coord[1][1].get_int()
            var col_offset_upper = upper_coord[1][0].get_int()
            var section_offset_lower = lower_coord[1][1].get_int()
            var col_offset_lower = lower_coord[1][0].get_int()

            local_upper_col = (
                section_offset_upper * Int64(num_stages * stageN)
                + col_offset_upper
            )
            local_lower_col = (
                section_offset_lower * Int64(num_stages * stageN)
                + col_offset_lower
            )

        else:
            # MMA_M=256: simple row-major indexing
            comptime fast_div = FastDiv[DType.uint32](Int(shared_n))

            local_upper_row = (
                Scalar[DType.int](offset_upper).cast[fast_div.uint_type]()
                / fast_div
            ).cast[DType.int64]()
            local_upper_col = Int64(offset_upper % Int(shared_n))

            local_lower_row = (
                Scalar[DType.int](offset_lower).cast[fast_div.uint_type]()
                / fast_div
            ).cast[DType.int64]()
            local_lower_col = Int64(offset_lower % Int(shared_n))

        # Convert local SMEM coords to global memory coords
        var gmem_upper_row = local_upper_row + Int64(c_row)
        var gmem_upper_col = local_upper_col + Int64(gmem_col)
        var gmem_lower_row = local_lower_row + Int64(c_row)
        var gmem_lower_col = local_lower_col + Int64(gmem_col)

        # Apply epilogue if within bounds
        if gmem_upper_row < Int64(Int(M)) and gmem_upper_col < Int64(Int(N)):
            c_smem_upper_frag[i, 0] = compute_lambda_fn[alignment=alignment](
                (Int(gmem_upper_row), Int(gmem_upper_col)),
                c_smem_upper_frag[i, 0],
            )

        if gmem_lower_row < Int64(Int(M)) and gmem_lower_col < Int64(Int(N)):
            c_smem_lower_frag[i, 0] = compute_lambda_fn[alignment=alignment](
                (Int(gmem_lower_row), Int(gmem_lower_col)),
                c_smem_lower_frag[i, 0],
            )

        # Advance to next chunk (spaced distribute_rows apart)
        upper_row += UInt(distribute_rows)
        lower_row += UInt(distribute_rows)

    WarpGroupBarrier[num_output_warps * WARP_SIZE].sync()
