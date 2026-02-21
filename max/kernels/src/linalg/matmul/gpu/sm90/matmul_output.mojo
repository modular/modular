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

from math import ceildiv
from sys import simd_width_of, size_of

from gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import lane_id
from gpu.memory import fence_async_view_proxy
from gpu.compute.mma import st_matrix
from gpu.sync import named_barrier
from layout import IntTuple, Layout, LayoutTensor
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout, RuntimeTuple
from layout.swizzle import Swizzle, make_ldmatrix_swizzle
from layout.tensor_core_async import st_matrix_n_layout
from layout.tma_async import TMATensorTile
from memory import bitcast
from std.bit import log2_floor

from utils.index import IndexList

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from collections import OptionalReg
from ....structuring import (
    SMemTile,
    RegTile,
)
from .tile_writer import (
    TileWriterTMA,
    TileWriterThreadwise,
    FragmentToSMemWriter,
    RegisterToGMemWriter,
    TileCoordinates,
    RegTileWriter,
)
import itertools


struct MatmulTileWriter[
    dtype: DType,
    layout: Layout,
    address_space: AddressSpace,
    element_layout: Layout,
    layout_int_type: DType,
    linear_idx_type: DType,
    masked: Bool,
    alignment: Int,
    smem_tile_layout: Layout,
    //,
    /,
    *,
    BM: Int,
    BN: Int,
    swizzle: TensorMapSwizzle,
    wgmma_shape: IndexList[3],
    num_consumer: Int = 1,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    swapAB: Bool = False,
](TrivialRegisterPassable):
    comptime N = Self.layout.shape[1].value()
    comptime frag_size = Self.wgmma_shape[0] * Self.wgmma_shape[
        1
    ] // WARPGROUP_SIZE
    comptime num_m_mmas = Self.BM // Self.wgmma_shape[0] // Self.num_consumer
    comptime num_n_mmas = Self.BN // Self.wgmma_shape[1]
    comptime num_consumer_threads = Self.num_consumer * WARPGROUP_SIZE
    comptime simd_size = simd_width_of[Self.dtype]()

    # Layout dimensions
    comptime WG_BM = Self.smem_tile_layout.shape[0].value()
    comptime WG_BN = Self.smem_tile_layout.shape[1].value()

    comptime CTensorType = LayoutTensor[
        Self.dtype,
        Self.layout,
        MutAnyOrigin,
        address_space = Self.address_space,
        element_layout = Self.element_layout,
        layout_int_type = Self.layout_int_type,
        linear_idx_type = Self.linear_idx_type,
        masked = Self.masked,
        alignment = Self.alignment,
    ]
    comptime lambda_type = fn[dtype: DType, width: Int, *, alignment: Int = 1](
        IndexList[2], mut SIMD[dtype, width]
    ) capturing -> None

    # Instance fields
    var tensor: Self.CTensorType
    var smem_tile: SMemTile[Self.dtype, Self.smem_tile_layout, alignment=128]
    var warp_group_thread_idx: UInt
    var local_warp_group_idx: UInt
    var local_thread_idx: UInt
    var block_y: Int
    var block_x: Int

    @always_inline
    fn __init__(
        out self,
        tensor: Self.CTensorType,
        smem_tile: SMemTile[Self.dtype, Self.smem_tile_layout, alignment=128],
        warp_group_thread_idx: UInt,
        local_warp_group_idx: UInt,
        local_thread_idx: UInt,
        block_y: Int,
        block_x: Int,
    ):
        self.tensor = tensor
        self.smem_tile = smem_tile
        self.warp_group_thread_idx = warp_group_thread_idx
        self.local_warp_group_idx = local_warp_group_idx
        self.local_thread_idx = local_thread_idx
        self.block_y = block_y
        self.block_x = block_x

    @always_inline
    fn _calculate_output_bounds(self) -> Tuple[UInt32, UInt32]:
        """Calculate valid output bounds for the current block's tile."""
        var rows = self.tensor.dim[0]()
        var max_row: UInt32
        var max_col: UInt32

        comptime if Self.swapAB:
            # swapAB: tile covers rows [block_x * BN, ...] and cols [block_y * BM, ...]
            max_row = min(UInt32((self.block_x + 1) * Self.BN), UInt32(rows))
            max_col = min(UInt32((self.block_y + 1) * Self.BM), UInt32(Self.N))
        else:
            # Normal: tile covers rows [block_y * BM, ...] and cols [block_x * BN, ...]
            max_row = min(UInt32((self.block_y + 1) * Self.BM), UInt32(rows))
            max_col = min(UInt32((self.block_x + 1) * Self.BN), UInt32(Self.N))

        return max_row, max_col

    @always_inline
    fn _apply_epilogue[
        epilogue_fn: Self.lambda_type
    ](
        self,
        output_tile: LayoutTensor[Self.dtype, _, MutAnyOrigin, ...],
        tile_row_offset: Int,
        tile_col_offset: Int,
        max_row: UInt32,
        max_col: UInt32,
    ):
        """Apply epilogue operations (bias, activation) to shared memory data.
        """
        comptime epilogue = epilogue_fn
        comptime smem_swizzle = make_ldmatrix_swizzle[Self.dtype, Self.WG_BN]()
        comptime thread_layout = Layout.row_major(
            Self.num_consumer_threads // (Self.WG_BN // Self.simd_size),
            Self.WG_BN // Self.simd_size,
        )

        var output_fragment, fragment_offsets, _ = output_tile.vectorize[
            1, Self.simd_size
        ]().distribute_with_offset[thread_layout](self.local_thread_idx)
        var row_coord = tile_row_offset + fragment_offsets[0]
        var col_coord = tile_col_offset + fragment_offsets[1] * Self.simd_size

        var shared_fragment = self.smem_tile.vectorize[
            1, Self.simd_size
        ]().distribute[thread_layout, swizzle=smem_swizzle](
            self.local_thread_idx
        )

        comptime num_elements_per_thread = output_fragment.layout.size()

        comptime for i in range(num_elements_per_thread):
            comptime smem_idx = shared_fragment.layout(i)
            comptime output_idx = output_fragment.layout(i)
            comptime row_offset = output_idx // Self.N
            comptime col_offset = output_idx % Self.N
            var row = UInt32(row_coord + row_offset)
            var col = UInt32(col_coord + col_offset)

            if row < max_row and col < max_col:
                epilogue(
                    IndexList[2](Int(row), Int(col)),
                    shared_fragment[i, 0],
                )

    @always_inline
    fn _write_tile_to_gmem[
        accum_type: DType,
        reg_tile_layout: Layout,
        //,
        check_runtime_bounds: Bool = False,
    ](self, reg_tile: RegTile[accum_type, reg_tile_layout]):
        """Write from registers to global memory."""

        comptime out_tile_size_m = Self.BM if not Self.swapAB else Self.BN
        comptime out_tile_size_n = Self.BN if not Self.swapAB else Self.BM

        var m_block = self.block_y if not Self.swapAB else self.block_x
        var n_block = self.block_x if not Self.swapAB else self.block_y

        var output_tile, tile_origin, _ = self.tensor.tile_with_offset[
            out_tile_size_m, out_tile_size_n
        ](m_block, n_block)

        # For normal: M is divided by num_consumer, N stays full
        # For swapAB: M stays full (BN), N is divided by num_consumer
        comptime tile_slice_m_regular = Self.BM // Self.num_consumer
        comptime tile_slice_n_regular = Self.BN
        comptime tile_slice_m_swapAB = Self.BN
        comptime tile_slice_n_swapAB = Self.BM // Self.num_consumer

        comptime tile_slice_m = tile_slice_m_regular if not Self.swapAB else tile_slice_m_swapAB
        comptime tile_slice_n = tile_slice_n_regular if not Self.swapAB else tile_slice_n_swapAB

        var coord_m = Int(self.local_warp_group_idx) if not Self.swapAB else 0
        var coord_n = 0 if not Self.swapAB else Int(self.local_warp_group_idx)

        var consumer_tile, consumer_coords, _ = output_tile.tile_with_offset[
            tile_slice_m, tile_slice_n
        ](coord_m, coord_n)

        var tile_coords: OptionalReg[TileCoordinates] = None
        var max_row: OptionalReg[UInt32] = None

        comptime if (
            Self.elementwise_lambda_fn is not None
            or Self.elementwise_compute_lambda_fn is not None
        ):
            tile_coords = TileCoordinates(
                IndexList[2](tile_origin[0], tile_origin[1]),
                IndexList[2](consumer_coords[0], consumer_coords[1]),
            )
            max_row = UInt32(self.tensor.dim[0]())

        var reg_writer = RegisterToGMemWriter[
            wgmma_shape = Self.wgmma_shape,
            num_consumer = Self.num_consumer,
            N = Self.N,
            epilogue_fn = Self.elementwise_lambda_fn,
            compute_lambda_fn = Self.elementwise_compute_lambda_fn,
            check_runtime_bounds=check_runtime_bounds,
            swapAB = Self.swapAB,
        ](
            consumer_tile,
            self.warp_group_thread_idx,
            Self.num_m_mmas,
            tile_coords,
            max_row,
        )

        comptime for row_tile, col_tile in itertools.product(
            range(Self.num_m_mmas), range(Self.num_n_mmas)
        ):
            reg_writer.write_tile(
                reg_tile,
                (UInt(row_tile), UInt(col_tile)),
            )

    @always_inline
    fn _write_tile_stmatrix[
        tma_layout: Layout,
        desc_layout: Layout,
        accum_type: DType,
        reg_tile_layout: Layout,
        //,
    ](
        self,
        tma_op: TMATensorTile[Self.dtype, tma_layout, desc_layout],
        reg_tile: RegTile[accum_type, reg_tile_layout],
        output_tile: LayoutTensor[Self.dtype, _, MutAnyOrigin, ...],
        tile_origin: Self.CTensorType.CornerCoordsType,
    ):
        """Use st.matrix instructions for optimized bf16 output."""
        var max_row, max_col = self._calculate_output_bounds()

        comptime TMA_BN_regular = tma_layout.shape[
            1
        ].value() if Self.use_tma_store else Self.WG_BN

        comptime TMA_BN_swapAB = tma_layout.shape[
            0
        ].value() if Self.use_tma_store else Self.WG_BM

        comptime TMA_BN = TMA_BN_swapAB if Self.swapAB else TMA_BN_regular

        comptime needs_x2_regular = Self.BN % Self.WG_BN != 0
        comptime needs_x2_swapAB = Self.BN % Self.WG_BM != 0

        comptime needs_x2 = needs_x2_swapAB if Self.swapAB else needs_x2_regular

        constrained[
            needs_x2 == (Self.frag_size % 4 == 0 and Self.frag_size % 8 != 0),
            "stmatrix and wgmma register count conflict: needs_x2 = "
            + String(needs_x2)
            + " frag_size ="
            + String(Self.frag_size),
        ]()

        comptime fragment_writer_type[
            sub_wg_id: Int, half_tile: Bool
        ] = FragmentToSMemWriter[
            tile_n_size=TMA_BN,
            num_m_mmas = Self.num_m_mmas,
            num_consumer = Self.num_consumer,
            half_tile=half_tile,
            WG_BM = Self.WG_BM,
            WG_BN = Self.WG_BN,
            sub_wg_id=sub_wg_id,
            swapAB = Self.swapAB,
        ]

        comptime num_column_tiles = ceildiv(Self.BN, Self.WG_BN)
        comptime num_row_tile = ceildiv(Self.BN, Self.WG_BM)

        comptime num_tile = num_column_tiles if not Self.swapAB else num_row_tile
        comptime last_tile = Self.BN // Self.WG_BN if not Self.swapAB else Self.BN // Self.WG_BM

        comptime for tile_idx in range(num_tile):
            comptime is_partial_tile = needs_x2 and tile_idx == last_tile

            # Write fragments to shared memory
            var fragment_writer = fragment_writer_type[
                tile_idx, is_partial_tile
            ](
                self.smem_tile,
                self.warp_group_thread_idx,
                self.local_warp_group_idx,
            )

            comptime for tma_chunk in range(
                (Self.WG_BN if not Self.swapAB else Self.WG_BM) // TMA_BN
            ):
                fragment_writer.write_tile(reg_tile, (UInt(0), UInt(tma_chunk)))

            named_barrier[Int32(Self.num_consumer_threads)](10)

            # swapAB: swap tile shape and position
            comptime tile_rows = Self.WG_BM if Self.swapAB else Self.BM
            comptime tile_cols = Self.WG_BN
            var pos_row = tile_idx if Self.swapAB else 0
            var pos_col = 0 if Self.swapAB else tile_idx

            var workgroup_tile, tile_coords, _ = output_tile.tile_with_offset[
                tile_rows, tile_cols
            ](pos_row, pos_col)

            var global_coords = (
                rebind[Self.CTensorType.CornerCoordsType](tile_coords)
                + tile_origin
            )

            @parameter
            fn apply_epilogue[lambda_fn: Self.lambda_type]():
                self._apply_epilogue[lambda_fn](
                    workgroup_tile,
                    global_coords[0],
                    global_coords[1],
                    max_row,
                    max_col,
                )

            comptime if Self.elementwise_compute_lambda_fn:
                comptime compute_fn = Self.elementwise_compute_lambda_fn.value()

                @parameter
                fn _compute[
                    dtype: DType, width: Int, *, alignment: Int = 1
                ](
                    index: IndexList[2], mut val: SIMD[dtype, width]
                ) capturing -> None:
                    val = compute_fn[alignment=alignment](index, val)

                apply_epilogue[_compute]()
                named_barrier[Int32(Self.num_consumer_threads)](10)

            comptime if Self.elementwise_lambda_fn:
                comptime epilogue_fn = Self.elementwise_lambda_fn.value()

                @parameter
                fn _epilogue[
                    dtype: DType, width: Int, *, alignment: Int = 1
                ](
                    index: IndexList[2], mut val: SIMD[dtype, width]
                ) capturing -> None:
                    _ = epilogue_fn[alignment=alignment](index, val)

                apply_epilogue[_epilogue]()
            else:
                comptime if Self.use_tma_store and not is_partial_tile:
                    var tma_writer = TileWriterTMA(Pointer(to=tma_op))

                    if self.local_thread_idx < UInt(Self.WG_BN // TMA_BN):
                        var smem_offset = self.smem_tile.ptr + (
                            Self.WG_BM * TMA_BN * Int(self.local_thread_idx)
                        )
                        var tma_tile = SMemTile[
                            Self.dtype, tma_layout, alignment=128
                        ](smem_offset)

                        var tma_coords = (
                            UInt(
                                self.block_x * Self.BN
                                + tile_idx * Self.WG_BN
                                + Int(self.local_thread_idx * UInt(TMA_BN))
                            ),
                            UInt(self.block_y * Self.BM),
                        )

                        tma_writer.write_tile(tma_tile, tma_coords)
                else:
                    comptime thread_layout = Layout.row_major(
                        Self.num_consumer_threads
                        // (Self.WG_BN // Self.simd_size),
                        Self.WG_BN // Self.simd_size,
                    )

                    var threadwise_writer = TileWriterThreadwise[
                        thread_layout=thread_layout,
                        simd_size = Self.simd_size,
                        half_tile=is_partial_tile,
                        swapAB = Self.swapAB,
                    ](workgroup_tile, self.local_thread_idx)

                    threadwise_writer.write_tile(
                        self.smem_tile, (UInt(0), UInt(0))
                    )

            named_barrier[Int32(Self.num_consumer_threads)](10)

    @always_inline
    fn write_tile[
        tma_layout: Layout,
        desc_layout: Layout,
        accum_type: DType,
        reg_tile_layout: Layout,
        //,
    ](
        self,
        tma_op: TMATensorTile[Self.dtype, tma_layout, desc_layout],
        reg_tile: RegTile[accum_type, reg_tile_layout],
    ):
        """Write output from registers to global memory.

        Selects optimized st.matrix path for bf16 when constraints are met,
        otherwise uses general register-to-global path.
        """
        # Output tile dimensions and block coordinates
        # For normal: tile is BM x BN, positioned at (block_y, block_x)
        # For swapAB: tile is BN x BM, positioned at (block_x, block_y)
        comptime tile_m = Self.BM if not Self.swapAB else Self.BN
        comptime tile_n = Self.BN if not Self.swapAB else Self.BM
        var block_row = self.block_y if not Self.swapAB else self.block_x
        var block_col = self.block_x if not Self.swapAB else self.block_y

        var output_tile, tile_origin, _ = self.tensor.tile_with_offset[
            tile_m, tile_n
        ](block_row, block_col)

        comptime output_tile_shape = String(output_tile.layout.shape)

        comptime TMA_BN = tma_layout.shape[
            1
        ].value() if Self.use_tma_store else Self.WG_BN
        comptime row_size_aligned = Self.N * size_of[Self.dtype]() % 16 == 0

        # Check if st.matrix optimization can be used
        # fmt: off
        comptime can_use_stmatrix_normal = (
            accum_type == DType.float32 and Self.dtype == DType.bfloat16  # F32→BF16
            and Self.frag_size % 4 == 0                               # Register count
            and Self.BM % Self.wgmma_shape[0] == 0                              # M alignment
            and Self.WG_BN % 16 == 0                                  # Shared memory
            and Self.num_consumer <= 2                                     # Thread limit
            and Self.BN == Self.wgmma_shape[1]                                  # Tile size
            and Self.BM == Self.WG_BM                                      # Block size
            and row_size_aligned                                      # Row alignment
        )

        comptime can_use_stmatrix_swapAB = (
            accum_type == DType.float32 and Self.dtype == DType.bfloat16             # F32→BF16
            and Self.frag_size % 4 == 0                                              # Register count (at least stmatrix x2 can be used)
            and Self.BM % Self.wgmma_shape[0] == 0                                   # each consumer should get one wgmma tile
            and Self.WG_BM % 8 == 0                                                  # Shared memory, must have at least 8 rows for swapAB
            and Self.num_consumer <= 2                                               # Thread limit
            and Self.BN == Self.wgmma_shape[1]                                       # Tile size
            and self.BM == Self.WG_BN                                                # Block size (we load by chunks of BM (this checks that this aligns with it))
            and row_size_aligned                                                     # Row alignment
        )

        # fmt: on
        comptime can_use_stmatrix = can_use_stmatrix_swapAB if Self.swapAB else can_use_stmatrix_normal

        comptime if can_use_stmatrix:
            self._write_tile_stmatrix(
                tma_op,
                reg_tile,
                output_tile,
                tile_origin,
            )
        else:
            comptime check_bounds = (
                Self.N % Self.BN != 0
            ) if not Self.swapAB else (Self.N % Self.BM != 0)
            self._write_tile_to_gmem[check_runtime_bounds=check_bounds](
                reg_tile
            )
