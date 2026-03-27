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

import std.gpu.primitives.warp as warp
from std.bit import log2_floor
from std.gpu import (
    barrier,
    lane_id_uint as lane_id,
    warp_id_uint as get_warp_id,
)
from layout import (
    Layout,
    LayoutTensor,
    TileTensor,
    row_major as tt_row_major,
    stack_allocation as tt_stack_allocation,
)
from layout._utils import idx2crd
from nn.softmax import _exp2_concrete, _exp_concrete


struct Softmax[
    dtype: DType,
    score_layout_by_mma_unit: Layout,
    block_layout_by_warp: Layout,
    warp_layout: Layout,
    fragment_layout: Layout,
    use_exp2: Bool = False,
]:
    comptime num_shuffles_per_row = log2_floor(
        Self.warp_layout.shape[1].value()
    )

    comptime num_rowwise_lanes = UInt32(Self.warp_layout.shape[1].value())
    comptime num_colwise_lanes = UInt32(Self.warp_layout.shape[0].value())
    comptime rowwise_lanes_stride = UInt32(Self.warp_layout.stride[1].value())

    comptime exp_function = _exp2_concrete if Self.use_exp2 else _exp_concrete
    comptime num_m_mmas = Self.score_layout_by_mma_unit.shape[0].value()
    comptime num_colwise_warps = Self.block_layout_by_warp.shape[0].value()
    comptime num_rowwise_warps = Self.block_layout_by_warp.shape[1].value()

    # Assume p_reg_tile has been properly vectorized. The element layout
    # represents number elements per thread in a row or column
    # Each mma fragment is a 2D tile e.g. (1, x) for nvidia and (x, 1) for AMD.

    # TODO: fragment_layout should ideally be inferred from the shape of output_reg_tile or score_reg_tile
    comptime frag_num_rows = Self.fragment_layout.shape[0].value()
    comptime frag_num_cols = Self.fragment_layout.shape[1].value()

    comptime frag_is_row_vector = Self.frag_num_rows == 1

    # Number of mma unit tiles in the score matrix.
    # 2*num_m_mmas
    comptime num_colwise_tiles = Self.score_layout_by_mma_unit.shape[0].value()
    # num_n_mmas
    comptime num_rowwise_tiles = Self.score_layout_by_mma_unit.shape[1].value()
    # The online softmax attributes for each thread's elements (fragments).
    comptime num_rows_per_thread = Self.num_colwise_tiles * Self.frag_num_rows

    comptime row_tt_layout = tt_row_major[
        Self.num_m_mmas, Self.fragment_layout.shape[0].value()
    ]()

    comptime RowMaxTensorType = TileTensor[
        Self.dtype,
        type_of(Self.row_tt_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    comptime RowSumTensorType = Self.RowMaxTensorType

    var rowmax_tensor: Self.RowMaxTensorType
    var rowsum_tensor: Self.RowSumTensorType

    comptime score_frag_tt_layout = tt_row_major[
        Self.num_colwise_tiles, Self.frag_num_rows
    ]()

    comptime ScoreFragTensorType = TileTensor[
        Self.dtype,
        type_of(Self.score_frag_tt_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    var score_frag_rowmax: Self.ScoreFragTensorType
    var score_frag_rowsum: Self.ScoreFragTensorType
    var correction: Self.ScoreFragTensorType

    @always_inline
    def __init__(out self):
        self.rowmax_tensor = tt_stack_allocation[
            dtype=Self.dtype, address_space=AddressSpace.LOCAL
        ](Self.row_tt_layout)
        self.rowsum_tensor = tt_stack_allocation[
            dtype=Self.dtype, address_space=AddressSpace.LOCAL
        ](Self.row_tt_layout)
        self.score_frag_rowmax = tt_stack_allocation[
            dtype=Self.dtype, address_space=AddressSpace.LOCAL
        ](Self.score_frag_tt_layout)
        self.score_frag_rowsum = tt_stack_allocation[
            dtype=Self.dtype, address_space=AddressSpace.LOCAL
        ](Self.score_frag_tt_layout).fill(0)
        self.correction = tt_stack_allocation[
            dtype=Self.dtype, address_space=AddressSpace.LOCAL
        ](Self.score_frag_tt_layout).fill(1)

    @always_inline
    def calculate_qk_max(
        self,
        score_reg_tile: LayoutTensor[Self.dtype, ...],
        warp_scratch: LayoutTensor[mut=True, Self.dtype, ...],
    ):
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.score_frag_rowmax[col_tile, row] = self.rowmax_tensor[
                    col_tile, row
                ]

        var warp_x = get_warp_id() % UInt(Self.num_rowwise_warps)

        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row_tile in range(Self.num_rowwise_tiles):
                comptime tile_id = col_tile + row_tile * Self.num_colwise_tiles

                # Assume this is a rowwise vector for now see above constraint.
                var frag = score_reg_tile[tile_id, 0]

                comptime for row in range(Self.frag_num_rows):
                    comptime for col in range(Self.frag_num_cols):
                        self.score_frag_rowmax[col_tile, row] = max(
                            self.score_frag_rowmax[col_tile, row],
                            frag[col if Self.frag_is_row_vector else row],
                        )

            # Every four threads have elements on the same row.
            # Reduce max for T0-T3, T4-T7, etc for nvidia
            #                T0-T15, T16-T31, etc for amd
            comptime for row in range(Self.frag_num_rows):
                self.score_frag_rowmax[col_tile, row] = warp.lane_group_max[
                    Int(Self.num_rowwise_lanes),
                    stride=Int(Self.rowwise_lanes_stride),
                ](self.score_frag_rowmax[col_tile, row])

        var coords = idx2crd[Self.warp_layout](Int(lane_id()))
        var lane_contains_first_column = coords[1] == 0
        var lane_row = coords[0]

        # If a row is split across multiple warps, communicate via shared memory
        # to achieve the rowwise max.
        comptime if Self.num_rowwise_warps > 1:
            # Write per warp rowmax to shared memory.
            if lane_contains_first_column:
                comptime for col_tile in range(Self.num_colwise_tiles):
                    comptime for row in range(Self.frag_num_rows):
                        var score_row_idx = (
                            UInt32(col_tile)
                            * Self.num_colwise_lanes
                            * UInt32(Self.frag_num_rows)
                            + UInt32(lane_row * Self.frag_num_rows)
                            + UInt32(row)
                        )

                        # warp scratch has layout row_major(num_warps, num_rows). The
                        # "score_row_idx" is the idx-th row in the score matrix.
                        warp_scratch[
                            Int(warp_x), Int(score_row_idx)
                        ] = self.score_frag_rowmax[col_tile, row][0]

            barrier()

            # Reduce the warpwise rowmax.
            if lane_contains_first_column:
                comptime for col_tile in range(Self.num_colwise_tiles):
                    comptime for row in range(Self.frag_num_rows):
                        var score_row_idx = (
                            UInt32(col_tile)
                            * Self.num_colwise_lanes
                            * UInt32(Self.frag_num_rows)
                            + UInt32(lane_row * Self.frag_num_rows)
                            + UInt32(row)
                        )

                        comptime for row_warp in range(Self.num_rowwise_warps):
                            self.score_frag_rowmax[col_tile, row] = max(
                                rebind[Scalar[Self.dtype]](
                                    self.score_frag_rowmax[col_tile, row]
                                ),
                                rebind[Scalar[Self.dtype]](
                                    warp_scratch[row_warp, Int(score_row_idx)]
                                ),
                            )

        # TODO: We can let all threads read shared memory in the above so that
        # we don't need to use warp shuffling.
        comptime for col_tile in range(Self.num_colwise_tiles):
            # Broadcast to 4 threads in the same row.
            comptime if Self.num_rowwise_warps > 1:
                comptime for row in range(Self.frag_num_rows):
                    self.score_frag_rowmax[col_tile, row] = warp.lane_group_max[
                        Int(Self.num_rowwise_lanes),
                        stride=Int(Self.rowwise_lanes_stride),
                    ](self.score_frag_rowmax[col_tile, row])

    @always_inline
    def calculate_qk_sum(
        self,
        score_reg_tile: LayoutTensor[Self.dtype, ...],
        warp_scratch: LayoutTensor[mut=True, Self.dtype, ...],
    ):
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.score_frag_rowsum[col_tile, row] = 0

        var warp_x = get_warp_id[broadcast=True]() % UInt(
            Self.num_rowwise_warps
        )

        var coords = idx2crd[Self.warp_layout](Int(lane_id()))
        var lane_contains_first_column = coords[1] == 0
        var lane_row = coords[0]

        comptime for col_tile in range(Self.num_colwise_tiles):
            # Sum softmax numerator from a thread's fragments.
            comptime for row_tile in range(Self.num_rowwise_tiles):
                comptime tile_id = col_tile + Self.num_colwise_tiles * row_tile
                var frag = score_reg_tile[tile_id, 0]

                comptime for row in range(Self.frag_num_rows):
                    comptime for col in range(Self.frag_num_cols):
                        self.score_frag_rowsum[col_tile, row] += frag[
                            col if Self.frag_is_row_vector else row
                        ]

            comptime for row in range(Self.frag_num_rows):
                self.score_frag_rowsum[col_tile, row] = warp.lane_group_sum[
                    Int(Self.num_rowwise_lanes),
                    stride=Int(Self.rowwise_lanes_stride),
                ](self.score_frag_rowsum[col_tile, row])

        # Reduce rowsum via shared memory.

        comptime if Self.num_rowwise_warps > 1:
            # Write per warp rowmax to shared memory.
            if lane_contains_first_column:
                comptime for col_tile in range(Self.num_colwise_tiles):
                    comptime for row in range(Self.frag_num_rows):
                        # Each thread handle two rows in the mma output.
                        var score_row_idx = (
                            UInt32(col_tile)
                            * Self.num_colwise_lanes
                            * UInt32(Self.frag_num_rows)
                            + UInt32(lane_row * Self.frag_num_rows)
                            + UInt32(row)
                        )

                        warp_scratch[
                            warp_x + UInt(Self.num_rowwise_warps),
                            Int(score_row_idx),
                        ] = self.score_frag_rowsum[col_tile, row][0]

            # Guard writing warp_scratch
            barrier()

            # Reduce the warpwise rowsum.
            if lane_contains_first_column:
                comptime for col_tile in range(Self.num_colwise_tiles):
                    comptime for row in range(Self.frag_num_rows):
                        var score_row_idx = (
                            UInt32(col_tile)
                            * Self.num_colwise_lanes
                            * UInt32(Self.frag_num_rows)
                            + UInt32(lane_row * Self.frag_num_rows)
                            + UInt32(row)
                        )

                        self.score_frag_rowsum[col_tile, row] = 0

                        # Reduce rowmax. Warps in the same row do the same reduction.
                        comptime for row_warp in range(Self.num_rowwise_warps):
                            self.score_frag_rowsum[col_tile, row] += rebind[
                                Scalar[Self.dtype]
                            ](
                                warp_scratch[
                                    row_warp + Self.num_rowwise_warps,
                                    Int(score_row_idx),
                                ]
                            )

                # Broadcast to 4 threads in the same row e.g. T0 -> T0-T3.

            comptime for col_tile in range(Self.num_colwise_tiles):
                comptime for row in range(Self.frag_num_rows):
                    # Broadcast to 4 threads in the same row.
                    self.score_frag_rowsum[col_tile, row] = warp.lane_group_max[
                        Int(Self.num_rowwise_lanes),
                        stride=Int(Self.rowwise_lanes_stride),
                    ](self.score_frag_rowsum[col_tile, row])

    @always_inline
    def exp[
        start: Int = 0, stride: Int = 1
    ](self, score_reg_tile: LayoutTensor[mut=True, Self.dtype, ...]):
        comptime frag_type = score_reg_tile.element_type

        comptime for col_tile in range(Self.num_colwise_tiles):
            # Softmax numerator based on mma results.
            comptime for row_tile in range(
                start, Self.num_rowwise_tiles, stride
            ):
                comptime tile_id = col_tile + Self.num_colwise_tiles * row_tile

                comptime if Self.frag_is_row_vector:
                    score_reg_tile[tile_id, 0] = Self.exp_function(
                        score_reg_tile[tile_id, 0]
                        - rebind[frag_type](
                            SIMD[Self.dtype, Self.frag_num_cols](
                                self.score_frag_rowmax[col_tile, 0][0]
                            )
                        )
                    )
                else:
                    comptime for row in range(Self.frag_num_rows):
                        score_reg_tile[tile_id, 0][row] = Self.exp_function(
                            score_reg_tile[tile_id, 0][row]
                            - self.score_frag_rowmax[col_tile, row][0]
                        )

    @always_inline
    def scale_rowmax(self, scale: Scalar[Self.dtype]):
        """Scale score_frag_rowmax by scale factor (e.g. scale * log2e).

        Must be called after exp_scaled so that score_frag_rowmax is in the
        same units as rowmax_tensor for calculate_correction.
        """
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.score_frag_rowmax[col_tile, row] *= scale

    @always_inline
    def exp_scaled[
        start: Int = 0, stride: Int = 1
    ](
        self,
        score_reg_tile: LayoutTensor[mut=True, Self.dtype, ...],
        scale: Scalar[Self.dtype],
    ):
        """Numerically stable scaled exp: exp2((score - max) * scale).

        Subtracts the unscaled max before scaling, so the subtraction is exact
        for the maximum element (IEEE 754 guarantees a - a == 0). This avoids
        the precision gap in exp_fma where fma(score, scale, -scaled_max) can
        produce nonzero results when score == max due to independent rounding
        of scaled_max.
        """
        comptime frag_type = score_reg_tile.element_type

        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row_tile in range(
                start, Self.num_rowwise_tiles, stride
            ):
                comptime tile_id = col_tile + Self.num_colwise_tiles * row_tile

                comptime if Self.frag_is_row_vector:
                    var neg_max = rebind[frag_type](
                        SIMD[Self.dtype, Self.frag_num_cols](
                            -self.score_frag_rowmax[col_tile, 0][0]
                        )
                    )
                    var scale_vec = rebind[frag_type](
                        SIMD[Self.dtype, Self.frag_num_cols](scale)
                    )
                    score_reg_tile[tile_id, 0] = Self.exp_function(
                        (score_reg_tile[tile_id, 0] + neg_max) * scale_vec
                    )
                else:
                    comptime for row in range(Self.frag_num_rows):
                        var neg_max = -self.score_frag_rowmax[col_tile, row][0]
                        score_reg_tile[tile_id, 0][row] = Self.exp_function(
                            (score_reg_tile[tile_id, 0][row] + neg_max) * scale
                        )

    @always_inline
    def calculate_correction(self):
        comptime for col_tile in range(Self.num_colwise_tiles):
            # Corrention since previous max may be updated.
            comptime for row in range(Self.frag_num_rows):
                self.correction[col_tile, row] = Self.exp_function(
                    self.rowmax_tensor[col_tile, row]
                    - self.score_frag_rowmax[col_tile, row]
                )

    @always_inline
    def update_output(
        self, output_reg_tile: LayoutTensor[mut=True, Self.dtype, ...]
    ):
        comptime num_output_replications = output_reg_tile.layout.shape[
            0
        ].value() // (Self.num_colwise_tiles * Self.num_rowwise_tiles)
        # if num_output_replications != 1, then `warp_split_k` and it must equal `num_warps_n`.
        # FIXME: require `warp_split_k` when delaying inter-warp communication.
        comptime assert (
            num_output_replications == 1
            or num_output_replications % Self.num_rowwise_warps == 0
        )

        # if num_output_replications
        comptime for k in range(num_output_replications):
            # Correct previous result
            comptime for col_tile in range(Self.num_colwise_tiles):
                comptime for row_tile in range(Self.num_rowwise_tiles):
                    comptime tile_id = col_tile + row_tile * Self.num_colwise_tiles + k * Self.num_colwise_tiles * Self.num_rowwise_tiles

                    comptime output_frag_type = type_of(
                        output_reg_tile
                    ).element_type

                    comptime if Self.frag_is_row_vector:
                        output_reg_tile[tile_id, 0] = output_reg_tile[
                            tile_id, 0
                        ] * output_frag_type(self.correction[col_tile, 0][0])
                    else:
                        comptime for row in range(Self.frag_num_rows):
                            output_reg_tile[tile_id, 0][row] = (
                                output_reg_tile[tile_id, 0][row]
                                * self.correction[col_tile, row][0]
                            )

    @always_inline
    def update_sum(self):
        # Save current rowmax and rowsum
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.rowsum_tensor[col_tile, row] = (
                    self.rowsum_tensor[col_tile, row]
                    * self.correction[col_tile, row]
                    + self.score_frag_rowsum[col_tile, row]
                )

    @always_inline
    def update_max(self):
        # Save current rowmax and rowsum
        comptime for i in range(Self.num_colwise_tiles):
            comptime for j in range(Self.frag_num_rows):
                self.rowmax_tensor[i, j] = self.score_frag_rowmax[i, j]

    @always_inline
    def full(
        self,
        output_reg_tile: LayoutTensor[mut=True, Self.dtype, ...],
        score_reg_tile: LayoutTensor[mut=True, Self.dtype, ...],
        warp_scratch: LayoutTensor[mut=True, Self.dtype, ...],
    ):
        self.calculate_qk_max(score_reg_tile, warp_scratch)
        self.exp(score_reg_tile)
        self.calculate_qk_sum(score_reg_tile, warp_scratch)
        self.calculate_correction()
        self.update_output(output_reg_tile)
        self.update_max()
        self.update_sum()
