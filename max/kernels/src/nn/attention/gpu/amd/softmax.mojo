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
from std.math.uutils import umod
from std.bit import log2_floor
from std.gpu import barrier, lane_id, warp_id as get_warp_id
from layout import (
    Layout,
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
    comptime frag_size = Self.frag_num_rows * Self.frag_num_cols

    # Number of mma unit tiles in the score matrix.
    # 2*num_m_mmas
    comptime num_colwise_tiles = Self.score_layout_by_mma_unit.shape[0].value()
    # num_n_mmas
    comptime num_rowwise_tiles = Self.score_layout_by_mma_unit.shape[1].value()
    # The online softmax attributes for each thread's elements (fragments).
    comptime num_rows_per_thread = Self.num_colwise_tiles * Self.frag_num_rows

    comptime row_layout = tt_row_major[
        Self.num_m_mmas, Self.fragment_layout.shape[0].value()
    ]()

    comptime RowMaxTensorType = TileTensor[
        Self.dtype,
        type_of(Self.row_layout),
        MutExternalOrigin,
        address_space=AddressSpace.LOCAL,
    ]

    comptime RowSumTensorType = Self.RowMaxTensorType

    var rowmax_tensor: Self.RowMaxTensorType
    var rowsum_tensor: Self.RowSumTensorType

    comptime score_frag_layout = tt_row_major[
        Self.num_colwise_tiles, Self.frag_num_rows
    ]()

    comptime ScoreFragTensorType = TileTensor[
        Self.dtype,
        type_of(Self.score_frag_layout),
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
        ](Self.row_layout)
        self.rowsum_tensor = tt_stack_allocation[
            dtype=Self.dtype, address_space=AddressSpace.LOCAL
        ](Self.row_layout)
        self.score_frag_rowmax = tt_stack_allocation[
            dtype=Self.dtype, address_space=AddressSpace.LOCAL
        ](Self.score_frag_layout)
        self.score_frag_rowsum = tt_stack_allocation[
            dtype=Self.dtype, address_space=AddressSpace.LOCAL
        ](Self.score_frag_layout).fill(0)
        self.correction = tt_stack_allocation[
            dtype=Self.dtype, address_space=AddressSpace.LOCAL
        ](Self.score_frag_layout).fill(1)

    @always_inline
    def _score_row_idx[col_tile: Int, row: Int](self, lane_row: Int) -> UInt32:
        """Map (col_tile, lane_row, row) to a linear score matrix row index."""
        return (
            UInt32(col_tile)
            * Self.num_colwise_lanes
            * UInt32(Self.frag_num_rows)
            + UInt32(lane_row * Self.frag_num_rows)
            + UInt32(row)
        )

    @always_inline
    def _reduce_rows[
        is_max: Bool
    ](
        self,
        score: TileTensor[Self.dtype, ...],
        warp_scratch: TileTensor[mut=True, Self.dtype, ...],
    ):
        """Reduce score rows to per-thread scalars (max or sum).

        Vectorizes the score tile, reduces within each warp via lane
        shuffles, then (if rows span multiple warps) communicates via
        shared memory to produce a single value per row.

        Parameters:
            is_max: True for rowwise max, False for rowwise sum.

        Args:
            score: Score tile in registers.
            warp_scratch: Shared memory scratch for cross-warp reduce.
        """
        var score_reg_tile = score.to_layout_tensor().vectorize[
            1, Self.frag_size
        ]()
        var scratch_lt = warp_scratch.to_layout_tensor()

        # Init accumulator: copy current max (for max) or zero (for sum).
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                comptime if is_max:
                    self.score_frag_rowmax[col_tile, row] = self.rowmax_tensor[
                        col_tile, row
                    ]
                else:
                    self.score_frag_rowsum[col_tile, row] = 0

        var warp_x = umod(get_warp_id[broadcast=True](), Self.num_rowwise_warps)
        var coords = idx2crd[Self.warp_layout](lane_id())
        var lane_contains_first_column = coords[1] == 0
        var lane_row = coords[0]

        # Per-thread fragment reduction + warp-level lane shuffle.
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row_tile in range(Self.num_rowwise_tiles):
                comptime tile_id = (
                    col_tile + row_tile * Self.num_colwise_tiles
                )
                var frag = score_reg_tile[tile_id, 0]

                comptime for row in range(Self.frag_num_rows):
                    comptime for col in range(Self.frag_num_cols):
                        comptime if is_max:
                            self.score_frag_rowmax[col_tile, row] = max(
                                self.score_frag_rowmax[col_tile, row],
                                frag[col if Self.frag_is_row_vector else row],
                            )
                        else:
                            self.score_frag_rowsum[col_tile, row] += frag[
                                col if Self.frag_is_row_vector else row
                            ]

            comptime for row in range(Self.frag_num_rows):
                comptime if is_max:
                    self.score_frag_rowmax[col_tile, row] = warp.lane_group_max[
                        Int(Self.num_rowwise_lanes),
                        stride=Int(Self.rowwise_lanes_stride),
                    ](self.score_frag_rowmax[col_tile, row])
                else:
                    self.score_frag_rowsum[col_tile, row] = warp.lane_group_sum[
                        Int(Self.num_rowwise_lanes),
                        stride=Int(Self.rowwise_lanes_stride),
                    ](self.score_frag_rowsum[col_tile, row])

        # Cross-warp reduce via shared memory (if rows span multiple warps).
        comptime if Self.num_rowwise_warps > 1:
            # SMEM offset: max uses rows [0, N), sum uses rows [N, 2N).
            comptime smem_row_offset = 0 if is_max else Self.num_rowwise_warps

            if lane_contains_first_column:
                comptime for col_tile in range(Self.num_colwise_tiles):
                    comptime for row in range(Self.frag_num_rows):
                        var sri = self._score_row_idx[col_tile, row](lane_row)
                        comptime if is_max:
                            scratch_lt[
                                warp_x, Int(sri)
                            ] = self.score_frag_rowmax[col_tile, row][0]
                        else:
                            scratch_lt[
                                warp_x + smem_row_offset, Int(sri)
                            ] = self.score_frag_rowsum[col_tile, row][0]

            barrier()

            if lane_contains_first_column:
                comptime for col_tile in range(Self.num_colwise_tiles):
                    comptime for row in range(Self.frag_num_rows):
                        var sri = self._score_row_idx[col_tile, row](lane_row)

                        comptime if is_max:
                            comptime for rw in range(Self.num_rowwise_warps):
                                self.score_frag_rowmax[col_tile, row] = max(
                                    rebind[Scalar[Self.dtype]](
                                        self.score_frag_rowmax[col_tile, row]
                                    ),
                                    rebind[Scalar[Self.dtype]](
                                        scratch_lt[rw, Int(sri)]
                                    ),
                                )
                        else:
                            self.score_frag_rowsum[col_tile, row] = 0
                            comptime for rw in range(Self.num_rowwise_warps):
                                self.score_frag_rowsum[col_tile, row] += rebind[
                                    Scalar[Self.dtype]
                                ](scratch_lt[rw + smem_row_offset, Int(sri)])

            # Broadcast reduced value to all lanes in the row.
            comptime for col_tile in range(Self.num_colwise_tiles):
                comptime for row in range(Self.frag_num_rows):
                    comptime if is_max:
                        self.score_frag_rowmax[
                            col_tile, row
                        ] = warp.lane_group_max[
                            Int(Self.num_rowwise_lanes),
                            stride=Int(Self.rowwise_lanes_stride),
                        ](
                            self.score_frag_rowmax[col_tile, row]
                        )
                    else:
                        # lane_group_max acts as broadcast (all lanes hold
                        # the same value after reduction).
                        self.score_frag_rowsum[
                            col_tile, row
                        ] = warp.lane_group_max[
                            Int(Self.num_rowwise_lanes),
                            stride=Int(Self.rowwise_lanes_stride),
                        ](
                            self.score_frag_rowsum[col_tile, row]
                        )

    @always_inline
    def calculate_qk_max(
        self,
        score: TileTensor[Self.dtype, ...],
        warp_scratch: TileTensor[mut=True, Self.dtype, ...],
    ):
        self._reduce_rows[is_max=True](score, warp_scratch)

    @always_inline
    def calculate_qk_sum(
        self,
        score: TileTensor[Self.dtype, ...],
        warp_scratch: TileTensor[mut=True, Self.dtype, ...],
    ):
        self._reduce_rows[is_max=False](score, warp_scratch)

    @always_inline
    def exp[
        start: Int = 0, stride: Int = 1
    ](self, score: TileTensor[mut=True, Self.dtype, ...]):
        var score_reg_tile = score.to_layout_tensor().vectorize[
            1, Self.frag_size
        ]()
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
        score: TileTensor[mut=True, Self.dtype, ...],
        scale: Scalar[Self.dtype],
    ):
        """Numerically stable scaled exp: exp2((score - max) * scale).

        Subtracts the unscaled max before scaling, so the subtraction is exact
        for the maximum element (IEEE 754 guarantees a - a == 0). This avoids
        the precision gap in exp_fma where fma(score, scale, -scaled_max) can
        produce nonzero results when score == max due to independent rounding
        of scaled_max.
        """
        var score_reg_tile = score.to_layout_tensor().vectorize[
            1, Self.frag_size
        ]()
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
            # Correction since previous max may be updated.
            comptime for row in range(Self.frag_num_rows):
                self.correction[col_tile, row] = Self.exp_function(
                    self.rowmax_tensor[col_tile, row]
                    - self.score_frag_rowmax[col_tile, row]
                )

    @always_inline
    def update_output(self, output: TileTensor[mut=True, Self.dtype, ...]):
        var output_reg_tile = output.to_layout_tensor().vectorize[
            1, Self.frag_size
        ]()
        comptime num_output_replications = output_reg_tile.layout.shape[
            0
        ].value() // (Self.num_colwise_tiles * Self.num_rowwise_tiles)

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
    def apply_sum_correction(self):
        """Apply rowsum *= correction (deferred sum rescale pattern)."""
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.rowsum_tensor[col_tile, row] = (
                    self.rowsum_tensor[col_tile, row]
                    * self.correction[col_tile, row]
                )

    @always_inline
    def update_sum_additive(self):
        """Additive rowsum update: rowsum += new_sum (no correction)."""
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.rowsum_tensor[col_tile, row] = (
                    self.rowsum_tensor[col_tile, row]
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
        output: TileTensor[mut=True, Self.dtype, ...],
        score: TileTensor[mut=True, Self.dtype, ...],
        warp_scratch: TileTensor[mut=True, Self.dtype, ...],
    ):
        self.calculate_qk_max(score, warp_scratch)
        self.exp(score)
        self.calculate_qk_sum(score, warp_scratch)
        self.calculate_correction()
        self.update_output(output)
        self.update_max()
        self.update_sum()
