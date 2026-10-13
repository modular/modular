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
"""Per-row output scales for the SM100 grouped matmul epilogues.

A grouped matmul scales each output row by its expert's scale. A `RowScales`
carrier adds a second, per-row factor: output row `m` is multiplied by
`expert_scale * row_scales[m]`. Activations quantized with a per-token tensor
scale need it to undo that scale in the epilogue.

Implementations:

- `NullRowScales`: zero-sized, `Enabled` is `False`, so it adds no
  kernel-argument bytes and no loads.
- `RealRowScales`: one `bfloat16` scale per output row in global memory,
  widened to `float32` on load.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import ceildiv
from std.memory import Pointer

from max.gpu import WARP_SIZE
import max.gpu.primitives.warp as warp


trait RowScales(DevicePassable, TrivialRegisterPassable):
    """Per-row output scales applied in the epilogue."""

    comptime Enabled: Bool
    """Whether the epilogue applies row scales. `False` compiles them out."""

    def load(self, row: Int) -> Float32:
        """Returns the scale of one output row.

        Args:
            row: Absolute output row (token) index.

        Returns:
            The row's scale.
        """
        ...


struct NullRowScales(RowScales):
    """Zero-sized no-op row scales, the default for every caller."""

    comptime Enabled = False
    comptime device_type: AnyType = Self

    @inline(.always)
    def __init__(out self):
        """Constructs the no-op row scales."""
        pass

    @inline(.always)
    def load(self, row: Int) -> Float32:
        """Returns 1.

        Args:
            row: Unused.

        Returns:
            Always 1.
        """
        return 1.0

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        pass

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name.

        Returns:
            Always `"NullRowScales"`.
        """
        return "NullRowScales"


struct RealRowScales(RowScales):
    """Row scales read from an array of one `bfloat16` per output row."""

    comptime Enabled = True
    comptime device_type: AnyType = Self

    @__allow_legacy_any_origin_fields
    var ptr: Pointer[BFloat16, ImmutAnyOrigin]
    """Device pointer to the per-row scales, indexed like the output rows."""

    @inline(.always)
    def __init__(out self, ptr: Pointer[BFloat16, ImmutAnyOrigin]):
        """Wraps a device pointer to the per-row scales.

        Args:
            ptr: Device pointer to one scale per output row.
        """
        self.ptr = ptr

    @inline(.always)
    def load(self, row: Int) -> Float32:
        """Reads the scale of one output row.

        Args:
            row: Absolute output row (token) index.

        Returns:
            The row's scale, widened to `float32`.
        """
        return self.ptr[unsafe_offset=row].cast[DType.float32]()

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name.

        Returns:
            Always `"RealRowScales"`.
        """
        return "RealRowScales"


struct TileRowScales[RowScalesT: RowScales, tile_rows: Int](
    TrivialRegisterPassable
):
    """One output tile's scaled row factors, spread across a warp's lanes.

    Lane `l` loads rows `l`, `l + 32`, ... of the tile once, with coalesced
    reads. Each epilogue stage then gathers the factors its fragments need
    with warp shuffles, so only the first stage waits on global memory.

    Parameters:
        RowScalesT: The per-row scales type.
        tile_rows: Number of output rows in the tile.
    """

    comptime rows_per_lane = ceildiv(Self.tile_rows, WARP_SIZE)

    var values: SIMD[DType.float32, Self.rows_per_lane]
    var scale: Float32

    @inline(.always)
    def __init__(
        out self,
        row_scales: Self.RowScalesT,
        first_row: UInt32,
        row_end: UInt32,
        lane: UInt32,
        scale: Float32,
    ):
        """Loads this lane's share of the tile's row factors.

        Args:
            row_scales: The per-row scales.
            first_row: Output row of the tile's first row.
            row_end: Rows at or past this bound are not loaded and get 0.
            lane: Lane index within the warp.
            scale: Per-expert scale folded into every factor.
        """
        self.scale = scale
        self.values = SIMD[DType.float32, Self.rows_per_lane](0)
        comptime if Self.RowScalesT.Enabled:
            comptime for i in range(Self.rows_per_lane):
                var row = first_row + UInt32(i * WARP_SIZE) + lane
                if row < row_end:
                    self.values[i] = scale * row_scales.load(Int(row))

    @inline(.always)
    def pairs[
        repeats: Int, stage_row: Int
    ](self, lane: UInt32) -> SIMD[DType.float32, 2 * repeats]:
        """Returns the factors of one thread's 16x256b accumulator fragments
        when the fragment column is the output row (`transpose_c`).

        The thread holds fragment columns `(lane % 4) * 2 + 8 * r + j` for
        `r < repeats` and `j < 2`. Fragment elements `4 * r + j` and
        `4 * r + 2 + j` both sit in that column.

        Parameters:
            repeats: Number of 8-column repeats per fragment.
            stage_row: Tile row of the stage's fragment column 0.

        Args:
            lane: Lane index within the warp.

        Returns:
            Element `2 * r + j` is the factor of column
            `(lane % 4) * 2 + 8 * r + j`, or the expert scale everywhere
            when `RowScalesT` is disabled.
        """
        comptime if not Self.RowScalesT.Enabled:
            return SIMD[DType.float32, 2 * repeats](self.scale)
        else:
            var result = SIMD[DType.float32, 2 * repeats]()
            var lane_col = (lane % 4) * 2
            comptime for r in range(repeats):
                # An 8-row group never straddles two lanes' slots.
                comptime group_row = stage_row + 8 * r
                comptime slot = group_row // WARP_SIZE
                comptime for j in range(2):
                    result[2 * r + j] = warp.shuffle_idx(
                        self.values[slot],
                        UInt32(group_row % WARP_SIZE + j) + lane_col,
                    )
            return result

    @inline(.always)
    def fragment[
        repeats: Int, stage_row: Int
    ](self, lane: UInt32) -> SIMD[DType.float32, 4 * repeats]:
        """Returns one factor per element of a thread's 16x256b accumulator
        fragment when the fragment column is the output row (`transpose_c`).

        Upper and lower fragments differ only in TMEM row, which is the weight
        dim here, so one result serves both.

        Parameters:
            repeats: Number of 8-column repeats per fragment.
            stage_row: Tile row of the stage's fragment column 0.

        Args:
            lane: Lane index within the warp.

        Returns:
            The factors, laid out like the fragment.
        """
        var pairs = self.pairs[repeats, stage_row](lane)
        var result = SIMD[DType.float32, 4 * repeats]()
        comptime for r in range(repeats):
            comptime for i in range(2):
                comptime for j in range(2):
                    result[4 * r + 2 * i + j] = pairs[2 * r + j]
        return result
