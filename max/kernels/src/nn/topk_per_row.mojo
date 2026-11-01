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
"""Top-k with a pick count per row, for the DeepSeek-V4 indexer.

Selection runs on the block-wide bitonic and histogram-select kernels in
`topk_bitonic`, which pick all `k` elements together. The two-stage
`topk_gpu` extracts one element per round, so at `k = 512` it is a few
hundred serial reductions per row. This file only supplies the output stage: a sink that
turns each selected column into the `(value, index)` pair the graph op
returns.
"""

from std.builtin.device_passable import DeviceTypeEncoder
from std.utils.numerics import min_or_neg_inf

from max.gpu.host import DeviceContext

from .topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    RowBound,
    TopkSink,
    persistent_topk_block_split,
)

comptime TOPK_PER_ROW_MAX_K = PERSISTENT_TOPK_MAX_N
"""Largest `max_k` `topk_per_row_gpu` accepts."""


@fieldwise_init
struct _PerRowSink[dtype: DType](TopkSink):
    """Writes one row's picks as values and int64 columns.

    Slot `rank` keeps its column and score when it is within the row's pick
    count; every other slot holds `(-inf, -1)`.
    """

    var scores: ImmPointer[Scalar[Self.dtype], ImmUntrackedOrigin]
    """The row's scores."""
    var out_vals: MutPointer[Scalar[Self.dtype], MutUntrackedOrigin]
    """Flat `[rows, max_k]` output values."""
    var out_idxs: MutPointer[Int64, MutUntrackedOrigin]
    """Flat `[rows, max_k]` output columns."""
    var row_base: Int
    """Flat output slot of the row's first pick."""
    var picks: Int
    """Number of picks the row keeps."""

    @inline(.always)
    def store(self, idx: Int, col: Int32):
        """Writes output slot `idx` from the selected column `col`.

        Args:
            idx: Flat output slot, `row * max_k + rank`.
            col: Selected column, or `-1` for a padding slot.
        """
        var val = min_or_neg_inf[Self.dtype]()
        var out_col = Int64(-1)
        if idx - self.row_base < self.picks and col >= 0:
            val = self.scores[Int(col)]
            out_col = Int64(col)
        self.out_vals[idx] = val
        self.out_idxs[idx] = out_col


@fieldwise_init
struct _PerRowBound[dtype: DType](RowBound):
    """Hands every row to the top-k whole, with the sink for its picks."""

    comptime device_type: AnyType = Self
    """Device-side type alias; the struct is trivially device-passable."""

    comptime Sink = _PerRowSink[Self.dtype]
    """Writes the values and columns of each pick."""

    var scores: ImmPointer[Scalar[Self.dtype], ImmUntrackedOrigin]
    """Flat `[rows, n]` scores."""
    var k: ImmPointer[Int64, ImmUntrackedOrigin]
    """Pick count per row."""
    var out_vals: MutPointer[Scalar[Self.dtype], MutUntrackedOrigin]
    """Flat `[rows, max_k]` output values."""
    var out_idxs: MutPointer[Int64, MutUntrackedOrigin]
    """Flat `[rows, max_k]` output columns."""
    var n: Int32
    """Scores per row."""
    var max_k: Int32
    """Output slots per row."""

    @inline(.always)
    def row(
        self, row: Int, out_idxs: MutPointer[Int32, MutAnyOrigin]
    ) -> Tuple[Int, Self.Sink]:
        """Returns the row's column count and its sink.

        Args:
            row: Row index within the launch.
            out_idxs: Unused. The sink writes the int64 output directly.

        Returns:
            All `n` columns, and the sink for the row's picks.
        """
        var n = Int(self.n)
        var max_k = Int(self.max_k)
        return (
            n,
            Self.Sink(
                self.scores.unsafe_offset(row * n),
                self.out_vals,
                self.out_idxs,
                row * max_k,
                min(Int(self.k[row]), max_k),
            ),
        )

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name for runtime diagnostics.

        Returns:
            Always `"_PerRowBound"`.
        """
        return "_PerRowBound"


def topk_per_row_gpu[
    dtype: DType, //
](
    ctx: DeviceContext,
    scores: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    k: ImmPointer[Int64, ImmutAnyOrigin],
    out_vals: MutPointer[Scalar[dtype], MutAnyOrigin],
    out_idxs: MutPointer[Int64, MutAnyOrigin],
    rows: Int,
    n: Int,
    max_k: Int,
) raises:
    """Selects the first `k[r]` picks of each row's top-`max_k`.

    Row `r` gets the `k[r]` largest scores in descending order as
    `(value, column)` pairs, with equal scores in any order. Every other slot
    of its `max_k` holds `(-inf, -1)`.

    Parameters:
        dtype: Score and value dtype, float32 or bfloat16.

    Args:
        ctx: Device context.
        scores: Flat `[rows, n]` scores, 16 byte aligned.
        k: `[rows]` picks per row, each in `[0, max_k]`.
        out_vals: Flat `[rows, max_k]` output values.
        out_idxs: Flat `[rows, max_k]` output columns.
        rows: Number of rows.
        n: Scores per row.
        max_k: Output slots per row, in `[1, TOPK_PER_ROW_MAX_K]`.
    """
    if rows == 0:
        return
    var bound = _PerRowBound[dtype](
        scores.unsafe_origin_cast[ImmUntrackedOrigin](),
        k.unsafe_origin_cast[ImmUntrackedOrigin](),
        out_vals.unsafe_origin_cast[MutUntrackedOrigin](),
        out_idxs.unsafe_origin_cast[MutUntrackedOrigin](),
        Int32(n),
        Int32(max_k),
    )
    persistent_topk_block_split[ordered=True, RB=type_of(bound)](
        ctx,
        scores,
        out_idxs.unsafe_bitcast[Int32](),
        n,
        max_k,
        rows,
        Optional(bound),
    )
