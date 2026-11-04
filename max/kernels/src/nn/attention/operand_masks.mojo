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
"""Attention masks built from graph operands.

Each mask reads its operands only through `load`, so the graph compiler may
fuse an operand's producer into it. The view parameters default to `NoView`;
`make` returns the mask instantiated with the operands' real views.
"""

from std.math import align_down, ceildiv
from std.memory import bitcast
from std.sys import is_gpu, is_nvidia_gpu, llvm_intrinsic
from std.utils import StaticTuple

from extensibility import Fused, Input, LoadAccess, Loadable, TensorTuple
from layout import Coord
from nn.attention.mha_mask import (
    MASK_VALUE,
    CausalMask,
    LoadView,
    MaskStrategy,
    MHAMask,
    NoView,
    SlidingWindowCausalMask,
    TileMaskStatus,
    _nonfull_sets_known,
    load_view,
    naively_compute_total_iters,
    naively_get_first_nonempty_mask_col,
)
from std.builtin.device_passable import DeviceTypeEncoder

# TODO(GEX-4477): Delete `MaterializedMask` and the pointer-based `SegmentMask`
# and `RelativeLogitsMask` in `mha_mask.mojo` once attention takes its mask from
# the graph.


@inline(.always)
def aligned_load[
    V: Loadable, //, width: Int
](view: V, idx: Coord, inner: Int, inner_dim: Int) -> SIMD[V.dtype, width]:
    """Loads `width` elements of `view` at `idx`, as one aligned vector when
    it can on GPU.

    The read is aligned when `inner`, the innermost coordinate of `idx`, and
    `inner_dim`, the length of the innermost dimension, are both multiples of
    `width`: the row-major flat index is then a multiple of `width`. The view's
    own alignment still caps that, so a misaligned view reads correctly.

    Parameters:
        V: The view type.
        width: Number of elements to load.

    Args:
        view: The operand's device view.
        idx: The coordinate of the first element.
        inner: The innermost coordinate of `idx`.
        inner_dim: The length of the view's innermost dimension.

    Returns:
        The loaded elements.
    """
    # A CPU buffer may be less aligned than its view claims (one backed by a
    # NumPy array, say), and an unaligned load costs nothing there.
    comptime if is_gpu():
        if inner % width == 0 and inner_dim % width == 0:
            return view.load[width, element_alignment=width](idx)
    return view.load[width](idx)


# ===-----------------------------------------------------------------------===#
# AdditiveMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct AdditiveMask[
    Bias: Loadable & TrivialRegisterPassable = NoView,
](MHAMask, TrivialRegisterPassable):
    """Adds `bias[b, q - p, k]` (rank 3) or `bias[b, h, q - p, k]` (rank 4) to
    the score, where `p` is the bias's key length minus its query length.
    Positions outside the bias are masked.

    Operand: the bias.
    """

    comptime apply_log2e_after_mask: Bool = True
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = False
    comptime check_mask_during_decoding: Bool = True

    var bias: Self.Bias

    comptime MaskType[*Ts: Input & Fused & LoadAccess]: MHAMask = AdditiveMask[
        LoadView[Ts[0]]
    ]

    @staticmethod
    def make[
        *Ts: Input & Fused & LoadAccess
    ](operands: TensorTuple[*Ts]) -> Self.MaskType[*Ts]:
        """Builds the mask from its bias.

        Parameters:
            Ts: The operand types: the bias.

        Args:
            operands: The bias.

        Returns:
            The mask.
        """
        comptime assert len(Ts) == 1, "AdditiveMask takes one mask operand"
        comptime assert Ts[0].rank in (3, 4), "the bias must be rank 3 or 4"
        return {load_view(operands[0])}

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "AdditiveMask"

    @staticmethod
    def name() -> String:
        return "AdditiveMask"

    @inline(.always)
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
    ](
        self,
        coord: Coord[Int, Int, Int, Int],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime rank = Self.Bias.rank
        var shape = self.bias.shape()
        var num_q = Int(shape[rank - 2].value())
        var num_k = Int(shape[rank - 1].value())
        var q = coord[2] - (num_k - num_q)
        var k = coord[3]
        var bias = SIMD[dtype, width](MASK_VALUE)
        if 0 <= q < num_q:
            if k + width <= num_k:
                comptime if rank == 3:
                    bias = aligned_load[width](
                        self.bias, (coord[0], q, k), k, num_k
                    ).cast[dtype]()
                else:
                    bias = aligned_load[width](
                        self.bias, (coord[0], coord[1], q, k), k, num_k
                    ).cast[dtype]()
            elif k < num_k:
                for i in range(num_k - k):
                    comptime if rank == 3:
                        bias[i] = self.bias.load[1]((coord[0], q, k + i))[
                            0
                        ].cast[dtype]()
                    else:
                        bias[i] = self.bias.load[1](
                            (coord[0], coord[1], q, k + i)
                        )[0].cast[dtype]()
        return score_vec + bias

    @inline(.always)
    def status(
        self,
        seq_id: UInt32,
        tile_offset: Coord[Int, Int],
        tile_size: Coord[Int, Int],
    ) -> TileMaskStatus:
        # Every visible tile reads the bias.
        return TileMaskStatus.PARTIAL_MASK

    @inline(.always)
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        return naively_get_first_nonempty_mask_col[BM, BN](self, seq_id, row)

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        return BN

    @inline(.always)
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return naively_compute_total_iters[BM, BN](self, seq_id, row, num_cols)

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1

    @inline(.always)
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return ceildiv(num_cols, UInt32(BN))

    @inline(.always)
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {
            self.last_masked_set_end[BM, BN, page_size](seq_id, row, num_cols)
        }

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.UNKNOWN_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS}

    @inline(.always)
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # Unreachable: an additive bias can't be a visibility bitmask, so
        # `mask_strategies` never advertises `BITMASK`.
        return UInt32(0xFFFF_FFFF)

    @staticmethod
    def sliding_window_size() -> Int:
        return 0


# ===-----------------------------------------------------------------------===#
# SegmentMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct SegmentMask[
    Ids: Loadable & TrivialRegisterPassable = NoView,
    Offsets: Loadable & TrivialRegisterPassable = NoView,
](MHAMask, TrivialRegisterPassable):
    """Segment mask over ragged sequences: a query sees a key iff either
    token's segment id is -1 or the two ids are equal.

    Tokens with id -1 see, and are seen by, every token of their sequence;
    every other segment sees only itself and the -1 tokens. Within each
    sequence the ids must be nondecreasing, with -1 the only negative id, so a
    tile's status follows from the ids at its corners, and the keys a query
    tile sees lie in one range of columns.

    Operands: one int32 segment id per token of the ragged batch, and the
    batch's `[batch + 1]` uint32 row offsets.
    """

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = True

    var segment_ids: Self.Ids
    var row_offsets: Self.Offsets

    comptime MaskType[*Ts: Input & Fused & LoadAccess]: MHAMask = SegmentMask[
        LoadView[Ts[0]], LoadView[Ts[1]]
    ]

    @staticmethod
    def make[
        *Ts: Input & Fused & LoadAccess
    ](operands: TensorTuple[*Ts]) -> Self.MaskType[*Ts]:
        """Builds the mask from the segment ids and row offsets.

        Parameters:
            Ts: The operand types: the segment ids and the row offsets.

        Args:
            operands: The segment ids and the row offsets.

        Returns:
            The mask.
        """
        comptime assert len(Ts) == 2, "SegmentMask takes two mask operands"
        comptime assert Ts[0].dtype == .int32, "segment ids must be int32"
        comptime assert Ts[1].dtype == .uint32, "row offsets must be uint32"
        return {load_view(operands[0]), load_view(operands[1])}

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SegmentMask"

    @staticmethod
    def name() -> String:
        return "SegmentMask"

    @inline(.always)
    def _offset(self, seq_id: Int) -> Int:
        return Int(self.row_offsets.load[1]((seq_id,))[0])

    @inline(.always)
    def _id_at(self, token: Int) -> Int32:
        return self.segment_ids.load[1]((token,))[0].cast[.int32]()

    @inline(.always)
    def _bounds(self, seq_id: Int) -> Tuple[Int, Int]:
        var start = self._offset(seq_id)
        return start, self._offset(seq_id + 1) - start

    @inline(.always)
    def _id(self, start: Int, length: Int, position: Int) -> Int32:
        # Clamped, so the reads past a sequence's end that the kernels mask
        # afterwards stay inside it.
        return self._id_at(start + min(position, length - 1))

    @inline(.always)
    def _end_of(self, start: Int, length: Int, id: Int32) -> Int:
        """Returns the first position whose id exceeds `id`."""
        var lo, hi = 0, length
        while lo < hi:
            var mid = (lo + hi) // 2
            if self._id_at(start + mid) <= id:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @inline(.always)
    def _key_end(self, start: Int, length: Int, row: Int, rows: Int) -> Int:
        """Returns the end column of the keys query rows `[row, row + rows)`
        see."""
        if self._id(start, length, row) == -1:
            return length
        return self._end_of(
            start, length, self._id(start, length, row + rows - 1)
        )

    @inline(.always)
    def _key_range[
        BM: Int, BN: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> Tuple[
        UInt32, UInt32
    ]:
        """Returns the BN-aligned first and the end column of the keys the
        query tile at `row` sees, empty for a tile past its sequence."""
        var start, length = self._bounds(Int(seq_id))
        if Int(row) >= length:
            return UInt32(0), UInt32(0)
        var q_first = self._id(start, length, Int(row))
        var end = self._key_end(start, length, Int(row), BM)
        var first = 0
        if q_first != -1 and self._id_at(start) != -1:
            first = align_down(self._end_of(start, length, q_first - 1), BN)
        return UInt32(first), max(min(UInt32(end), num_cols), UInt32(first))

    @inline(.always)
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
    ](
        self,
        coord: Coord[Int, Int, Int, Int],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        var start, length = self._bounds(coord[0])
        var q_id = self._id(start, length, coord[2])
        var visible = SIMD[DType.bool, width](fill=q_id == -1)
        comptime for i in range(width):
            var k_id = self._id(start, length, coord[3] + i)
            visible[i] = visible[i] or k_id == -1 or k_id == q_id
        return visible.select(score_vec, MASK_VALUE)

    @inline(.always)
    def status(
        self,
        seq_id: UInt32,
        tile_offset: Coord[Int, Int],
        tile_size: Coord[Int, Int],
    ) -> TileMaskStatus:
        var start, length = self._bounds(Int(seq_id))
        var row, col = tile_offset[0], tile_offset[1]
        if col >= length:
            return TileMaskStatus.FULL_MASK
        if row >= length:
            return TileMaskStatus.PARTIAL_MASK
        if col >= self._key_end(start, length, row, tile_size[0]):
            return TileMaskStatus.FULL_MASK
        var q_first = self._id(start, length, row)
        var q_last = self._id(start, length, row + tile_size[0] - 1)
        var k_first = self._id(start, length, col)
        var k_last = self._id(start, length, col + tile_size[1] - 1)
        if col + tile_size[1] > length:
            return TileMaskStatus.PARTIAL_MASK
        if (
            q_last == -1
            or k_last == -1
            or (q_first == q_last and k_first == k_last and q_first == k_first)
        ):
            return TileMaskStatus.NO_MASK
        return TileMaskStatus.PARTIAL_MASK

    @inline(.always)
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        return self._key_range[BM, BN](seq_id, row, UInt32.MAX)[0]

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        return BN

    @inline(.always)
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        var first, end = self._key_range[BM, BN](seq_id, row, num_cols)
        return ceildiv(end - first, UInt32(BN))

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1

    @inline(.always)
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](seq_id, row, num_cols)

    @inline(.always)
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {
            self.last_masked_set_end[BM, BN, page_size](seq_id, row, num_cols)
        }

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.UNKNOWN_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS}

    @inline(.always)
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # Unreachable: `mask_strategies` never selects `BITMASK`.
        return UInt32(0xFFFF_FFFF)

    @staticmethod
    def sliding_window_size() -> Int:
        return 0


# ===-----------------------------------------------------------------------===#
# RelativeLogitsMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct RelativeLogitsMask[
    window_size: Int = 0,
    Bias: Loadable & TrivialRegisterPassable = NoView,
    CacheLengths: Loadable & TrivialRegisterPassable = NoView,
    RowOffsets: Loadable & TrivialRegisterPassable = NoView,
](MHAMask, TrivialRegisterPassable):
    """Causal (or sliding-window causal when `window_size > 0`) visibility,
    plus an additive bias by relative distance.

    The bias is gathered by `rel_dist = q - k` from a `(tokens, heads, extent)`
    table and added on every visible position; distances outside
    `[0, extent)` carry no bias. Masked lanes come out as exactly
    `MASK_VALUE`.

    Operands: the bias, the uint32 cache lengths `[batch]` (tokens cached
    before this call's), and the uint32 input row offsets `[batch + 1]`.
    """

    comptime _causal = CausalMask()
    comptime _window = SlidingWindowCausalMask[max(Self.window_size, 1)]()

    comptime apply_log2e_after_mask: Bool = True
    comptime mask_out_of_bound: Bool = is_nvidia_gpu()
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = True

    var bias: Self.Bias
    var cache_lengths: Self.CacheLengths
    var row_offsets: Self.RowOffsets

    comptime MaskType[*Ts: Input & Fused & LoadAccess]: MHAMask = (
        RelativeLogitsMask[
            Self.window_size,
            LoadView[Ts[0]],
            LoadView[Ts[1]],
            LoadView[Ts[2]],
        ]
    )

    @staticmethod
    def make[
        *Ts: Input & Fused & LoadAccess
    ](operands: TensorTuple[*Ts]) -> Self.MaskType[*Ts]:
        """Builds the mask from the bias, cache lengths and row offsets.

        Parameters:
            Ts: The operand types: the bias, the cache lengths and the row
                offsets.

        Args:
            operands: The bias, the cache lengths and the row offsets.

        Returns:
            The mask.
        """
        comptime assert (
            len(Ts) == 3
        ), "RelativeLogitsMask takes three mask operands"
        comptime assert (
            Ts[0].rank == 3
        ), "the bias must be (tokens, heads, extent)"
        comptime assert Ts[1].dtype == .uint32, "cache lengths must be uint32"
        comptime assert Ts[2].dtype == .uint32, "row offsets must be uint32"
        return {
            load_view(operands[0]),
            load_view(operands[1]),
            load_view(operands[2]),
        }

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "RelativeLogitsMask"

    @staticmethod
    def name() -> String:
        return String(t"RelativeLogitsMask[{Self.window_size}]")

    @inline(.always)
    def _flat_row(self, seq_id: Int, q_idx_abs: Int) -> Int:
        # `q_idx_abs` is the query's absolute in-sequence position (cached
        # prefix + new tokens); subtract the cached length and add this
        # sequence's ragged offset to recover the global flat row into `bias`.
        var cache_len = Int(self.cache_lengths.load[1]((seq_id,))[0])
        var row_offset = Int(self.row_offsets.load[1]((seq_id,))[0])
        return row_offset + (q_idx_abs - cache_len)

    @inline(.always)
    def _bias_at(self, row: Int, head: Int, dist: Int) -> Float32:
        return self.bias.load[1]((row, head, dist))[0].cast[.float32]()

    @inline(.always)
    def _bias_word(self, row: Int, head: Int, dist: Int) -> UInt32:
        """Returns the bf16 pair at even `dist` as one 32-bit word."""
        return bitcast[.uint32, 1](
            self.bias.load[2, element_alignment=2]((row, head, dist))
        )

    @inline(.always)
    def visibility_mask[
        dtype: DType,
        width: SIMDLength,
        //,
    ](
        self,
        coord: Coord[Int, Int, Int, Int],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime if Self.window_size > 0:
            return Self._window.mask(coord, score_vec)
        else:
            return Self._causal.mask(coord, score_vec)

    @inline(.always)
    def mask[
        dtype: DType,
        width: SIMDLength,
        //,
    ](
        self,
        coord: Coord[Int, Int, Int, Int],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        var q_idx = coord[2]
        var k_idx = coord[3]
        var shape = self.bias.shape()
        var extent = Int(shape[2].value())
        if extent == 0:
            return self.visibility_mask(coord, score_vec)
        var flat_row = self._flat_row(coord[0], q_idx)
        var row_in_bounds = 0 <= flat_row < Int(shape[0].value())
        # Clamp the row and the distance so every read is in bounds; `valid`
        # decides whether the value is used.
        var safe_flat_row = flat_row if row_in_bounds else 0
        var bias_vec = SIMD[dtype, width](0)
        comptime for i in range(width):
            var rel_dist = q_idx - (k_idx + i)
            var valid = row_in_bounds and 0 <= rel_dist < extent
            var safe_rel_dist = max(0, min(rel_dist, extent - 1))
            var raw_bias = self.bias.load[1](
                (safe_flat_row, coord[1], safe_rel_dist)
            )[0].cast[dtype]()
            bias_vec[i] = raw_bias if valid else Scalar[dtype](0)
        # Add the bias before the visibility select so masked lanes come out
        # as exactly `MASK_VALUE`, never `MASK_VALUE + bias`.
        return self.visibility_mask(coord, score_vec + bias_vec)

    comptime has_row_bias: Bool = True

    @inline(.always)
    def add_row_bias[
        BN: Int, //
    ](
        self,
        mut srow: Array[Float32, BN],
        seq_id: UInt32,
        head: UInt32,
        q_idx: UInt32,
        k_start: Int32,
    ):
        var shape = self.bias.shape()
        var extent = Int(shape[2].value())
        # Column `c` takes the bias at distance `hi - c`, so the row covers
        # distances `[lo, hi]` back to front.
        var hi = Int(q_idx) - Int(k_start)
        var lo = hi - (BN - 1)
        # Every key is in the future (masked anyway) or beyond the table.
        if extent == 0 or hi < 0 or lo >= extent:
            return
        var flat_row = self._flat_row(Int(seq_id), Int(q_idx))
        # Rows past the ragged total are tile padding; `mask()` gives them no
        # bias either.
        if not (0 <= flat_row < Int(shape[0].value())):
            return
        var h = Int(head)
        comptime if Self.Bias.dtype == DType.bfloat16:
            # Read the span as aligned bf16 pairs, `BN // 2 + 1` words, so a
            # funnel shift can realign them by one element when `lo` is odd.
            # A row starts at element `(flat_row * heads + head) * extent`, so
            # an even extent keeps every pair 4-byte aligned.
            var lo_even = lo & ~1
            if extent % 2 == 0 and lo_even >= 0 and lo_even + BN + 2 <= extent:
                var shift = UInt32(lo & 1) * 16
                var w = self._bias_word(flat_row, h, lo_even)
                comptime for m in range(BN // 2):
                    var w_next = self._bias_word(
                        flat_row, h, lo_even + 2 * m + 2
                    )
                    var pair = llvm_intrinsic[
                        "llvm.fshr", UInt32, has_side_effect=False
                    ](w_next, w, shift)
                    # The low half is distance `lo + 2m`, the high half the
                    # next one; a bf16 widens to f32 by shifting in zeros.
                    srow[BN - 1 - 2 * m] += bitcast[.float32](pair << 16)
                    srow[BN - 2 - 2 * m] += bitcast[.float32](pair & 0xFFFF0000)
                    w = w_next
                return
        # Spans that cross the table's edges, and other dtypes: clamp the read
        # so it stays in bounds, and drop the value when out of range. A
        # negative distance wraps to a large unsigned one.
        comptime for c in range(BN):
            var rel_dist = UInt32(hi - c)
            var value = self._bias_at(
                flat_row, h, Int(min(rel_dist, UInt32(extent - 1)))
            )
            srow[c] += value if rel_dist < UInt32(extent) else Float32(0)

    @inline(.always)
    def status(
        self,
        seq_id: UInt32,
        tile_offset: Coord[Int, Int],
        tile_size: Coord[Int, Int],
    ) -> TileMaskStatus:
        # The visibility mask owns the FULL_MASK boundary; downgrade its
        # NO_MASK (skip `mask()`) to PARTIAL_MASK, since the bias must be added
        # on every visible tile.
        var inner: TileMaskStatus
        comptime if Self.window_size > 0:
            inner = Self._window.status(seq_id, tile_offset, tile_size)
        else:
            inner = Self._causal.status(seq_id, tile_offset, tile_size)
        return (
            TileMaskStatus.PARTIAL_MASK if inner
            == TileMaskStatus.NO_MASK else inner
        )

    @inline(.always)
    def start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32) -> UInt32:
        comptime if Self.window_size > 0:
            return Self._window.start_column[BM, BN, page_size](seq_id, row)
        else:
            return Self._causal.start_column[BM, BN, page_size](seq_id, row)

    @staticmethod
    def start_column_alignment[BM: Int, BN: Int, page_size: Int]() -> Int:
        comptime if Self.window_size > 0:
            return type_of(Self._window).start_column_alignment[
                BM, BN, page_size
            ]()
        else:
            return CausalMask.start_column_alignment[BM, BN, page_size]()

    @inline(.always)
    def total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        comptime if Self.window_size > 0:
            return Self._window.total_iters[BM, BN, page_size](
                seq_id, row, num_cols
            )
        else:
            return Self._causal.total_iters[BM, BN, page_size](
                seq_id, row, num_cols
            )

    @staticmethod
    def count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1

    @inline(.always)
    def last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](seq_id, row, num_cols)

    @inline(.always)
    def masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, seq_id: UInt32, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {
            self.last_masked_set_end[BM, BN, page_size](seq_id, row, num_cols)
        }

    @staticmethod
    def nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        # One contiguous PARTIAL band when the visibility mask publishes a
        # known partition (the bias hits every visible tile); else UNKNOWN.
        comptime known = _nonfull_sets_known[
            type_of(Self._window), BM, BN
        ]() if Self.window_size > 0 else _nonfull_sets_known[
            CausalMask, BM, BN
        ]()
        comptime if known:
            return {TileMaskStatus.PARTIAL_MASK}
        else:
            return {TileMaskStatus.UNKNOWN_MASK}

    @staticmethod
    def mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS}

    @inline(.always)
    def mask_bits(
        self,
        seq_id: UInt32,
        score_row: Int32,
        col_start: Int32,
        num_keys: Int32,
    ) -> UInt32:
        # Unreachable: `mask_strategies` never advertises `BITMASK`.
        return UInt32(0xFFFF_FFFF)

    @staticmethod
    def sliding_window_size() -> Int:
        return max(Self.window_size, 0)
