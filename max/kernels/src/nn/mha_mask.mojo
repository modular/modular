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

from utils import StaticTuple
from math import iota, ceildiv
from sys import is_nvidia_gpu

from layout import LayoutTensor, Layout, UNKNOWN_VALUE
from memory import LegacyUnsafePointer
from collections import OptionalReg

comptime OpaquePointer = LegacyUnsafePointer[
    mut=True, NoneType, origin=MutAnyOrigin
]

from utils.index import IndexList, Index
from builtin.device_passable import DevicePassable

# ===-----------------------------------------------------------------------===#
# MaskName
# ===-----------------------------------------------------------------------===#


struct MaskName(Stringable):
    """A tile's masking status."""

    var name: String

    comptime NULL = Self("null")
    comptime CAUSAL = Self("causal")
    comptime CHUNKED = Self("chunked")
    comptime SLIDING_WINDOW_CAUSAL = Self("sliding_window_causal")
    comptime MATERIALIZED = Self("materialized")
    comptime CHUNKED_CAUSAL = Self("chunked_causal")

    fn __init__(out self, name: String):
        self.name = name

    fn __str__(self) -> String:
        return self.name

    fn __eq__(self, rhs: Self) -> Bool:
        return self.name == rhs.name

    fn __eq__(self, rhs: String) -> Bool:
        return self.name == rhs

    fn __ne__(self, rhs: Self) -> Bool:
        return self.name != rhs.name


# ===-----------------------------------------------------------------------===#
# TileMaskStatus
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct TileMaskStatus(
    Equatable,
    Identifiable,
    Stringable,
    TrivialRegisterPassable,
    Writable,
):
    """A tile's masking status."""

    var status: UInt8

    # No element is masked.
    comptime NO_MASK = Self(0)

    # Some elements in the tile are masked.
    comptime PARTIAL_MASK = Self(1)

    # All elements in the tile are masked.
    comptime FULL_MASK = Self(3)

    # Unkown mask -- a further check needed.
    # This is used by masks not yet supporting
    # the predefined trip count information.
    comptime UNKNOWN_MASK = Self(4)

    fn __eq__(self, rhs: Self) -> Bool:
        return self.status == rhs.status

    fn __ne__(self, rhs: Self) -> Bool:
        return self.status != rhs.status

    fn __is__(self, rhs: Self) -> Bool:
        return self.status == rhs.status

    fn __str__(self) -> String:
        return String.write(self)

    fn __and__(self, rhs: Self) -> Self:
        return Self(self.status & rhs.status)

    fn __or__(self, rhs: Self) -> Self:
        return Self(self.status | rhs.status)

    fn write_to(self, mut writer: Some[Writer]):
        if self is Self.NO_MASK:
            return writer.write("not masked")
        if self is Self.PARTIAL_MASK:
            return writer.write("partially masked")
        if self is Self.FULL_MASK:
            return writer.write("fully masked")
        writer.write("unknown mask")


struct MaskStrategy(TrivialRegisterPassable):
    var _value: Int32
    var _upper_triangular_window_size: Int32
    comptime NO_MASK = Self(0)
    """
    No mask is to be applied.
    """
    comptime LOWER_TRIANGULAR = Self(1)
    """
    Masks above the diagonal, e.g. a CausalMask or a SlidingWindowCausalMask.
    """
    comptime UPPER_TRIANGULAR = Self(2)
    """
    Masks below the diagonal, e.g. a SlidingWindowCausalMask.
    """
    comptime COMPUTED = Self(4)
    """
    Mask where the call operator must be used to compute the masked value.
    """
    comptime OUT_OF_BOUNDS = Self(8)
    """
    Check if we are out of bounds, e.g. at the end of a `NullMask` but unnecessary for a `CausalMask`.
    """

    @always_inline
    fn __init__(out self, value: Int32):
        self._value = value
        self._upper_triangular_window_size = 0

    @always_inline
    fn __init__(out self, value: Int32, window_size: Int32):
        self._value = value
        self._upper_triangular_window_size = window_size

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    @always_inline
    fn __and__(self, other: Self) -> Self:
        return {self._value & other._value}

    @always_inline
    fn __or__(self, other: Self) -> Self:
        return {self._value | other._value}

    @always_inline
    fn __contains__(self, other: Self) -> Bool:
        return (self._value | other._value) == self._value


# ===-----------------------------------------------------------------------===#
# MHAMask
# ===-----------------------------------------------------------------------===#


trait MHAMask(Copyable, DevicePassable, TrivialRegisterPassable):
    """The MHAMask trait describes masks for MHA kernels, such as the causal mask.
    """

    comptime apply_log2e_after_mask: Bool
    """
    Does the mask require `log2e` to be applied after the mask, or
    can it be fused with the scaling?
    """
    comptime mask_out_of_bound: Bool
    comptime mask_safe_out_of_bounds: Bool
    """
    Is the mask safe to read out of bounds?
    """
    comptime check_mask_during_decoding: Bool
    """
    Should we check the mask during decoding, or should we assume that it
    does not return `FULL_MASK`?
    """

    fn mask[
        dtype: DType, width: Int, //, *, element_type: DType = DType.uint32
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        """Return mask vector at given coordinates.

        Arguments:
          coord is (seq_id, head, q_idx, k_idx)
          score_vec is at `coord` of the score matrix

        The functor could capture an mask tensor and add to the score e.g. Replit.
        """
        ...

    fn status[
        *, element_type: DType = DType.uint32
    ](
        self,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        """Given a tile's index range, return its masking status."""
        ...

    fn start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32) -> UInt32:
        """
        Returns the first column for which this mask does not return
        `TileMaskStatus.FULL_MASK`.
        This may not be a multiple of `BN`, in which case iterating using
        `start_column` and `masked_set_ends` will not necessarilly produce
        the same set or number of iterations as iterating from
        `0` and checking `status` to skip.
        The return value of `total_iters` should be less than or equal to
        the number of non-skipped iterations.
        The practical consequence is that all warp group specializations
        within a kernel that loop over columns need to be in agreement.
        Either they all loop over all columns and check status to skip,
        or they loop using the `masked_set_ends`.
        """
        ...

    fn total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        """
        The total number of column iterations for which this mask returns either
        `TileMaskStatus.NO_MASK' or 'TileMaskStatus.PARTIAL_MASK'.
        This is to be used by warp specializations that do not need to
        use `kv_row`.
        """
        ...

    @staticmethod
    fn count_nonfull_sets(BM: Int, BN: Int) -> Int:
        """
        The number of blocks that are all partial-masks or not masked.
        """
        ...

    fn masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        """
        For each set of iterations in `nonfull_sets`, indicate the end idx
        belonging to that set (i.e., the last idx would be `end - 1`).
        Note that the final `masked_set_ends` may not necessarilly equal
        `total_iters`, if we have `UNKNOWN_MASK`s.
        In case of `UNKNOWN_MASK`s, `masked_set_ends` with tile-skipping
        must be used to have the correct kv_row values at each iteration.
        """
        ...

    fn last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        """
        Equivalent to `masked_set_ends[BM,BN,page_size](row, num_cols)[-1]`.
        """
        ...

    @staticmethod
    fn nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        """
        For each set of iterations that are either partially masked or not masked,
        this indicates the mask status.
        `UNKNOWN_MASK` here is an indicator meaning that we should check the status
        at runtime.
        It is semantically equivalent to `partial`, but with the optimization
        hint that it's worth checking on each iteration at runtime for
        `FULL_MASK` (in which case we can skip the tile) or `NO_MASK`
        (in which case we can unswitch and avoid masking in an inner loop).
        """
        ...

    @staticmethod
    fn mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        """
        For each set of iterations that are either partially masked or not masked,
        this indicates the `MaskStrategy` to use.
        """
        ...

    @staticmethod
    fn name() -> String:
        ...


# ===-----------------------------------------------------------------------===#
# CausalMask
# ===-----------------------------------------------------------------------===#

comptime MASK_VALUE = -10_000


@fieldwise_init
struct CausalMask(MHAMask, TrivialRegisterPassable):
    """MHA causal mask ensures a token is only affected by previous tokens."""

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = is_nvidia_gpu()
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = False

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "CausalMask"

    @staticmethod
    fn name() -> String:
        return "CausalMask"

    @always_inline
    fn mask[
        dtype: DType,
        width: Int,
        //,
        *,
        element_type: DType = DType.uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime index_type = coord.element_type

        # coord[2] and coord[3] are the token index in query and key respectively.
        var q_idx = coord[2]
        var k_idx = coord[3]

        # coords[2] >= coords[3] ensures the current tokens is only affected by
        # itself and previous tokens.
        # TODO(KERN-782): -10000 should be -inf but softmax saturates with NaNs.
        var masked_score_vec = (
            SIMD[index_type, width](q_idx).ge(
                iota[index_type, width](Scalar[index_type](k_idx))
            )
        ).select(score_vec, MASK_VALUE)

        return masked_score_vec

    @always_inline
    fn status[
        *, element_type: DType = DType.uint32
    ](
        self,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        # Consider tile corners
        #
        # 1
        # ^
        # C--------------D        A: (offset0,         offset1)
        # |              |        B: (offset0 + size0, offset1)
        # |              |        C: (offset0,         offset1 + size1)
        # |              |        D: (offset0 + size0, offset1 + size1)
        # A--------------B --> 0
        #
        # Key Points:
        #   * A is inside the tile but B, C, D are not.
        #   * If B is on or above the diagonal i.e. offset0 + size0 <= offset1
        #     the tile is fully masked.
        #   * If C is on or below the diagonal i.e. offset0 >= offset1 + size1 - 1
        #     the tile is not masked at all.

        # If false, the tile is not masked.
        var min_q_lt_max_k = (
            (tile_offset.data[0] + 1).lt(
                tile_offset.data[1] + tile_size.data[1]
            )
        ).cast[DType.uint8]()

        # If true, the tile is fully masked
        var max_q_lt_min_k = (
            (tile_offset.data[0] + tile_size.data[0]).le(tile_offset.data[1])
        ).cast[DType.uint8]()

        # Use 2 bits to represent:
        # (F, F) -> no mask
        # (T, F) -> partial mask
        # (T, T) -> full mask
        return TileMaskStatus(min_q_lt_max_k + (max_q_lt_min_k << 1))

    @always_inline
    fn start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32) -> UInt32:
        # offset0 is row
        # offset1 is col
        # B is (row + BM, col) -- means fully masked (we skip)
        # C is (row, col + BN) -- means not masked
        # causal mask is
        return 0

    @always_inline
    fn total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        # Masked off when row < col
        # row + BM - 1 < BN * x
        # We want the smallest value of `x` such that this is true
        # NOTE: by not checking `num_cols`, this method assumes that
        # `seq_len <= cache_len`.
        # If `seq_len` (and thus `rows`) may be larger than
        # `cache_len` (and thus `num_cols`), this will return the wrong result.
        return ceildiv(min(row + UInt32(BM), num_cols), UInt32(BN))

    @staticmethod
    fn count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 2

    @always_inline
    fn last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](row, num_cols)

    @always_inline
    fn masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        # Unmasked when row >= col
        # We want to find the maximum `x` such that
        # row >= x*BN - 1
        # Each col tile extends from col...col+BN-1,
        # so iter `i` (starting from 0) has maximum value
        # `i*BN + BN - 1 = (i+1)*BN - 1`.
        # Thus, iter `i` is fulle <= row if `row >= (i + 1)*BN - 1`.
        # `x`, the number of unmasked iters, is thus
        # x = i+1 = (row + 1) // BN
        num_unmasked = (row + 1) // UInt32(BN)
        partial_mask_end = self.total_iters[BM, BN, page_size](row, num_cols)
        return {num_unmasked, partial_mask_end}

    @staticmethod
    fn nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.NO_MASK, TileMaskStatus.PARTIAL_MASK}

    @staticmethod
    fn mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.NO_MASK, MaskStrategy.LOWER_TRIANGULAR}


# ===-----------------------------------------------------------------------===#
# NullMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct NullMask(MHAMask, TrivialRegisterPassable):
    """Mask that's effectively a noop."""

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = False

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "NullMask"

    @staticmethod
    fn name() -> String:
        return "NullMask"

    @always_inline
    fn mask[
        dtype: DType, width: Int, //, *, element_type: DType = DType.uint32
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        return score_vec

    @always_inline
    fn status[
        *, element_type: DType = DType.uint32
    ](
        self,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        # no mask
        return TileMaskStatus.NO_MASK

    @always_inline
    fn start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32) -> UInt32:
        return 0

    @always_inline
    fn total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        """
        The total number of column iterations for which this mask returns either
        `TileMaskStatus.NO_MASK' or 'TileMaskStatus.PARTIAL_MASK'.
        """
        return ceildiv(num_cols, UInt32(BN))

    @always_inline
    fn last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](row, num_cols)

    @staticmethod
    fn count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1

    @always_inline
    fn masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {self.total_iters[BM, BN, page_size](row, num_cols)}

    @staticmethod
    fn nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.NO_MASK}

    @staticmethod
    fn mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.OUT_OF_BOUNDS}


# ===-----------------------------------------------------------------------===#
# ChunkedMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct ChunkedMask[local_window_size: Int](MHAMask, TrivialRegisterPassable):
    """Mask implementing Chunked attention.

    This groups the mask into chunks of size `local_window_size`.
    Considering the following case:
    - Q_len = 7
    - K_len = 10
    - local_window_size = 4

    The mask will be applied as follows:
        K > 0 1 2 3 4 5 6 7 8 9
        Q v x--------------------x
        0 | 1 1 1 1 0 0 0 0 0 0
        1 | 0 0 0 0 1 1 1 1 0 0
        2 | 0 0 0 0 1 1 1 1 0 0
        3 | 0 0 0 0 1 1 1 1 0 0
        4 | 0 0 0 0 1 1 1 1 0 0
        5 | 0 0 0 0 0 0 0 0 1 1
        6 | 0 0 0 0 0 0 0 0 1 1
    """

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = True

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "ChunkedMask"

    @staticmethod
    fn name() -> String:
        return "ChunkedMask[" + String(Self.local_window_size) + "]"

    @always_inline
    fn mask[
        dtype: DType,
        width: Int,
        //,
        *,
        element_type: DType = DType.uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime assert (
            width <= Self.local_window_size
        ), "SIMD width of chunked mask must be <= local window size"

        var k_start_idx = coord.data[3]
        var k_end_idx = k_start_idx + Scalar[element_type](width) - 1

        q_chunk_idx = Int(
            coord.data[2] // Scalar[element_type](Self.local_window_size)
        )
        k_start_chunk_idx = Int(k_start_idx) // Self.local_window_size
        k_end_chunk_idx = Int(k_end_idx) // Self.local_window_size

        if q_chunk_idx == k_start_chunk_idx == k_end_chunk_idx:
            # fully unmasked, return the value
            return score_vec

        elif q_chunk_idx == k_start_chunk_idx or q_chunk_idx == k_end_chunk_idx:
            # partial mask
            var retval = score_vec
            var boundary = UInt32(
                (k_start_idx + Scalar[element_type](Self.local_window_size) - 1)
                // Scalar[element_type](Self.local_window_size)
            ) * UInt32(Self.local_window_size)

            var mask_val = SIMD[DType.bool, width](fill=False)
            var k_indices = (
                k_start_idx.cast[DType.uint32]() + iota[DType.uint32, width]()
            )
            if q_chunk_idx == k_start_chunk_idx:
                mask_val = k_indices.ge(boundary)
            elif q_chunk_idx == k_end_chunk_idx:
                mask_val = k_indices.lt(boundary)

            return mask_val.select(SIMD[dtype, width](MASK_VALUE), retval)

        # fully masked
        return SIMD[dtype, width](MASK_VALUE)

    @always_inline
    fn status[
        *, element_type: DType = DType.uint32
    ](
        self,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        var q_start_window = tile_offset[0] // Self.local_window_size
        var q_end_window = (
            tile_offset[0] + tile_size[0] - 1
        ) // Self.local_window_size
        var k_start_window = tile_offset[1] // Self.local_window_size
        var k_end_window = (
            tile_offset[1] + tile_size[1] - 1
        ) // Self.local_window_size

        var overlapping_windows = (
            k_end_window >= q_start_window and q_end_window >= k_start_window
        )

        if q_start_window == k_start_window == k_end_window == q_end_window:
            return TileMaskStatus.NO_MASK
        elif overlapping_windows:
            return TileMaskStatus.PARTIAL_MASK
        else:
            return TileMaskStatus.FULL_MASK

    @always_inline
    fn start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32) -> UInt32:
        # First column for which `row` is not masked is
        var col: UInt32 = (row // UInt32(Self.local_window_size)) * UInt32(
            Self.local_window_size
        )
        comptime align_to = min(page_size, BN)

        comptime if align_to == 1:
            return col
        else:
            return (col // UInt32(align_to)) * UInt32(align_to)

    @always_inline
    fn total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        start_col = self.start_column[BM, BN, page_size](row)
        # end_col is 1 past the end, the first that is masked off
        end_col = (
            1 + ((row + UInt32(BM) - 1) // UInt32(Self.local_window_size))
        ) * UInt32(Self.local_window_size)
        return ceildiv(end_col - start_col, UInt32(BN))

    @staticmethod
    fn count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1  # TODO: 3, for large chunk size

    @always_inline
    fn last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](row, num_cols)

    @always_inline
    fn masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {self.total_iters[BM, BN, page_size](row, num_cols)}

    @staticmethod
    fn nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.PARTIAL_MASK}

    @staticmethod
    fn mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS}


# ===-----------------------------------------------------------------------===#
# SlidingWindowCausalMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct SlidingWindowCausalMask[window_size: Int](
    MHAMask, TrivialRegisterPassable
):
    """Mask implementing Sliding Window attention.

    Considering the following case:
    - Q_len = 7
    - K_len = 7
    - window_size = 3

    The mask will be applied as follows:
        K > 0 1 2 3 4 5 6
        Q v x------------x
        0 | 1 0 0 0 0 0 0
        1 | 1 1 0 0 0 0 0
        2 | 1 1 1 0 0 0 0
        3 | 0 1 1 1 0 0 0
        4 | 0 0 1 1 1 0 0
        5 | 0 0 0 1 1 1 0
        6 | 0 0 0 0 1 1 1
    """

    comptime apply_log2e_after_mask: Bool = False
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = True
    comptime check_mask_during_decoding: Bool = True

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "SlidingWindowCausalMask"

    @staticmethod
    fn name() -> String:
        return "SlidingWindowCausalMask[" + String(Self.window_size) + "]"

    @always_inline
    fn mask[
        dtype: DType,
        width: Int,
        *,
        element_type: DType = DType.uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime index_type = coord.element_type

        comptime assert (
            width <= Self.window_size
        ), "SIMD width of sliding window mask must be <= window size"

        var q_idx = coord[2]
        var k_idx = coord[3]

        # first, check if the query is after the key, this step is the same
        # as the causal mask
        var masked_score_vec = (
            SIMD[index_type, width](q_idx).ge(
                iota[index_type, width](Scalar[index_type](k_idx))
            )
        ).select(score_vec, MASK_VALUE)

        # second, check if the query is within the window size of the key
        # It looks like we will encounter a problem here when the q_idx is
        # smaller than k_idx, but this is not possible because of the causal mask
        # that we have applied.
        return (
            (
                SIMD[index_type, width](q_idx)
                - iota[index_type, width](Scalar[index_type](k_idx))
            )
            .lt(Scalar[index_type](Self.window_size))
            .select(masked_score_vec, SIMD[dtype, width](MASK_VALUE))
        )

    @always_inline
    fn status[
        *, element_type: DType = DType.uint32
    ](
        self,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        # --- Check for FULL_MASK scenarios ---

        # Case 1: If the entire tile is too far to the right
        # (all query positions come before all key positions)
        var query_ends_before_keys_begin = (
            tile_offset.data[0] + tile_size.data[0] <= tile_offset.data[1]
        )

        # Case 2: If the entire tile is too far to the left
        # (all query positions are more than window_size away from all key positions)
        # Rewrite the inequality to use only addition so that we never subtract
        # `window_size` from an unsigned value (which can underflow).
        # Original condition:
        #     q_start - window_size + 1 >= k_start + k_size
        # is equivalent to:
        #     q_start + 1 >= k_start + k_size + window_size
        # where
        #     q_start = tile_offset[0]
        #     k_start = tile_offset[1]
        #     k_size  = tile_size[1]
        # Hence we compare two *added* terms, avoiding any risk of wrapping
        # around zero.

        var lhs = tile_offset.data[0] + 1
        var rhs = (
            tile_offset.data[1]
            + tile_size.data[1]
            + Scalar[element_type](Self.window_size)
        )
        var queries_too_far_ahead_of_keys = lhs >= rhs

        if query_ends_before_keys_begin or queries_too_far_ahead_of_keys:
            return TileMaskStatus.FULL_MASK

        # --- Check for NO_MASK scenario ---

        # Two conditions must BOTH be true for the tile to have no masks:

        # Condition 1: The earliest query position must be after the latest key position
        # (diagonal condition)
        var min_query_after_max_key = (
            tile_offset.data[0] >= tile_offset.data[1] + tile_size.data[1] - 1
        )

        # Condition 2: The latest query position must be within the window range of the
        # earliest key position
        var max_query_within_window_of_min_key = tile_offset.data[
            0
        ] + tile_size.data[0] - 1 < tile_offset.data[1] + Scalar[element_type](
            Self.window_size
        )

        if min_query_after_max_key and max_query_within_window_of_min_key:
            return TileMaskStatus.NO_MASK

        # If we reached here, some positions are masked and others aren't
        return TileMaskStatus.PARTIAL_MASK

    @always_inline
    fn start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32) -> UInt32:
        var col: UInt32 = UInt32(
            max(Int32(row) - Int32(Self.window_size) + 1, 0)
        )

        comptime if page_size <= 1:
            return col
        else:
            comptime align_to = min(page_size, BN)

            comptime if align_to == 1:
                return col
            else:
                return (col // UInt32(align_to)) * UInt32(align_to)

    @always_inline
    fn total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        start_col = self.start_column[BM, BN, page_size](row)
        end_col = min(row + UInt32(BM), num_cols)  # one past end
        return ceildiv(end_col - start_col, UInt32(BN))

    @staticmethod
    fn count_nonfull_sets(BM: Int, BN: Int) -> Int:
        if ((Self.window_size) // BN) > ((BM + BN - 2) // BN):
            return 3
        else:
            return 1

    @always_inline
    fn masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        start_col = self.start_column[BM, BN, page_size](row)
        # partial_exit_end_col = row + BM
        partial_exit_end_col = min(row + UInt32(BM), num_cols)
        # partial's end uses `ceildiv` and unmasked uses floored division
        # Partials must cover the entire `BN` tile with an masked entry
        # The unmasked region can't handle a tile with any
        #
        # (unmasked_end_col - start_col) // BN -
        #     ceildiv(partial_entry_end_col - start_col, BN)
        #
        # = (unmasked_end_col - start_col) // BN -
        #       (partial_entry_end_col - start_col + BN - 1) // BN
        # = (row + 1 - row + window_size - 1) // BN -
        #       (row + BM - window_size - row + window_size - 1 + BN - 1) // BN
        # = (window_size) // BN - (BM + BN - 2) // BN
        #
        # Thus, we will have unmasked iters when
        # (window_size) // BN - (BM + BN - 2) // BN > 0
        #
        end_tile = ceildiv(partial_exit_end_col - start_col, UInt32(BN))

        comptime if ((Self.window_size) // BN) > ((BM + BN - 2) // BN):
            # the partial entry region ends when row + BM - 1 is unmasked
            var partial_entry_end_col: UInt32 = row + UInt32(BM)
            if partial_entry_end_col > UInt32(Self.window_size):
                partial_entry_end_col -= UInt32(Self.window_size)
            else:
                partial_entry_end_col = 0
            unmasked_end_col = row + 1
            return {
                ceildiv(partial_entry_end_col - start_col, UInt32(BN)),
                (unmasked_end_col - start_col) // UInt32(BN),
                end_tile,
            }
        else:
            return {end_tile}

    @always_inline
    fn last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        return self.total_iters[BM, BN, page_size](row, num_cols)

    @staticmethod
    fn nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        comptime if (((Self.window_size) // BN) - ((BM + BN - 2) // BN)) > 0:
            return {
                TileMaskStatus.PARTIAL_MASK,
                TileMaskStatus.NO_MASK,
                TileMaskStatus.PARTIAL_MASK,
            }
        else:
            return {
                TileMaskStatus.PARTIAL_MASK,
            }

    @staticmethod
    fn mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        comptime if (((Self.window_size) // BN) - ((BM + BN - 2) // BN)) > 0:
            return {
                MaskStrategy(
                    MaskStrategy.UPPER_TRIANGULAR._value,
                    Int32(Self.window_size),
                ),
                MaskStrategy.NO_MASK,
                MaskStrategy.LOWER_TRIANGULAR,
            }
        else:
            return {
                MaskStrategy(
                    (
                        MaskStrategy.UPPER_TRIANGULAR
                        | MaskStrategy.LOWER_TRIANGULAR
                    )._value,
                    Int32(Self.window_size),
                )
            }


# ===-----------------------------------------------------------------------===#
# MaterializedMask
# ===-----------------------------------------------------------------------===#


@always_inline
fn naively_compute_total_iters[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType, q_row: UInt32, end: UInt32) -> UInt32:
    var iter_count: UInt32 = 0
    var kv_row: UInt32 = 0
    while kv_row < end:
        iter_count += UInt32(
            Int(
                mask.status(
                    Index[dtype = DType.int32](Int(q_row), Int(kv_row)),
                    Index[dtype = DType.int32](BM, BN),
                )
                != TileMaskStatus.FULL_MASK
            )
        )
        kv_row += UInt32(BN)
    return iter_count


@always_inline
fn naively_get_first_nonempty_mask_col[
    MaskType: MHAMask, //, BM: Int, BN: Int
](mask: MaskType, q_row: UInt32) -> UInt32:
    var kv_row: UInt32 = 0
    while (
        mask.status(
            Index[dtype = DType.int32](Int(q_row), Int(kv_row)),
            Index[dtype = DType.int32](BM, BN),
        )
        == TileMaskStatus.FULL_MASK
    ):
        kv_row += UInt32(BN)
    return kv_row


struct MaterializedMask[dtype_: DType, layout_: Layout](
    MHAMask, TrivialRegisterPassable
):
    """Mask that's backed by a materialized tensor."""

    comptime apply_log2e_after_mask: Bool = True
    comptime mask_out_of_bound: Bool = True
    comptime mask_safe_out_of_bounds: Bool = False
    comptime check_mask_during_decoding: Bool = True

    comptime MaskType = LayoutTensor[Self.dtype_, Self.layout_, ImmutAnyOrigin]
    var mask_tensor: Self.MaskType
    var start_pos: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ]
    var is_multiple_of_2: Bool

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "MaterializedMask"

    @staticmethod
    fn name() -> String:
        return "MaterializedMask"

    fn __init__(
        out self,
        mask_tensor: Self.MaskType,
        start_pos: OptionalReg[
            LayoutTensor[
                DType.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
            ]
        ] = None,
    ):
        comptime assert Self.layout_.rank() in (
            3,
            4,
        ), "Expected rank 3 or 4 for mask tensor"
        self.mask_tensor = mask_tensor
        self.start_pos = start_pos
        self.is_multiple_of_2 = (
            self.mask_tensor.dim[Self.layout_.rank() - 1]() % 2 == 0
        )

    @always_inline
    fn get_start_pos(self, batch_idx: Int) -> Int:
        if self.start_pos:
            return Int(self.start_pos.value()[batch_idx])
        else:
            return (
                self.mask_tensor.dim[Self.layout_.rank() - 1]()
                - self.mask_tensor.dim[Self.layout_.rank() - 2]()
            )

    @always_inline
    fn mask[
        dtype: DType,
        width: Int,
        //,
        *,
        element_type: DType = DType.uint32,
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime IndexListType = IndexList[
            Self.layout_.rank(), element_type=element_type
        ]
        var adjusted_coord: IndexListType

        var start_pos = self.get_start_pos(coord[0])

        comptime if Self.layout_.rank() == 3:
            adjusted_coord = IndexListType(
                coord[0], coord[2] - start_pos, coord[3]
            )
        else:
            adjusted_coord = IndexListType(
                coord[0], coord[1], coord[2] - start_pos, coord[3]
            )

        var retval = SIMD[dtype, width](MASK_VALUE)
        comptime rank = Self.layout_.rank()
        if adjusted_coord[rank - 2] < self.mask_tensor.dim[rank - 2]():
            if (
                adjusted_coord[rank - 1] + width
                <= self.mask_tensor.dim[rank - 1]()
                and self.is_multiple_of_2
            ):
                retval = self.mask_tensor.load[width=width](
                    adjusted_coord.canonicalize()
                ).cast[dtype]()
            elif adjusted_coord[rank - 1] < self.mask_tensor.dim[rank - 1]():
                for i in range(
                    min(
                        width,
                        self.mask_tensor.dim[rank - 1]() - coord[3],
                    )
                ):
                    adjusted_coord[rank - 1] = coord[3] + i
                    retval[i] = self.mask_tensor.load[width=1](
                        adjusted_coord.canonicalize()
                    ).cast[dtype]()

        return score_vec + retval

    @always_inline
    fn status[
        *, element_type: DType = DType.uint32
    ](
        self,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        # This is counter-intuitive but setting to `partial` ensures we
        # always read the values for the tensor.
        return TileMaskStatus.PARTIAL_MASK

    @always_inline
    fn start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32) -> UInt32:
        return naively_get_first_nonempty_mask_col[BM, BN](self, row)

    @always_inline
    fn total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        return naively_compute_total_iters[BM, BN](self, row, num_cols)

    @staticmethod
    fn count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1

    @always_inline
    fn last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        return ceildiv(num_cols, UInt32(BN))

    @always_inline
    fn masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {self.last_masked_set_end[BM, BN, page_size](row, num_cols)}

    @staticmethod
    fn nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.UNKNOWN_MASK}

    @staticmethod
    fn mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS}


# ===-----------------------------------------------------------------------===#
# AndMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct AndMask[T: MHAMask, S: MHAMask, //, lhs: T, rhs: S](
    MHAMask, TrivialRegisterPassable
):
    """Mask that's the AND of two masks."""

    comptime apply_log2e_after_mask: Bool = Self.T.apply_log2e_after_mask or Self.S.apply_log2e_after_mask
    comptime mask_out_of_bound: Bool = Self.T.mask_out_of_bound or Self.S.mask_out_of_bound
    comptime mask_safe_out_of_bounds: Bool = Self.T.mask_safe_out_of_bounds and Self.S.mask_safe_out_of_bounds
    comptime check_mask_during_decoding: Bool = Self.T.check_mask_during_decoding and Self.S.check_mask_during_decoding

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "AndMask"

    @staticmethod
    fn name() -> String:
        return "AndMask[" + Self.T.name() + ", " + Self.S.name() + "]"

    @always_inline
    fn mask[
        dtype: DType, width: Int, //, *, element_type: DType = DType.uint32
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime if dtype == DType.bool or dtype.is_integral():
            return self.lhs.mask(coord, score_vec) & self.rhs.mask(
                coord, score_vec
            )

        else:
            return min(
                self.lhs.mask(coord, score_vec),
                self.rhs.mask(coord, score_vec),
            )

    @always_inline
    fn status[
        *, element_type: DType = DType.uint32
    ](
        self,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        var lhs_status = self.lhs.status(tile_offset, tile_size)
        var rhs_status = self.rhs.status(tile_offset, tile_size)

        return lhs_status & rhs_status

    @always_inline
    fn start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32) -> UInt32:
        return naively_get_first_nonempty_mask_col[BM, BN](self, row)

    @always_inline
    fn total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        return naively_compute_total_iters[BM, BN](self, row, num_cols)

    @staticmethod
    fn count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1

    @always_inline
    fn last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        return ceildiv(num_cols, UInt32(BN))

    @always_inline
    fn masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {self.last_masked_set_end[BM, BN, page_size](row, num_cols)}

    @staticmethod
    fn nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.UNKNOWN_MASK}

    @staticmethod
    fn mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS}


# ===-----------------------------------------------------------------------===#
# OrMask
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct OrMask[T: MHAMask, S: MHAMask, //, lhs: T, rhs: S](
    MHAMask, TrivialRegisterPassable
):
    """Mask that's the OR of two masks."""

    comptime apply_log2e_after_mask: Bool = Self.T.apply_log2e_after_mask or Self.S.apply_log2e_after_mask
    comptime mask_out_of_bound: Bool = Self.T.mask_out_of_bound and Self.S.mask_out_of_bound
    comptime mask_safe_out_of_bounds: Bool = Self.T.mask_safe_out_of_bounds and Self.S.mask_safe_out_of_bounds
    comptime check_mask_during_decoding: Bool = Self.T.check_mask_during_decoding or Self.S.check_mask_during_decoding

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "OrMask"

    @staticmethod
    fn name() -> String:
        return "OrMask[" + Self.T.name() + ", " + Self.S.name() + "]"

    @always_inline
    fn mask[
        dtype: DType, width: Int, //, *, element_type: DType = DType.uint32
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        comptime if dtype == DType.bool or dtype.is_integral():
            return self.lhs.mask(coord, score_vec) | self.rhs.mask(
                coord, score_vec
            )
        else:
            return min(
                self.lhs.mask(coord, score_vec),
                self.rhs.mask(coord, score_vec),
            )

    @always_inline
    fn status[
        *, element_type: DType = DType.uint32
    ](
        self,
        tile_offset: IndexList[2, element_type=element_type],
        tile_size: IndexList[2, element_type=element_type],
    ) -> TileMaskStatus:
        var lhs_status = self.lhs.status(tile_offset, tile_size)
        var rhs_status = self.rhs.status(tile_offset, tile_size)
        return lhs_status | rhs_status

    @always_inline
    fn start_column[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32) -> UInt32:
        return naively_get_first_nonempty_mask_col[BM, BN](self, row)

    @always_inline
    fn total_iters[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        return naively_compute_total_iters[BM, BN](self, row, num_cols)

    @staticmethod
    fn count_nonfull_sets(BM: Int, BN: Int) -> Int:
        return 1

    @always_inline
    fn last_masked_set_end[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> UInt32:
        return ceildiv(num_cols, UInt32(BN))

    @always_inline
    fn masked_set_ends[
        BM: Int, BN: Int, page_size: Int
    ](self, row: UInt32, num_cols: UInt32) -> StaticTuple[
        UInt32, Self.count_nonfull_sets(BM, BN)
    ]:
        return {self.last_masked_set_end[BM, BN, page_size](row, num_cols)}

    @staticmethod
    fn nonfull_sets[
        BM: Int, BN: Int
    ]() -> StaticTuple[TileMaskStatus, Self.count_nonfull_sets(BM, BN)]:
        return {TileMaskStatus.UNKNOWN_MASK}

    @staticmethod
    fn mask_strategies[
        BM: Int, BN: Int
    ]() -> StaticTuple[MaskStrategy, Self.count_nonfull_sets(BM, BN)]:
        return {MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS}


# ===-----------------------------------------------------------------------===#
# ChunkedCausalMask
# ===-----------------------------------------------------------------------===#


@always_inline
fn ChunkedCausalMask[
    local_window_size: Int
](out res: OrMask[CausalMask(), ChunkedMask[local_window_size]()]):
    """Mask implementing Chunked Causal attention for Llama4 models.

    This groups the mask into chunks of size `local_window_size` and performs causal
    attention within each local chunk. Considering the following case:
    - Q_len = 7
    - K_len = 10
    - start_pos = 3
    - local_window_size = 4

    The mask will be applied as follows:
        K > 0 1 2 3 4 5 6 7 8 9
        Q v x--------------------x
        0 | 1 1 1 1 0 0 0 0 0 0
        1 | 0 0 0 0 1 0 0 0 0 0
        2 | 0 0 0 0 1 1 0 0 0 0
        3 | 0 0 0 0 1 1 1 0 0 0
        4 | 0 0 0 0 1 1 1 1 0 0
        5 | 0 0 0 0 0 0 0 0 1 0
        6 | 0 0 0 0 0 0 0 0 1 1
    """
    res = {}
