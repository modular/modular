# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import iota
from utils.index import StaticIntTuple
from utils.numerics import min_or_neg_inf


@value
@register_passable("trivial")
struct TileMaskStatus:
    """A tile's masking status."""

    var status: UInt

    # No element is masked.
    alias NO_MASK = Self(0)

    # Some elements in the tile are masked.
    alias PARTIAL_MASK = Self(1)

    # All elements in the tile are masked.
    alias FULL_MASK = Self(3)

    fn __eq__(self, rhs: Self) -> Bool:
        return self.status == rhs.status


trait MHAMask:
    """The MHAMask trait desctribes mask for mha kernel like causal mask."""

    fn mask[
        type: DType, width: Int
    ](self, coord: StaticIntTuple[4], score_vec: SIMD[type, width]) -> SIMD[
        type, width
    ]:
        """Return mask vector at given coordinates.

        Arguments:
          coord is (seq_id, head, q_idx, k_idx)
          score_vec is at `coord` of the score matrix

        The functor could capture an mask tensor and add to the score e.g. Replit.
        """
        ...

    fn status(
        self,
        tile_offset: StaticIntTuple[2],
        tile_size: StaticIntTuple[2],
    ) -> TileMaskStatus:
        """Given a tile' index range, return its masking status."""
        ...


@value
@register_passable("trivial")  # No effect MOCO-1205
struct CausalMask(MHAMask):
    """MHA causal mask ensures a token is only affected by previous tokens."""

    @always_inline
    fn mask[
        type: DType, width: Int
    ](self, coord: StaticIntTuple[4], score_vec: SIMD[type, width]) -> SIMD[
        type, width
    ]:
        var masked_score_vec = score_vec

        # coord[2] and coord[3] are the token index in query and key respectively.
        var q_idx = SIMD[DType.index, width](coord[2])
        var k_idx = SIMD[DType.index, width](coord[3])

        @parameter
        for i in range(width):
            # coords[2] >= coords[3] ensures the current tokens is only affected by
            # itself and previous tokens.
            masked_score_vec = (
                q_idx >= (k_idx + iota[DType.index, width]())
            ).select(score_vec, min_or_neg_inf[type]())

        return masked_score_vec

    @always_inline
    fn status(
        self,
        tile_offset: StaticIntTuple[2],
        tile_size: StaticIntTuple[2],
    ) -> TileMaskStatus:
        # If false, the tile is not masked.
        var min_q_lt_max_k = UInt(
            tile_offset[0] < (tile_offset[1] + tile_size[1])
        )

        # If true, the tile is fully masked
        var max_q_lt_min_k = UInt(
            tile_offset[0] + tile_size[0] < tile_offset[1]
        )

        # Use 2 bits to represent:
        # (F, F) -> no mask
        # (T, F) -> partial mask
        # (T, T) -> full mask
        return TileMaskStatus(min_q_lt_max_k + (max_q_lt_min_k << 1))


@value
@register_passable("trivial")  # No effect MOCO-1205
struct NullMask(MHAMask):
    """Mask that's effectively a noop."""

    @always_inline
    fn mask[
        type: DType, width: Int
    ](self, coord: StaticIntTuple[4], score_vec: SIMD[type, width]) -> SIMD[
        type, width
    ]:
        return score_vec

    @always_inline
    fn status(
        self,
        tile_offset: StaticIntTuple[2],
        tile_size: StaticIntTuple[2],
    ) -> TileMaskStatus:
        # no mask
        return TileMaskStatus(0)
