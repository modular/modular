# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import iota, exp2
from sys import bitwidthof
from utils.index import IndexList


trait ScoreModTrait:
    """The ScoreMod trait desctribes score_mod for mha kernel like alibi bias.
    """

    fn score_mod[
        type: DType,
        width: Int, //,
        *,
        element_bitwidth: Int = bitwidthof[Int](),
        unsigned: Bool = False,
    ](
        self,
        coord: IndexList[
            4, element_bitwidth=element_bitwidth, unsigned=unsigned
        ],
        score_vec: SIMD[type, width],
    ) -> SIMD[type, width]:
        """Return score vector at given coordinates given a score_mod.

        Arguments:
          coord is (seq_id, head, q_idx, k_idx)
          score_vec is at `coord` of the score matrix

        Score_mod calculates a tensor given the functor and adds to score_vec.
        """
        ...


@value
@register_passable("trivial")  # No effect MOCO-1205
struct AlibiScoreMod(ScoreModTrait):
    """AlibiScoreMod adds the appropriate ALiBi constant bias to attention score.
    """

    # Number of heads; needed in ALiBi slopes calculation.
    var num_heads: UInt32

    @always_inline
    fn __init__(out self, num_heads: Int):
        self.num_heads = num_heads

    @always_inline
    fn _generate_alibi_bias[
        coords_dtype: DType,
        type: DType,
        width: Int,
    ](
        self,
        head_idx: SIMD[coords_dtype, width],
        q_idx: SIMD[coords_dtype, width],
        k_idx: SIMD[coords_dtype, width],
    ) -> SIMD[type, width]:
        var scale = exp2(
            -((head_idx + 1).cast[type]() * 8.0 // self.num_heads.cast[type]())
        )
        var bias = (q_idx - k_idx).cast[type]() * scale
        return bias

    @always_inline
    fn score_mod[
        type: DType,
        width: Int, //,
        *,
        element_bitwidth: Int = bitwidthof[Int](),
        unsigned: Bool = False,
    ](
        self,
        coord: IndexList[
            4, element_bitwidth=element_bitwidth, unsigned=unsigned
        ],
        score_vec: SIMD[type, width],
    ) -> SIMD[type, width]:
        var score_mod_vec = score_vec

        # coord[1] is the head index.
        # coord[2] and coord[3] are the token index in query and key respectively.

        alias coords_dtype = coord.element_type
        var head_idx = SIMD[coords_dtype, width](coord[1])
        var q_idx = SIMD[coords_dtype, width](coord[2])
        var k_idx = SIMD[coords_dtype, width](coord[3])

        # coords[2] >= coords[3] ensures the current tokens is only affected by
        # itself and previous tokens.
        score_mod_vec = (q_idx >= (k_idx + iota[coords_dtype, width]())).select(
            score_vec
            + self._generate_alibi_bias[coords_dtype, type, width](
                head_idx, q_idx, k_idx
            ),
            score_vec,
        )

        return score_mod_vec


@value
@register_passable("trivial")  # No effect MOCO-1205
struct IdentityScoreMod(ScoreModTrait):
    """IdentityScoreMod simply returns attention score."""

    @always_inline
    fn score_mod[
        type: DType,
        width: Int, //,
        *,
        element_bitwidth: Int = bitwidthof[Int](),
        unsigned: Bool = False,
    ](
        self,
        coord: IndexList[
            4, element_bitwidth=element_bitwidth, unsigned=unsigned
        ],
        score_vec: SIMD[type, width],
    ) -> SIMD[type, width]:
        return score_vec


@value
@register_passable("trivial")  # No effect MOCO-1205
struct AddFactorMod[factor: Float32](ScoreModTrait):
    """AddFactorMod adds a constant bias to attention score for q_idx >= k_idx.
    """

    @always_inline
    fn score_mod[
        type: DType,
        width: Int, //,
        *,
        element_bitwidth: Int = bitwidthof[Int](),
        unsigned: Bool = True,
    ](
        self,
        coord: IndexList[
            4, element_bitwidth=element_bitwidth, unsigned=unsigned
        ],
        score_vec: SIMD[type, width],
    ) -> SIMD[type, width]:
        var score_mod_vec = score_vec

        # coord[1] is the head index.
        # coord[2] and coord[3] are the token index in query and key respectively.

        alias coords_dtype = coord.element_type
        var q_idx = SIMD[coords_dtype, width](coord[2])
        var k_idx = SIMD[coords_dtype, width](coord[3])

        # coords[2] >= coords[3] ensures the current tokens is only affected by
        # itself and previous tokens.
        score_mod_vec = (q_idx >= (k_idx + iota[coords_dtype, width]())).select(
            score_vec + factor.cast[type](),
            score_vec,
        )

        return score_mod_vec
