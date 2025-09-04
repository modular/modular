# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from math import exp2, iota

from bit import prev_power_of_two

from utils.index import IndexList


@register_passable("trivial")
trait ScoreModTrait(Copyable):
    """The ScoreMod trait desctribes score_mod for mha kernel like alibi bias.
    """

    alias name_str: String

    fn score_mod[
        dtype: DType, width: Int, //, *, element_type: DType = DType.int32
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
        max_prompt_len: Int = 0,
    ) -> SIMD[dtype, width]:
        """Return score vector at given coordinates given a score_mod.

        Arguments:
          coord is (seq_id, head, q_idx, k_idx)
          score_vec is at `coord` of the score matrix

        Score_mod calculates a tensor given the functor and adds to score_vec.
        """
        ...


@fieldwise_init
@register_passable("trivial")
struct AlibiScoreMod[
    num_heads: Int,
](ScoreModTrait):
    """AlibiScoreMod adds the appropriate ALiBi constant bias to attention score.
    """

    alias name_str: String = "alibi"

    @always_inline
    fn _generate_alibi_bias[
        coords_dtype: DType,
        dtype: DType,
        width: Int,
    ](
        self,
        head_idx: SIMD[coords_dtype, width],
        k_idx: SIMD[coords_dtype, width],
        max_prompt_len: Int,
    ) -> SIMD[dtype, width]:
        var scale: SIMD[dtype, width]

        @parameter
        if num_heads.is_power_of_two():
            scale = exp2(-((head_idx + 1).cast[dtype]() * 8.0 / num_heads))
        else:
            alias floor_power_of_2 = prev_power_of_two(num_heads)
            if head_idx[0] < floor_power_of_2:
                scale = exp2(
                    -((head_idx + 1).cast[dtype]() * 8.0 / floor_power_of_2)
                )
            else:
                scale = exp2(
                    -(
                        ((head_idx - floor_power_of_2) * 2 + 1).cast[dtype]()
                        * 8.0
                        / (floor_power_of_2 * 2)
                    )
                )
        var bias = -(
            max_prompt_len - 1 - k_idx - iota[coords_dtype, width]()
        ).cast[dtype]()
        var alibi_bias = bias * scale
        return alibi_bias

    @always_inline
    fn score_mod[
        dtype: DType, width: Int, //, *, element_type: DType = DType.int32
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
        max_prompt_len: Int,
    ) -> SIMD[dtype, width]:
        # coord[1] is the head index.
        # coord[2] and coord[3] are the token index in query and key respectively.

        alias coords_dtype = coord.element_type
        var head_idx = SIMD[coords_dtype, width](coord[1])
        var q_idx = SIMD[coords_dtype, width](coord[2])
        var k_idx = SIMD[coords_dtype, width](coord[3])

        # coords[2] >= coords[3] ensures the current tokens is only affected by
        # itself and previous tokens.
        var score_mod_vec = (
            q_idx.ge(k_idx + iota[coords_dtype, width]())
        ).select(
            score_vec
            + self._generate_alibi_bias[coords_dtype, dtype, width](
                head_idx, k_idx, max_prompt_len
            ),
            score_vec,
        )

        return score_mod_vec


@fieldwise_init
@register_passable("trivial")
struct IdentityScoreMod(ImplicitlyCopyable, Movable, ScoreModTrait):
    """IdentityScoreMod simply returns attention score."""

    alias name_str: String = "no_pos"

    @always_inline
    fn score_mod[
        dtype: DType, width: Int, //, *, element_type: DType = DType.int32
    ](
        self,
        coord: IndexList[4, element_type=element_type],
        score_vec: SIMD[dtype, width],
        max_prompt_len: Int = 0,
    ) -> SIMD[dtype, width]:
        return score_vec
