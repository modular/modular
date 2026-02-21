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

from math import exp2, iota

from bit import prev_power_of_two

from memory import LegacyUnsafePointer

comptime OpaquePointer = LegacyUnsafePointer[
    mut=True, NoneType, origin=MutAnyOrigin
]
from utils.index import IndexList
from builtin.device_passable import DevicePassable


trait ScoreModTrait(Copyable, DevicePassable, TrivialRegisterPassable):
    """The ScoreMod trait desctribes score_mod for mha kernel like alibi bias.
    """

    comptime name_str: String

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
struct AlibiScoreMod[
    num_heads: Int,
](ScoreModTrait, TrivialRegisterPassable):
    """AlibiScoreMod adds the appropriate ALiBi constant bias to attention score.
    """

    comptime name_str: String = "alibi"
    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "AlibiScoreMod"

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
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        var scale: SIMD[dtype, width]

        comptime if Self.num_heads.is_power_of_two():
            scale = exp2(
                -(
                    (head_idx + 1).cast[dtype]()
                    * 8.0
                    / Scalar[dtype](Self.num_heads)
                )
            )
        else:
            comptime floor_power_of_2 = prev_power_of_two(Self.num_heads)
            if head_idx[0] < Scalar[coords_dtype](floor_power_of_2):
                scale = exp2(
                    -(
                        (head_idx + 1).cast[dtype]()
                        * 8.0
                        / Scalar[dtype](floor_power_of_2)
                    )
                )
            else:
                scale = exp2(
                    -(
                        (
                            (head_idx - Scalar[coords_dtype](floor_power_of_2))
                            * 2
                            + 1
                        ).cast[dtype]()
                        * 8.0
                        / Scalar[dtype](floor_power_of_2 * 2)
                    )
                )
        var bias = -(
            Scalar[coords_dtype](max_prompt_len - 1)
            - k_idx
            - iota[coords_dtype, width]()
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

        comptime coords_dtype = coord.element_type
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
struct IdentityScoreMod(ScoreModTrait, TrivialRegisterPassable):
    """IdentityScoreMod simply returns attention score."""

    comptime name_str: String = "no_pos"

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "IdentityScoreMod"

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
