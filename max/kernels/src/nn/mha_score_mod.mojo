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

from memory import LegacyUnsafePointer

comptime OpaquePointer = LegacyUnsafePointer[
    mut=True, NoneType, origin=MutAnyOrigin
]
from utils.index import IndexList
from builtin.device_passable import DevicePassable


trait ScoreModTrait(Copyable, DevicePassable, TrivialRegisterPassable):
    """The ScoreMod trait describes score modifiers for MHA kernels."""

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
