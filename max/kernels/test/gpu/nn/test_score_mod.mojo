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
from nn.mha_score_mod import AlibiScoreMod, IdentityScoreMod
from testing import assert_equal

from utils.index import Index


fn generate_alibi_bias[
    dtype: DType,
    width: Int,
    num_heads: Int,
](
    head_idx: SIMD[DType.int, width],
    q_idx: SIMD[DType.int, width],
    k_idx: SIMD[DType.int, width],
    max_prompt_len: Int = 0,
) -> SIMD[dtype, width] where dtype.is_floating_point():
    var scale: SIMD[dtype, width]

    comptime if num_heads.is_power_of_two():
        scale = exp2(
            -((head_idx + 1).cast[dtype]() * 8.0 / Scalar[dtype](num_heads))
        )
    else:
        comptime floor_power_of_2 = prev_power_of_two(num_heads)
        if head_idx < Scalar[DType.int](floor_power_of_2):
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
                        (head_idx - Scalar[DType.int](floor_power_of_2)) * 2 + 1
                    ).cast[dtype]()
                    * 8.0
                    / Scalar[dtype](floor_power_of_2 * 2)
                )
            )
    # print(scale)
    var bias = (
        -(
            Scalar[DType.int](max_prompt_len - 1)
            - k_idx
            - iota[DType.int, width]()
        ).cast[dtype]()
        * scale
    )
    # print(bias)
    return bias


def test_alibi_score_mod():
    print("test_alibi_score_mod")
    comptime dtype = DType.float32
    comptime width = 4
    comptime num_heads = 4
    var max_seq_len = 12

    var alibi_mod = AlibiScoreMod[num_heads]()

    var head_idx = SIMD[DType.int, width](0)
    var q_idx = SIMD[DType.int, width](2)
    var k_idx = SIMD[DType.int, width](1)

    var score_vec = SIMD[dtype, width](0, 1, 2, 3)

    var reference = q_idx.ge(k_idx + iota[DType.int, width]()).select(
        score_vec
        + generate_alibi_bias[dtype, width, num_heads](
            head_idx,
            q_idx,
            k_idx,
            max_seq_len,
        ),
        score_vec,
    )

    var result = alibi_mod.score_mod(
        Index(0, 0, 2, 1), SIMD[dtype, width](0, 1, 2, 3), max_seq_len
    )

    assert_equal(reference, result)


def test_identity_score_mod():
    print("test_identity_score_mod")
    comptime dtype = DType.float32
    comptime width = 4

    var identity_mod = IdentityScoreMod()
    var reference = SIMD[dtype, width](0, 1, 2, 3)
    var result = identity_mod.score_mod(
        Index(0, 0, 1, 2), SIMD[dtype, width](0, 1, 2, 3)
    )

    assert_equal(reference, result)


def main():
    test_alibi_score_mod()
    test_identity_score_mod()
