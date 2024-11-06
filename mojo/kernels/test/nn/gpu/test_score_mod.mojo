# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import iota, exp2
from nn.mha_score_mod import AlibiScoreMod, IdentityScoreMod, AddFactorMod
from testing import assert_equal
from utils.index import Index


fn generate_alibi_bias[
    type: DType,
    width: Int,
    num_heads: Int,
](
    head_idx: SIMD[DType.index, width],
    q_idx: SIMD[DType.index, width],
    k_idx: SIMD[DType.index, width],
) -> SIMD[type, width]:
    var scale = exp2(-((head_idx + 1).cast[type]() * 8.0 // num_heads))
    var bias = (q_idx - k_idx).cast[type]() * scale
    return bias


def test_alibi_score_mod():
    print("test_alibi_score_mod")
    alias type = DType.float32
    alias width = 4
    alias num_heads = 4

    var alibi_mod = AlibiScoreMod(num_heads)

    var head_idx = SIMD[DType.index, width](0)
    var q_idx = SIMD[DType.index, width](2)
    var k_idx = SIMD[DType.index, width](1)

    var score_vec = SIMD[type, width](0, 1, 2, 3)

    var reference = (q_idx >= (k_idx + iota[DType.index, width]())).select(
        score_vec
        + generate_alibi_bias[type, width, num_heads](head_idx, q_idx, k_idx),
        score_vec,
    )

    var result = alibi_mod.score_mod(
        Index(0, 0, 2, 1), SIMD[type, width](0, 1, 2, 3)
    )

    assert_equal(reference, result)


def test_identity_score_mod():
    print("test_identity_score_mod")
    alias type = DType.float32
    alias width = 4

    var identity_mod = IdentityScoreMod()
    var reference = SIMD[type, width](0, 1, 2, 3)
    var result = identity_mod.score_mod(
        Index(0, 0, 1, 2), SIMD[type, width](0, 1, 2, 3)
    )

    assert_equal(reference, result)


def test_add_factor_mod():
    print("test_add_factor_mod")
    alias type = DType.float32
    alias width = 4
    alias add_factor = 1500000

    var add_factor_mod = AddFactorMod[add_factor]()
    var q_idx = SIMD[DType.index, width](2)
    var k_idx = SIMD[DType.index, width](1)

    var score_vec = SIMD[type, width](0, 1, 2, 3)

    var reference = (q_idx >= (k_idx + iota[DType.index, width]())).select(
        score_vec + add_factor,
        score_vec,
    )
    var result = add_factor_mod.score_mod(
        Index(0, 0, 2, 1), SIMD[type, width](0, 1, 2, 3)
    )

    assert_equal(reference, result)


def main():
    test_alibi_score_mod()
    test_identity_score_mod()
    test_add_factor_mod()
