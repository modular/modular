# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import exp2
from nn.mha_score_mod import AlibiScoreMod, IdentityScoreMod
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

    var reference = generate_alibi_bias[type, width, num_heads](
        SIMD[DType.index, width](0),
        SIMD[DType.index, width](1),
        SIMD[DType.index, width](2),
    )
    reference = reference + SIMD[type, width](0, 1, 2, 3)
    var result = alibi_mod.score_mod(
        Index(0, 0, 1, 2), SIMD[type, width](0, 1, 2, 3)
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


def main():
    test_alibi_score_mod()
    test_identity_score_mod()
