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
# RUN: %mojo %s

from bit import pop_count
from hashlib._djbx33a import DJBX33A
from hashlib._hasher import _Hasher, _HashableWithHasher, _hash_with_hasher
from testing import assert_equal, assert_not_equal, assert_true
from memory import memset_zero, UnsafePointer
from test_utils import (
    dif_bits,
    gen_word_pairs,
    assert_dif_hashes,
    assert_fill_factor,
    words_ar,
    words_el,
    words_en,
    words_he,
    words_lv,
    words_pl,
    words_ru,
)


def test_hash_byte_array():
    assert_equal(
        _hash_with_hasher[HasherType=DJBX33A](String("a")),
        _hash_with_hasher[HasherType=DJBX33A](String("a")),
    )
    assert_equal(
        _hash_with_hasher[HasherType=DJBX33A](String("b")),
        _hash_with_hasher[HasherType=DJBX33A](String("b")),
    )

    assert_equal(
        _hash_with_hasher[HasherType=DJBX33A](String("c")),
        _hash_with_hasher[HasherType=DJBX33A](String("c")),
    )

    assert_equal(
        _hash_with_hasher[HasherType=DJBX33A](String("d")),
        _hash_with_hasher[HasherType=DJBX33A](String("d")),
    )
    assert_equal(
        _hash_with_hasher[HasherType=DJBX33A](String("d")),
        _hash_with_hasher[HasherType=DJBX33A](String("d")),
    )


def test_avalanche():
    # test that values which differ just in one bit,
    # produce significatly different hash values
    var buffer = InlineArray[UInt8, 256](fill=0)
    var hashes = List[UInt64]()
    hashes.append(
        _hash_with_hasher[HasherType=DJBX33A](buffer.unsafe_ptr(), 256)
    )

    for i in range(256):
        memset_zero(buffer.unsafe_ptr(), 256)
        var v = 1 << (i & 7)
        buffer[i >> 3] = v
        hashes.append(
            _hash_with_hasher[HasherType=DJBX33A](buffer.unsafe_ptr(), 256)
        )

    assert_dif_hashes(hashes, -1)


def test_trailing_zeros():
    # checks that a value with different amount of trailing zeros,
    # results in significantly different hash values
    var buffer = InlineArray[UInt8, 8](fill=0)
    buffer[0] = 23
    var hashes = List[UInt64]()
    for i in range(1, 9):
        hashes.append(
            _hash_with_hasher[HasherType=DJBX33A](buffer.unsafe_ptr(), i)
        )

    assert_dif_hashes(hashes, -1)


def test_fill_factor():
    # this algorithm is generally very unstable and returns different result on different archs
    words = gen_word_pairs[words_ar]()
    assert_fill_factor["AR", DJBX33A](words, len(words), 0.06)
    assert_fill_factor["AR", DJBX33A](words, len(words) // 2, 0.125)
    assert_fill_factor["AR", DJBX33A](words, len(words) // 4, 0.25)
    assert_fill_factor["AR", DJBX33A](words, len(words) // 14, 0.25)

    words = gen_word_pairs[words_el]()
    assert_fill_factor["EL", DJBX33A](words, len(words), 0.60)
    assert_fill_factor["EL", DJBX33A](words, len(words) // 2, 0.06)
    assert_fill_factor["EL", DJBX33A](words, len(words) // 4, 0.089)
    assert_fill_factor["EL", DJBX33A](words, len(words) // 13, 0.5)

    words = gen_word_pairs[words_en]()
    assert_fill_factor["EN", DJBX33A](words, len(words), 0.015)
    assert_fill_factor["EN", DJBX33A](words, len(words) // 2, 0.03)
    assert_fill_factor["EN", DJBX33A](words, len(words) // 4, 0.062)
    assert_fill_factor["EN", DJBX33A](words, len(words) // 14, 0.25)

    words = gen_word_pairs[words_he]()
    assert_fill_factor["HE", DJBX33A](words, len(words), 0.59)
    assert_fill_factor["HE", DJBX33A](words, len(words) // 2, 0.249)
    assert_fill_factor["HE", DJBX33A](words, len(words) // 4, 0.499)
    assert_fill_factor["HE", DJBX33A](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_lv]()
    assert_fill_factor["LV", DJBX33A](words, len(words), 0.242)
    assert_fill_factor["LV", DJBX33A](words, len(words) // 2, 0.485)
    assert_fill_factor["LV", DJBX33A](words, len(words) // 4, 0.971)
    assert_fill_factor["LV", DJBX33A](words, len(words) // 14, 0.062)

    words = gen_word_pairs[words_pl]()
    assert_fill_factor["PL", DJBX33A](words, len(words), 0.242)
    assert_fill_factor["PL", DJBX33A](words, len(words) // 2, 0.485)
    assert_fill_factor["PL", DJBX33A](words, len(words) // 4, 0.971)
    assert_fill_factor["PL", DJBX33A](words, len(words) // 14, 0.5)

    words = gen_word_pairs[words_ru]()
    assert_fill_factor["RU", DJBX33A](words, len(words), 0.607)
    assert_fill_factor["RU", DJBX33A](words, len(words) // 2, 0.038)
    assert_fill_factor["RU", DJBX33A](words, len(words) // 4, 0.066)
    assert_fill_factor["RU", DJBX33A](words, len(words) // 14, 1.0)


def test_hash_simd_values():
    fn hash(value: SIMD) -> UInt64:
        hasher = DJBX33A()
        hasher._update_with_simd(value)
        return hasher^.finish()

    assert_equal(hash(SIMD[DType.float16, 1](1.5)), 10240)
    assert_equal(hash(SIMD[DType.float32, 1](1.5)), 83886080)
    assert_equal(hash(SIMD[DType.float64, 1](1.5)), 6962565023914786816)
    assert_equal(hash(SIMD[DType.float16, 1](1)), 53248)
    assert_equal(hash(SIMD[DType.float32, 1](1)), 167772160)
    assert_equal(hash(SIMD[DType.float64, 1](1)), 4701758010974797824)

    assert_equal(hash(SIMD[DType.int8, 1](1)), 20)
    assert_equal(hash(SIMD[DType.int16, 1](1)), 31764)
    assert_equal(hash(SIMD[DType.int32, 1](1)), 2135587860)
    assert_equal(hash(SIMD[DType.int64, 1](1)), 11400714819323198484)
    assert_equal(hash(SIMD[DType.bool, 1](True)), 20)
    assert_equal(hash(SIMD[DType.int128, 1](1)), 11400714819323198484)
    assert_equal(hash(SIMD[DType.int64, 2](1, 0)), 0)
    assert_equal(hash(SIMD[DType.int256, 1](1)), 11400714819323198484)
    assert_equal(hash(SIMD[DType.int64, 4](1, 0, 0, 0)), 0)

    assert_equal(hash(SIMD[DType.int8, 1](-1)), 20)
    assert_equal(hash(SIMD[DType.int16, 1](-1)), 31764)
    assert_equal(hash(SIMD[DType.int32, 1](-1)), 2135587860)
    assert_equal(hash(SIMD[DType.int64, 1](-1)), 11400714819323198484)
    assert_equal(hash(SIMD[DType.int128, 1](-1)), 11400714819323198484)
    assert_equal(hash(SIMD[DType.int64, 2](-1)), 4387139453214858628)
    assert_equal(hash(SIMD[DType.int256, 1](-1)), 11400714819323198484)
    assert_equal(hash(SIMD[DType.int64, 4](-1)), 4736282921286787396)

    assert_equal(hash(SIMD[DType.int8, 1](0)), 0)
    assert_equal(hash(SIMD[DType.int8, 2](0)), 0)
    assert_equal(hash(SIMD[DType.int8, 4](0)), 0)
    assert_equal(hash(SIMD[DType.int8, 8](0)), 0)
    assert_equal(hash(SIMD[DType.int8, 16](0)), 0)
    assert_equal(hash(SIMD[DType.int8, 32](0)), 0)
    assert_equal(hash(SIMD[DType.int8, 64](0)), 0)

    assert_equal(hash(SIMD[DType.int32, 1](0)), 0)
    assert_equal(hash(SIMD[DType.int32, 2](0)), 0)
    assert_equal(hash(SIMD[DType.int32, 4](0)), 0)
    assert_equal(hash(SIMD[DType.int32, 8](0)), 0)
    assert_equal(hash(SIMD[DType.int32, 16](0)), 0)
    assert_equal(hash(SIMD[DType.int32, 32](0)), 0)
    assert_equal(hash(SIMD[DType.int32, 64](0)), 0)


def test_hash_at_compile_time():
    alias h = _hash_with_hasher[HasherType=DJBX33A](String("hello"))
    # can not do equality compare as the hash function is unstable on different platforms
    assert_true(h != 0)


def main():
    test_hash_byte_array()
    test_avalanche()
    test_trailing_zeros()
    test_fill_factor()
    test_hash_simd_values()
    test_hash_at_compile_time()
