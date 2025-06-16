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
from builtin._location import __call_location
from hashlib._fnv1a import Fnv1a
from hashlib._hasher import _Hasher, _HashableWithHasher
from testing import assert_equal, assert_not_equal, assert_true
from memory import memset_zero
from test_utils import (
    gen_word_pairs,
    words_ar,
    words_el,
    words_en,
    words_he,
    words_lv,
    words_pl,
    words_ru,
)


def dif_bits(i1: UInt64, i2: UInt64) -> Int:
    return Int(pop_count(i1 ^ i2))


fn hash[
    HashableType: _HashableWithHasher, HasherType: _Hasher
](hashable: HashableType) -> UInt64:
    var hasher = HasherType()
    hasher.update(hashable)
    var value = hasher^.finish()
    return value


fn hash[HasherType: _Hasher](bytes: Span[UInt8, _], n: Int) -> UInt64:
    var hasher = HasherType()
    hasher._update_with_bytes(bytes.unsafe_ptr(), n)
    var value = hasher^.finish()
    return value


@always_inline
def assert_dif_hashes(hashes: List[UInt64], upper_bound: Int):
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            var diff = dif_bits(hashes[i], hashes[j])
            assert_true(
                diff > upper_bound,
                String("Index: {}:{}, diff between: {} and {} is: {}").format(
                    i, j, hashes[i], hashes[j], diff
                ),
                location=__call_location(),
            )


def test_hash_byte_array():
    assert_equal(
        hash[HasherType=Fnv1a](String("a")), hash[HasherType=Fnv1a](String("a"))
    )
    assert_equal(
        hash[HasherType=Fnv1a](String("b")), hash[HasherType=Fnv1a](String("b"))
    )

    assert_equal(
        hash[HasherType=Fnv1a](String("c")), hash[HasherType=Fnv1a](String("c"))
    )

    assert_equal(
        hash[HasherType=Fnv1a](String("d")), hash[HasherType=Fnv1a](String("d"))
    )
    assert_equal(
        hash[HasherType=Fnv1a](String("d")), hash[HasherType=Fnv1a](String("d"))
    )


def test_avalanche():
    # test that values which differ just in one bit,
    # produce significatly different hash values
    var buffer = InlineArray[UInt8, 256](fill=0)
    var hashes = List[UInt64]()
    hashes.append(hash[HasherType=Fnv1a](buffer, 256))

    for i in range(256):
        memset_zero(buffer.unsafe_ptr(), 256)
        var v = 1 << (i & 7)
        buffer[i >> 3] = v
        hashes.append(hash[HasherType=Fnv1a](buffer, 256))

    assert_dif_hashes(hashes, 15)


def test_trailing_zeros():
    # checks that a value with different amount of trailing zeros,
    # results in significantly different hash values
    var buffer = InlineArray[UInt8, 8](fill=0)
    buffer[0] = 23
    var hashes = List[UInt64]()
    for i in range(1, 9):
        hashes.append(hash[HasherType=Fnv1a](buffer, i))

    assert_dif_hashes(hashes, 21)


@always_inline
def assert_fill_factor[
    label: String
](words: List[String], num_buckets: Int, lower_bound: Float64):
    # A perfect hash function is when the number of buckets is equal to number of words
    # and the fill factor results in 1.0
    var buckets = List[Int](0) * num_buckets
    for w in words:
        var h = hash[HasherType=Fnv1a](w)
        buckets[h % num_buckets] += 1
    var unfilled = 0
    for v in buckets:
        if v == 0:
            unfilled += 1

    var fill_factor = 1 - unfilled / num_buckets
    assert_true(
        fill_factor >= lower_bound,
        String("Fill factor for {} is {}, provided lower bound was {}").format(
            label, fill_factor, lower_bound
        ),
        location=__call_location(),
    )


def test_fill_factor():
    words = gen_word_pairs[words_ar]()
    assert_fill_factor["AR"](words, len(words), 0.63)
    assert_fill_factor["AR"](words, len(words) // 2, 0.86)
    assert_fill_factor["AR"](words, len(words) // 4, 0.98)
    assert_fill_factor["AR"](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_el]()
    assert_fill_factor["EL"](words, len(words), 0.63)
    assert_fill_factor["EL"](words, len(words) // 2, 0.86)
    assert_fill_factor["EL"](words, len(words) // 4, 0.98)
    assert_fill_factor["EL"](words, len(words) // 13, 1.0)

    words = gen_word_pairs[words_en]()
    assert_fill_factor["EN"](words, len(words), 0.63)
    assert_fill_factor["EN"](words, len(words) // 2, 0.86)
    assert_fill_factor["EN"](words, len(words) // 4, 0.98)
    assert_fill_factor["EN"](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_he]()
    assert_fill_factor["HE"](words, len(words), 0.63)
    assert_fill_factor["HE"](words, len(words) // 2, 0.86)
    assert_fill_factor["HE"](words, len(words) // 4, 0.98)
    assert_fill_factor["HE"](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_lv]()
    assert_fill_factor["LV"](words, len(words), 0.63)
    assert_fill_factor["LV"](words, len(words) // 2, 0.86)
    assert_fill_factor["LV"](words, len(words) // 4, 0.98)
    assert_fill_factor["LV"](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_pl]()
    assert_fill_factor["PL"](words, len(words), 0.63)
    assert_fill_factor["PL"](words, len(words) // 2, 0.86)
    assert_fill_factor["PL"](words, len(words) // 4, 0.98)
    assert_fill_factor["PL"](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_ru]()
    assert_fill_factor["RU"](words, len(words), 0.63)
    assert_fill_factor["RU"](words, len(words) // 2, 0.86)
    assert_fill_factor["RU"](words, len(words) // 4, 0.98)
    assert_fill_factor["RU"](words, len(words) // 14, 1.0)


def test_hash_simd_values():
    fn hash(value: SIMD) -> UInt64:
        hasher = Fnv1a()
        hasher._update_with_simd(value)
        return hasher^.finish()

    assert_equal(hash(SIMD[DType.float16, 1](1.5)), 12636464265834235359)
    assert_equal(hash(SIMD[DType.float32, 1](1.5)), 8026467504136239071)
    assert_equal(hash(SIMD[DType.float64, 1](1.5)), 15000291120250992607)
    assert_equal(hash(SIMD[DType.float16, 1](1)), 12637027215787879391)
    assert_equal(hash(SIMD[DType.float32, 1](1)), 3414781483884328927)
    assert_equal(hash(SIMD[DType.float64, 1](1)), 14020758201297909727)

    assert_equal(hash(SIMD[DType.int8, 1](1)), 12638152016183539244)
    assert_equal(hash(SIMD[DType.int16, 1](1)), 12638152016183539244)
    assert_equal(hash(SIMD[DType.int32, 1](1)), 12638152016183539244)
    assert_equal(hash(SIMD[DType.int64, 1](1)), 12638152016183539244)
    assert_equal(hash(SIMD[DType.bool, 1](True)), 12638152016183539244)
    assert_equal(hash(SIMD[DType.int128, 1](1)), 589727492704079044)
    assert_equal(hash(SIMD[DType.int64, 2](1, 0)), 589727492704079044)
    assert_equal(hash(SIMD[DType.int256, 1](1)), 12478008331234465636)
    assert_equal(hash(SIMD[DType.int64, 4](1, 0, 0, 0)), 12478008331234465636)

    assert_equal(hash(SIMD[DType.int8, 1](-1)), 5808589858502755950)
    assert_equal(hash(SIMD[DType.int16, 1](-1)), 5808589858502755950)
    assert_equal(hash(SIMD[DType.int32, 1](-1)), 5808589858502755950)
    assert_equal(hash(SIMD[DType.int64, 1](-1)), 5808589858502755950)
    assert_equal(hash(SIMD[DType.int128, 1](-1)), 591639543425159523)
    assert_equal(hash(SIMD[DType.int64, 2](-1)), 591639543425159523)
    assert_equal(hash(SIMD[DType.int256, 1](-1)), 16463485165778154321)
    assert_equal(hash(SIMD[DType.int64, 4](-1)), 16463485165778154321)

    assert_equal(hash(SIMD[DType.int8, 1](0)), 12638153115695167455)
    assert_equal(hash(SIMD[DType.int8, 2](0)), 590684067820433389)
    assert_equal(hash(SIMD[DType.int8, 4](0)), 5558979605539197941)
    assert_equal(hash(SIMD[DType.int8, 8](0)), 12161962213042174405)
    assert_equal(hash(SIMD[DType.int8, 16](0)), 9808874869469701221)
    assert_equal(hash(SIMD[DType.int8, 32](0)), 901300984310592933)
    assert_equal(hash(SIMD[DType.int8, 64](0)), 13380826962402805797)

    assert_equal(hash(SIMD[DType.int32, 1](0)), 12638153115695167455)
    assert_equal(hash(SIMD[DType.int32, 2](0)), 590684067820433389)
    assert_equal(hash(SIMD[DType.int32, 4](0)), 5558979605539197941)
    assert_equal(hash(SIMD[DType.int32, 8](0)), 12161962213042174405)
    assert_equal(hash(SIMD[DType.int32, 16](0)), 9808874869469701221)
    assert_equal(hash(SIMD[DType.int32, 32](0)), 901300984310592933)
    assert_equal(hash(SIMD[DType.int32, 64](0)), 13380826962402805797)


def test_hash_at_compile_time():
    alias h = hash[HasherType=Fnv1a](String("hello"))
    assert_equal(h, 11831194018420276491)


def main():
    test_hash_byte_array()
    test_avalanche()
    test_trailing_zeros()
    test_fill_factor()
    test_hash_simd_values()
    test_hash_at_compile_time()
