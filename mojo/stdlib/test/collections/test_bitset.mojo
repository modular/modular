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

from collections import BitSet
from testing import assert_equal, assert_true, assert_false


def test_bitset_init():
    # Test default initialization
    var bs = BitSet[128]()
    assert_equal(len(bs), 0, msg="Empty BitSet should have length 0")
    assert_true(bs.is_empty(), msg="Empty BitSet should be empty")

    # Test with initial bits
    var bs2 = BitSet[64]()
    assert_equal(len(bs2), 0, msg="Empty BitSet should have length 0")


def test_bitset_set_test_clear():
    var bs = BitSet[128]()

    # Test setting bits
    bs.set(42)
    assert_equal(
        len(bs), 1, msg="BitSet length should be 43 after setting bit 42"
    )
    assert_true(bs.test(42), msg="Bit 42 should be set")
    assert_false(bs.test(41), msg="Bit 41 should not be set")

    # Test setting multiple bits
    bs.set(10)
    bs.set(100)
    assert_equal(
        len(bs), 3, msg="BitSet length should be 101 after setting bit 100"
    )
    assert_true(bs.test(10), msg="Bit 10 should be set")
    assert_true(bs.test(42), msg="Bit 42 should still be set")
    assert_true(bs.test(100), msg="Bit 100 should be set")

    # Test clearing bits
    bs.clear(42)
    assert_false(bs.test(42), msg="Bit 42 should be cleared")
    assert_true(bs.test(10), msg="Bit 10 should still be set")
    assert_true(bs.test(100), msg="Bit 100 should still be set")

    # Test clear_all
    bs.clear_all()
    assert_equal(len(bs), 0, msg="BitSet should be empty after clear_all")
    assert_false(bs.test(10), msg="Bit 10 should be cleared after clear_all")
    assert_false(bs.test(100), msg="Bit 100 should be cleared after clear_all")


def test_bitset_toggle():
    var bs = BitSet[64]()

    # Toggle bit from 0 to 1
    bs.toggle(5)
    assert_true(bs.test(5), msg="Bit 5 should be set after toggle")

    # Toggle bit from 1 to 0
    bs.toggle(5)
    assert_false(bs.test(5), msg="Bit 5 should be cleared after second toggle")

    # Toggle multiple bits
    bs.toggle(10)
    bs.toggle(20)
    bs.toggle(30)
    assert_true(bs.test(10), msg="Bit 10 should be set")
    assert_true(bs.test(20), msg="Bit 20 should be set")
    assert_true(bs.test(30), msg="Bit 30 should be set")
    assert_equal(len(bs), 3, msg="BitSet length should be 31")


def test_bitset_count():
    var bs = BitSet[256]()

    # Empty set should have count 0
    assert_equal(len(bs), 0, msg="Empty BitSet should have count 0")

    # Set some bits and check count
    bs.set(1)
    bs.set(10)
    bs.set(100)
    assert_equal(
        len(bs), 3, msg="BitSet should have count 3 after setting 3 bits"
    )

    # Clear a bit and check count
    bs.clear(10)
    assert_equal(
        len(bs), 2, msg="BitSet should have count 2 after clearing 1 bit"
    )

    # Clear all and check count
    bs.clear_all()
    assert_equal(len(bs), 0, msg="BitSet should have count 0 after clear_all")


def test_bitset_bounds():
    var bs = BitSet[32]()

    # Test valid operations
    bs.set(0)
    bs.set(31)
    assert_true(bs.test(0), msg="Bit 0 should be set")
    assert_true(bs.test(31), msg="Bit 31 should be set")


def test_bitset_str_repr():
    var bs = BitSet[16]()
    bs.set(1)
    bs.set(5)
    bs.set(10)

    var str_rep = String(bs)

    # Check that the string representations contain the set bits
    assert_true(
        "1" in str_rep, msg="String representation should contain bit 1"
    )
    assert_true(
        "5" in str_rep, msg="String representation should contain bit 5"
    )
    assert_true(
        "10" in str_rep, msg="String representation should contain bit 10"
    )
    assert_false(
        "0, " in str_rep, msg="String representation should not contain bit 0"
    )
    assert_false(
        "4, " in str_rep, msg="String representation should not contain bit 4"
    )
    assert_false(
        "16, " in str_rep, msg="String representation should not contain bit 16"
    )


def test_bitset_edge_cases():
    # Test with minimum size
    var bs_min = BitSet[1]()
    bs_min.set(0)
    assert_true(
        bs_min.test(0), msg="Bit 0 should be set in minimum size BitSet"
    )
    assert_equal(
        len(bs_min),
        1,
        msg="Count should be 1 for minimum size BitSet with one bit set",
    )

    # Test with size that spans multiple words
    var bs_multi = BitSet[128]()
    bs_multi.set(63)  # Last bit in first word
    bs_multi.set(64)  # First bit in second word
    assert_true(
        bs_multi.test(63), msg="Bit 63 should be set (last bit in first word)"
    )
    assert_true(
        bs_multi.test(64), msg="Bit 64 should be set (first bit in second word)"
    )
    assert_equal(
        len(bs_multi), 2, msg="Count should be 2 for bits in different words"
    )


def test_bitset_consecutive_operations():
    var bs = BitSet[64]()

    # Set and immediately clear
    bs.set(10)
    assert_true(bs.test(10), msg="Bit 10 should be set after set operation")
    bs.clear(10)
    assert_false(
        bs.test(10), msg="Bit 10 should be cleared after clear operation"
    )

    # Toggle twice to return to original state
    bs.toggle(20)
    bs.toggle(20)
    assert_false(
        bs.test(20), msg="Bit 20 should be cleared after double toggle"
    )

    # Set multiple bits and check count
    for i in range(0, 10):
        bs.set(i)
    assert_equal(len(bs), 10, msg="Count should be 10 after setting 10 bits")

    # Clear all and verify
    bs.clear_all()
    assert_equal(len(bs), 0, msg="Count should be 0 after clear_all")
    assert_true(bs.is_empty(), msg="BitSet should be empty after clear_all")


def test_bitset_word_boundaries():
    var bs = BitSet[128]()

    # Test bits at word boundaries (assuming 64-bit words)
    bs.set(63)  # Last bit of first word
    bs.set(64)  # First bit of second word

    assert_true(
        bs.test(63), msg="Bit 63 (last bit of first word) should be set"
    )
    assert_true(
        bs.test(64), msg="Bit 64 (first bit of second word) should be set"
    )

    # Toggle bits at boundaries
    bs.toggle(63)
    bs.toggle(64)

    assert_false(bs.test(63), msg="Bit 63 should be cleared after toggle")
    assert_false(bs.test(64), msg="Bit 64 should be cleared after toggle")

    # Set bits across multiple words and check count
    for i in range(60, 70):
        bs.set(i)

    assert_equal(
        len(bs),
        10,
        msg="Count should be 10 after setting bits across word boundary",
    )


def test_bitset_large_indices():
    var bs = BitSet[256]()

    # Test with larger indices
    bs.set(200)
    bs.set(250)

    assert_true(bs.test(200), msg="Bit 200 should be set")
    assert_true(bs.test(250), msg="Bit 250 should be set")
    assert_equal(
        len(bs), 2, msg="Count should be 2 after setting 2 high-index bits"
    )

    # Test string representation with large indices
    var str_rep = String(bs)
    assert_true(
        "200" in str_rep, msg="String representation should contain bit 200"
    )
    assert_true(
        "250" in str_rep, msg="String representation should contain bit 250"
    )


def test_bitset_simd_init():
    var bs1 = BitSet(SIMD[DType.bool, 128](True))
    assert_equal(len(bs1), 128, msg="BitSet count should be 128")

    var bs2 = BitSet(SIMD[DType.bool, 128](False))
    assert_equal(len(bs2), 0, msg="BitSet count should be 0")

    var bs3 = BitSet(SIMD[DType.bool, 4](True, False, True, False))
    assert_equal(len(bs3), 2, msg="BitSet count should be 2")


def test_bitset_len():
    # 1. Empty BitSet
    var bs = BitSet[128]()
    assert_equal(len(bs), 0, msg="Len: Empty BitSet should be 0")

    # 2. Single insertion
    bs.set(7)
    assert_equal(len(bs), 1, msg="Len: After setting one bit")

    # 3. Insertion across a word boundary (index 64)
    bs.set(64)
    assert_equal(len(bs), 2, msg="Len: Two bits set across word boundary")

    # 4. Toggle a set bit off
    bs.toggle(7)
    assert_equal(len(bs), 1, msg="Len: After toggling a bit off")

    # 5. Clear the remaining bit
    bs.clear(64)
    assert_equal(len(bs), 0, msg="Len: After clearing all bits")

    # 6. Bulk pattern insertion (every 3rd index)
    for i in range(128):
        if i % 3 == 0:
            bs.set(i)
    expected = 43  # floor(128 / 3) + 1
    assert_equal(len(bs), expected, msg="Len: Pattern insertion")


def main():
    test_bitset_init()
    test_bitset_set_test_clear()
    test_bitset_toggle()
    test_bitset_count()
    test_bitset_bounds()
    test_bitset_str_repr()
    test_bitset_edge_cases()
    test_bitset_consecutive_operations()
    test_bitset_word_boundaries()
    test_bitset_large_indices()
    test_bitset_simd_init()
    test_bitset_len()
