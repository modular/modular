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

from testing import assert_equal, assert_false, assert_true


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


def test_bitset_union():
    # Basic case
    var bs1 = BitSet[128]()
    bs1.set(1)
    bs1.set(2)
    bs1.set(3)

    var bs2 = BitSet[128]()
    bs2.set(3)
    bs2.set(4)
    bs2.set(5)

    var bs3 = bs1.union(bs2)
    assert_equal(len(bs3), 5, msg="Union: Basic case count")
    assert_true(bs3.test(1), msg="Union: Basic case bit 1")
    assert_true(bs3.test(2), msg="Union: Basic case bit 2")
    assert_true(bs3.test(3), msg="Union: Basic case bit 3")
    assert_true(bs3.test(4), msg="Union: Basic case bit 4")
    assert_true(bs3.test(5), msg="Union: Basic case bit 5")

    # Union with empty set
    var bs_empty = BitSet[128]()
    var bs4 = bs1.union(bs_empty)
    assert_equal(len(bs4), 3, msg="Union: With empty set count")
    assert_true(bs4.test(1), msg="Union: With empty set bit 1")
    assert_true(bs4.test(2), msg="Union: With empty set bit 2")
    assert_true(bs4.test(3), msg="Union: With empty set bit 3")
    assert_false(bs4.test(4), msg="Union: With empty set bit 4")

    var bs5 = bs_empty.union(bs1)
    assert_equal(len(bs5), 3, msg="Union: Empty with non-empty set count")
    assert_true(bs5.test(1), msg="Union: Empty with non-empty set bit 1")
    assert_true(bs5.test(2), msg="Union: Empty with non-empty set bit 2")
    assert_true(bs5.test(3), msg="Union: Empty with non-empty set bit 3")

    # Union of identical sets
    var bs6 = bs1.union(bs1)
    assert_equal(len(bs6), 3, msg="Union: Identical sets count")
    assert_true(bs6.test(1), msg="Union: Identical sets bit 1")
    assert_true(bs6.test(2), msg="Union: Identical sets bit 2")
    assert_true(bs6.test(3), msg="Union: Identical sets bit 3")

    # Union of disjoint sets
    var bs7 = BitSet[128]()
    bs7.set(10)
    bs7.set(20)
    var bs8 = bs1.union(bs7)
    assert_equal(len(bs8), 5, msg="Union: Disjoint sets count")
    assert_true(bs8.test(1), msg="Union: Disjoint sets bit 1")
    assert_true(bs8.test(2), msg="Union: Disjoint sets bit 2")
    assert_true(bs8.test(3), msg="Union: Disjoint sets bit 3")
    assert_true(bs8.test(10), msg="Union: Disjoint sets bit 10")
    assert_true(bs8.test(20), msg="Union: Disjoint sets bit 20")

    # Union across word boundaries
    var bs9 = BitSet[128]()
    bs9.set(60)
    bs9.set(65)
    var bs10 = BitSet[128]()
    bs10.set(63)
    bs10.set(70)
    var bs11 = bs9.union(bs10)
    assert_equal(len(bs11), 4, msg="Union: Across words count")
    assert_true(bs11.test(60), msg="Union: Across words bit 60")
    assert_true(bs11.test(63), msg="Union: Across words bit 63")
    assert_true(bs11.test(65), msg="Union: Across words bit 65")
    assert_true(bs11.test(70), msg="Union: Across words bit 70")


def test_bitset_intersection():
    # Basic case
    var bs1 = BitSet[128]()
    bs1.set(1)
    bs1.set(2)
    bs1.set(3)

    var bs2 = BitSet[128]()
    bs2.set(3)
    bs2.set(4)
    bs2.set(5)

    var bs3 = bs1.intersection(bs2)
    assert_equal(len(bs3), 1, msg="Intersection: Basic case count")
    assert_true(bs3.test(3), msg="Intersection: Basic case bit 3")
    assert_false(bs3.test(1), msg="Intersection: Basic case bit 1")
    assert_false(bs3.test(2), msg="Intersection: Basic case bit 2")
    assert_false(bs3.test(4), msg="Intersection: Basic case bit 4")
    assert_false(bs3.test(5), msg="Intersection: Basic case bit 5")

    # Intersection with empty set
    var bs_empty = BitSet[128]()
    var bs4 = bs1.intersection(bs_empty)
    assert_equal(len(bs4), 0, msg="Intersection: With empty set count")

    var bs5 = bs_empty.intersection(bs1)
    assert_equal(
        len(bs5), 0, msg="Intersection: Empty with non-empty set count"
    )

    # Intersection of identical sets
    var bs6 = bs1.intersection(bs1)
    assert_equal(len(bs6), 3, msg="Intersection: Identical sets count")
    assert_true(bs6.test(1), msg="Intersection: Identical sets bit 1")
    assert_true(bs6.test(2), msg="Intersection: Identical sets bit 2")
    assert_true(bs6.test(3), msg="Intersection: Identical sets bit 3")

    # Intersection of disjoint sets
    var bs7 = BitSet[128]()
    bs7.set(10)
    bs7.set(20)
    var bs8 = bs1.intersection(bs7)
    assert_equal(len(bs8), 0, msg="Intersection: Disjoint sets count")

    # Intersection across word boundaries
    var bs9 = BitSet[128]()
    bs9.set(60)
    bs9.set(65)
    bs9.set(70)
    var bs10 = BitSet[128]()
    bs10.set(63)
    bs10.set(65)
    bs10.set(75)
    var bs11 = bs9.intersection(bs10)
    assert_equal(len(bs11), 1, msg="Intersection: Across words count")
    assert_true(bs11.test(65), msg="Intersection: Across words bit 65")
    assert_false(bs11.test(60), msg="Intersection: Across words bit 60")
    assert_false(bs11.test(63), msg="Intersection: Across words bit 63")
    assert_false(bs11.test(70), msg="Intersection: Across words bit 70")
    assert_false(bs11.test(75), msg="Intersection: Across words bit 75")


def test_bitset_difference():
    # Basic case (bs1 - bs2)
    var bs1 = BitSet[128]()
    bs1.set(1)
    bs1.set(2)
    bs1.set(3)

    var bs2 = BitSet[128]()
    bs2.set(3)
    bs2.set(4)
    bs2.set(5)

    var bs3 = bs1.difference(bs2)
    assert_equal(len(bs3), 2, msg="Difference: Basic case (bs1-bs2) count")
    assert_true(bs3.test(1), msg="Difference: Basic case (bs1-bs2) bit 1")
    assert_true(bs3.test(2), msg="Difference: Basic case (bs1-bs2) bit 2")
    assert_false(bs3.test(3), msg="Difference: Basic case (bs1-bs2) bit 3")
    assert_false(bs3.test(4), msg="Difference: Basic case (bs1-bs2) bit 4")

    # Basic case (bs2 - bs1)
    var bs4 = bs2.difference(bs1)
    assert_equal(len(bs4), 2, msg="Difference: Basic case (bs2-bs1) count")
    assert_true(bs4.test(4), msg="Difference: Basic case (bs2-bs1) bit 4")
    assert_true(bs4.test(5), msg="Difference: Basic case (bs2-bs1) bit 5")
    assert_false(bs4.test(1), msg="Difference: Basic case (bs2-bs1) bit 1")
    assert_false(bs4.test(3), msg="Difference: Basic case (bs2-bs1) bit 3")

    # Difference with empty set
    var bs_empty = BitSet[128]()
    var bs5 = bs1.difference(bs_empty)
    assert_equal(len(bs5), 3, msg="Difference: With empty set count")
    assert_true(bs5.test(1), msg="Difference: With empty set bit 1")
    assert_true(bs5.test(2), msg="Difference: With empty set bit 2")
    assert_true(bs5.test(3), msg="Difference: With empty set bit 3")

    var bs6 = bs_empty.difference(bs1)
    assert_equal(len(bs6), 0, msg="Difference: Empty with non-empty set count")

    # Difference of identical sets
    var bs7 = bs1.difference(bs1)
    assert_equal(len(bs7), 0, msg="Difference: Identical sets count")

    # Difference of disjoint sets
    var bs8 = BitSet[128]()
    bs8.set(10)
    bs8.set(20)
    var bs9 = bs1.difference(bs8)  # bs1 - bs8
    assert_equal(len(bs9), 3, msg="Difference: Disjoint sets (bs1-bs8) count")
    assert_true(bs9.test(1), msg="Difference: Disjoint sets (bs1-bs8) bit 1")
    assert_true(bs9.test(2), msg="Difference: Disjoint sets (bs1-bs8) bit 2")
    assert_true(bs9.test(3), msg="Difference: Disjoint sets (bs1-bs8) bit 3")
    assert_false(bs9.test(10), msg="Difference: Disjoint sets (bs1-bs8) bit 10")

    var bs10 = bs8.difference(bs1)  # bs8 - bs1
    assert_equal(len(bs10), 2, msg="Difference: Disjoint sets (bs8-bs1) count")
    assert_true(bs10.test(10), msg="Difference: Disjoint sets (bs8-bs1) bit 10")
    assert_true(bs10.test(20), msg="Difference: Disjoint sets (bs8-bs1) bit 20")
    assert_false(bs10.test(1), msg="Difference: Disjoint sets (bs8-bs1) bit 1")

    # Difference across word boundaries
    var bs11 = BitSet[128]()
    bs11.set(60)
    bs11.set(65)
    bs11.set(70)
    var bs12 = BitSet[128]()
    bs12.set(63)
    bs12.set(65)
    bs12.set(75)
    var bs13 = bs11.difference(bs12)  # bs11 - bs12
    assert_equal(len(bs13), 2, msg="Difference: Across words (bs11-bs12) count")
    assert_true(
        bs13.test(60), msg="Difference: Across words (bs11-bs12) bit 60"
    )
    assert_true(
        bs13.test(70), msg="Difference: Across words (bs11-bs12) bit 70"
    )
    assert_false(
        bs13.test(63), msg="Difference: Across words (bs11-bs12) bit 63"
    )
    assert_false(
        bs13.test(65), msg="Difference: Across words (bs11-bs12) bit 65"
    )
    assert_false(
        bs13.test(75), msg="Difference: Across words (bs11-bs12) bit 75"
    )

    var bs14 = bs12.difference(bs11)  # bs12 - bs11
    assert_equal(len(bs14), 2, msg="Difference: Across words (bs12-bs11) count")
    assert_true(
        bs14.test(63), msg="Difference: Across words (bs12-bs11) bit 63"
    )
    assert_true(
        bs14.test(75), msg="Difference: Across words (bs12-bs11) bit 75"
    )
    assert_false(
        bs14.test(60), msg="Difference: Across words (bs12-bs11) bit 60"
    )
    assert_false(
        bs14.test(65), msg="Difference: Across words (bs12-bs11) bit 65"
    )
    assert_false(
        bs14.test(70), msg="Difference: Across words (bs12-bs11) bit 70"
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


def test_bitset_small_size():
    # Test BitSet with size less than 64 (word size)
    var bs = BitSet[32]()
    assert_equal(len(bs), 0, msg="Small BitSet: Empty should have length 0")

    # Set a few bits
    bs.set(0)
    bs.set(15)
    bs.set(31)  # Edge of the small bitset
    assert_equal(len(bs), 3, msg="Small BitSet: Should have 3 bits set")

    # Test individual bits
    assert_true(bs.test(0), msg="Small BitSet: Bit 0 should be set")
    assert_true(bs.test(15), msg="Small BitSet: Bit 15 should be set")
    assert_true(bs.test(31), msg="Small BitSet: Bit 31 should be set")
    assert_false(bs.test(16), msg="Small BitSet: Bit 16 should not be set")

    # Test clear
    bs.clear(15)
    assert_equal(
        len(bs), 2, msg="Small BitSet: Should have 2 bits after clearing"
    )
    assert_false(bs.test(15), msg="Small BitSet: Bit 15 should be cleared")

    # Test toggle
    bs.toggle(16)
    assert_equal(
        len(bs), 3, msg="Small BitSet: Should have 3 bits after toggle"
    )
    assert_true(
        bs.test(16), msg="Small BitSet: Bit 16 should be set after toggle"
    )

    # Test clear_all
    bs.clear_all()
    assert_equal(
        len(bs), 0, msg="Small BitSet: Should be empty after clear_all"
    )
    assert_true(
        bs.is_empty(), msg="Small BitSet: Should be empty after clear_all"
    )

    # Test very small BitSet (size 1)
    var bs1 = BitSet[1]()
    assert_equal(len(bs1), 0, msg="BitSet[1]: Empty should have length 0")
    bs1.set(0)
    assert_equal(len(bs1), 1, msg="BitSet[1]: Should have 1 bit set")
    assert_true(bs1.test(0), msg="BitSet[1]: Bit 0 should be set")

    # Test BitSet with size 2
    var bs2 = BitSet[2]()
    bs2.set(0)
    bs2.set(1)
    assert_equal(len(bs2), 2, msg="BitSet[2]: Should have 2 bits set")
    bs2.toggle(0)
    assert_equal(len(bs2), 1, msg="BitSet[2]: Should have 1 bit after toggle")
    assert_false(bs2.test(0), msg="BitSet[2]: Bit 0 should be toggled off")

    # Test BitSet with size 3 (odd size)
    var bs3 = BitSet[3]()
    bs3.set(0)
    bs3.set(1)
    bs3.set(2)
    assert_equal(len(bs3), 3, msg="BitSet[3]: Should have all 3 bits set")
    assert_true(bs3.test(2), msg="BitSet[3]: Bit 2 should be set")

    # Test BitSet with size 8 (byte boundary)
    var bs8 = BitSet[8]()
    for i in range(8):
        bs8.set(i)
    assert_equal(len(bs8), 8, msg="BitSet[8]: Should have all 8 bits set")
    bs8.clear_all()
    assert_equal(len(bs8), 0, msg="BitSet[8]: Should be empty after clear_all")

    # Test BitSet with size 63 (just under word size)
    var bs63 = BitSet[63]()
    bs63.set(62)  # Last valid bit
    assert_equal(len(bs63), 1, msg="BitSet[63]: Should have 1 bit set")
    assert_true(bs63.test(62), msg="BitSet[63]: Bit 62 should be set")

    # Test operations on small BitSets
    var bsA = BitSet[16]()
    var bsB = BitSet[16]()
    bsA.set(1)
    bsA.set(3)
    bsA.set(5)
    bsB.set(3)
    bsB.set(7)

    # Test union
    var bsUnion = bsA.union(bsB)
    assert_equal(
        len(bsUnion), 4, msg="Small BitSet union: Should have 4 bits set"
    )
    assert_true(bsUnion.test(1), msg="Small BitSet union: Bit 1 should be set")
    assert_true(bsUnion.test(3), msg="Small BitSet union: Bit 3 should be set")
    assert_true(bsUnion.test(5), msg="Small BitSet union: Bit 5 should be set")
    assert_true(bsUnion.test(7), msg="Small BitSet union: Bit 7 should be set")

    # Test intersection
    var bsIntersection = bsA.intersection(bsB)
    assert_equal(
        len(bsIntersection),
        1,
        msg="Small BitSet intersection: Should have 1 bit set",
    )
    assert_true(
        bsIntersection.test(3),
        msg="Small BitSet intersection: Bit 3 should be set",
    )

    # Test difference
    var bsDifference = bsA.difference(bsB)
    assert_equal(
        len(bsDifference),
        2,
        msg="Small BitSet difference: Should have 2 bits set",
    )
    assert_true(
        bsDifference.test(1), msg="Small BitSet difference: Bit 1 should be set"
    )
    assert_true(
        bsDifference.test(5), msg="Small BitSet difference: Bit 5 should be set"
    )
    assert_false(
        bsDifference.test(3),
        msg="Small BitSet difference: Bit 3 should not be set",
    )


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
    test_bitset_union()
    test_bitset_intersection()
    test_bitset_difference()
    test_bitset_simd_init()
    test_bitset_len()
    test_bitset_small_size()
