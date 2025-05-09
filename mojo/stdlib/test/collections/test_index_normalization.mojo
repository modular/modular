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

from collections._index_normalization import normalize_index

from testing import assert_equal


def test_out_of_bounds_message():
    # CHECK: index out of bounds
    _ = normalize_index[""](2, 2)
    # CHECK: index out of bounds
    _ = normalize_index[""](UInt(2), 2)
    # CHECK: index out of bounds
    _ = normalize_index[""](2, UInt(2))
    # CHECK: index out of bounds
    _ = normalize_index[""](UInt(2), UInt(2))
    # CHECK: index out of bounds
    _ = normalize_index[""](UInt8(2), 2)

    # CHECK: index out of bounds
    _ = normalize_index[""](-3, 2)
    # CHECK: index out of bounds
    _ = normalize_index[""](-3, UInt(2))
    # CHECK: index out of bounds
    _ = normalize_index[""](Int8(-3), 2)

    # CHECK: index out of bounds
    _ = normalize_index[""](2, 0)
    # CHECK: index out of bounds
    _ = normalize_index[""](UInt(2), 0)
    # CHECK: index out of bounds
    _ = normalize_index[""](2, UInt(0))
    # CHECK: index out of bounds
    _ = normalize_index[""](UInt(2), UInt(0))

    # CHECK: index out of bounds
    _ = normalize_index[""](Int.MIN, 10)
    # CHECK: index out of bounds
    _ = normalize_index[""](Int.MIN, UInt(10))
    # CHECK: index out of bounds
    _ = normalize_index[""](Int.MAX, 10)
    # CHECK: index out of bounds
    _ = normalize_index[""](Int.MAX, UInt(10))
    # CHECK: index out of bounds
    _ = normalize_index[""](Int.MIN, Int.MAX)

    # CHECK: index out of bounds
    _ = normalize_index[""](UInt.MAX, 10)
    # CHECK: index out of bounds
    _ = normalize_index[""](UInt.MAX, UInt(10))
    # CHECK: index out of bounds
    _ = normalize_index[""](UInt.MAX, UInt.MAX)
    # CHECK: index out of bounds
    _ = normalize_index[""](UInt.MAX, UInt.MAX - 10)


def test_normalize_index():
    assert_equal(normalize_index[""](-3, 3), 0)
    assert_equal(normalize_index[""](-2, 3), 1)
    assert_equal(normalize_index[""](-1, 3), 2)
    assert_equal(normalize_index[""](0, 3), 0)
    assert_equal(normalize_index[""](1, 3), 1)
    assert_equal(normalize_index[""](2, 3), 2)

    assert_equal(normalize_index[""](-3, UInt(3)), 0)
    assert_equal(normalize_index[""](-2, UInt(3)), 1)
    assert_equal(normalize_index[""](-1, UInt(3)), 2)
    assert_equal(normalize_index[""](0, UInt(3)), 0)
    assert_equal(normalize_index[""](1, UInt(3)), 1)
    assert_equal(normalize_index[""](2, UInt(3)), 2)

    assert_equal(normalize_index[""](UInt(0), UInt(3)), 0)
    assert_equal(normalize_index[""](UInt(1), UInt(3)), 1)
    assert_equal(normalize_index[""](UInt(2), UInt(3)), 2)

    assert_equal(normalize_index[""](Int8(-3), 3), 0)
    assert_equal(normalize_index[""](Int8(-2), 3), 1)
    assert_equal(normalize_index[""](Int8(-1), 3), 2)
    assert_equal(normalize_index[""](Int8(0), 3), 0)
    assert_equal(normalize_index[""](Int8(1), 3), 1)
    assert_equal(normalize_index[""](Int8(2), 3), 2)

    assert_equal(normalize_index[""](UInt8(0), 3), 0)
    assert_equal(normalize_index[""](UInt8(1), 3), 1)
    assert_equal(normalize_index[""](UInt8(2), 3), 2)

    assert_equal(normalize_index[""](UInt(1), UInt.MAX), 1)
    assert_equal(normalize_index[""](UInt.MAX - 5, UInt.MAX), UInt.MAX - 5)

    assert_equal(normalize_index[""](-1, Int.MAX), Int.MAX - 1)
    assert_equal(normalize_index[""](-10, Int.MAX), Int.MAX - 10)
    assert_equal(normalize_index[""](-1, UInt.MAX), UInt.MAX - 1)
    assert_equal(normalize_index[""](-10, UInt.MAX), UInt.MAX - 10)
    assert_equal(normalize_index[""](-1, UInt(Int.MAX) + 1), UInt(Int.MAX))
    assert_equal(normalize_index[""](Int.MIN, UInt(Int.MAX) + 1), 0)


def main():
    test_out_of_bounds_message()
    test_normalize_index()
