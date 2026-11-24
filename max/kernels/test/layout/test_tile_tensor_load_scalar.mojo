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
"""Tests scalar loads from static, dynamic, and vectorized tensors."""

from layout import Coord, TileTensor, row_major
from std.testing import TestSuite, assert_equal


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_scalar_load_static_layout() raises:
    """Tests scalar loads with a static 2x3 row-major layout."""
    comptime layout = row_major[2, 3]()
    var storage: Array[Float32, 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    var tensor = TileTensor(storage, layout)

    # Test scalar access at various positions
    var v00: Float32 = tensor.load[width=1]((0, 0))
    assert_equal(v00, 0.0)

    var v01: Float32 = tensor.load[width=1]((0, 1))
    assert_equal(v01, 1.0)

    var v02: Float32 = tensor.load[width=1]((0, 2))
    assert_equal(v02, 2.0)

    var v10: Float32 = tensor.load[width=1]((1, 0))
    assert_equal(v10, 3.0)

    var v11: Float32 = tensor.load[width=1]((1, 1))
    assert_equal(v11, 4.0)

    var v12: Float32 = tensor.load[width=1]((1, 2))
    assert_equal(v12, 5.0)


def test_scalar_load_dynamic_layout() raises:
    """Tests scalar loads with a dynamic runtime layout."""
    var storage = Array[Float32, 12](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )

    var tensor = TileTensor[linear_idx_type=.int32](
        storage, row_major(Coord(Int32(3), Int32(4)))
    )

    # Test scalar loads at various positions
    var v00: Float32 = tensor.load[width=1]((0, 0))
    assert_equal(v00, 0.0)

    var v11: Float32 = tensor.load[width=1]((1, 1))
    assert_equal(v11, 5.0)  # row 1, col 1 = 1*4 + 1 = 5

    var v23: Float32 = tensor.load[width=1]((2, 3))
    assert_equal(v23, 11.0)  # row 2, col 3 = 2*4 + 3 = 11


def test_scalar_load_with_coord() raises:
    """Tests scalar loads using 32-bit coordinates."""
    var storage = Array[Float32, 16](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )

    var tensor = TileTensor[linear_idx_type=.int32](
        storage, row_major(Coord(Int32(4), Int32(4)))
    )

    var coords = Coord(Int32(2), Int32(3))
    var val: Float32 = tensor.load[width=1](coords)
    assert_equal(val, 11.0)  # row 2, col 3 = 2*4 + 3 = 11


def test_scalar_load_matches_getitem_lane0() raises:
    """Tests that scalar loads match the first lane of element indexing."""
    comptime layout = row_major[4, 4]()
    var storage = Array[Float32, 16](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )

    var tensor = TileTensor(storage, layout)

    # Verify scalar loads match the 0th lane of __getitem__
    for i in range(4):
        for j in range(4):
            var scalar_val = tensor.load[width=1]((i, j))
            var simd_val = tensor[i, j]
            assert_equal(scalar_val, simd_val[0])


def test_scalar_load_vectorized_element_size_gt_1() raises:
    """Tests scalar loads with a vectorized tensor where element_size > 1.

    Element indexing returns a SIMD vector; an explicit width of one loads
    only the first lane at the same logical coordinate.
    """
    # Create an 8x8 tensor and vectorize it to have 4-element vectors
    comptime layout = row_major[8, 8]()
    var storage = Array[Float32, 64](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )

    var tensor = TileTensor(storage, layout)

    # Vectorize to 1x4 elements - this creates a tensor where each "element"
    # is a SIMD[float32, 4] (element_size = 4)
    var vec_tensor = tensor.vectorize[1, 4]()

    # Verify element_size > 1
    comptime assert (
        vec_tensor.element_size == 4
    ), "Expected element_size == 4 for vectorized tensor"

    # __getitem__ returns SIMD[float32, 4], a width-one load returns just the 0th lane
    # For position (0, 0): the element contains [0, 1, 2, 3], a width-one load returns 0
    var simd_val = vec_tensor[0, 0]  # Returns SIMD[float32, 4] = [0, 1, 2, 3]
    var scalar_val = vec_tensor.load[width=1]((0, 0))  # Returns Scalar = 0.0

    assert_equal(scalar_val, simd_val[0])
    assert_equal(scalar_val, 0.0)

    # Test another position: (0, 1) should have elements [4, 5, 6, 7]
    var simd_val2 = vec_tensor[0, 1]
    var scalar_val2 = vec_tensor.load[width=1]((0, 1))

    assert_equal(scalar_val2, simd_val2[0])
    assert_equal(scalar_val2, 4.0)

    # Test position (1, 0): row 1, col 0 of vectorized tensor
    # In original tensor this is position (1, 0) = value 8.0
    var scalar_val3 = vec_tensor.load[width=1]((1, 0))
    assert_equal(scalar_val3, 8.0)
