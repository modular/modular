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

from layout import (
    ComptimeInt,
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout.tile_layout import Layout as TileLayout
from layout.tile_tensor import NullableTileTensor
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_cast_from_mutable_tile_tensor() raises:
    """Cast from a mutable TileTensor gives a non-null ptr."""
    var data = InlineArray[Float32, 6](uninitialized=True)
    var tile = TileTensor(data, row_major[2, 3]())
    var nullable = NullableTileTensor(tile)
    assert_true(Bool(nullable.ptr))


def test_ptr_none_after_clear() raises:
    """Ptr is None when cleared manually."""
    var data = InlineArray[Float32, 6](uninitialized=True)
    var tile = TileTensor(data, row_major[2, 3]())
    var nullable = NullableTileTensor(tile)
    assert_true(Bool(nullable.ptr))
    nullable.ptr = Optional[type_of(nullable).PtrType](None)
    assert_false(Bool(nullable.ptr))


def test_dim_comptime() raises:
    """Dim[i]() returns the correct shape for each dimension."""
    var data = InlineArray[Int32, 12](uninitialized=True)
    var nullable = NullableTileTensor(TileTensor(data, row_major[3, 4]()))
    assert_equal(nullable.dim[0](), 3)
    assert_equal(nullable.dim[1](), 4)


def test_dim_runtime() raises:
    """Dim(i) with a runtime index returns the same values as dim[i]()."""
    var data = InlineArray[Int32, 20](uninitialized=True)
    var nullable = NullableTileTensor(TileTensor(data, row_major[4, 5]()))
    assert_equal(nullable.dim(0), nullable.dim[0]())
    assert_equal(nullable.dim(1), nullable.dim[1]())


def test_dim_3d() raises:
    """Dim queries work on a 3D tensor."""
    var data = InlineArray[Float32, 24](uninitialized=True)
    var nullable = NullableTileTensor(TileTensor(data, row_major[2, 3, 4]()))
    assert_equal(nullable.dim[0](), 2)
    assert_equal(nullable.dim[1](), 3)
    assert_equal(nullable.dim[2](), 4)


def test_value_returns_working_tile_tensor() raises:
    """Value() unwraps the pointer and allows element access."""
    var data = InlineArray[Int32, 6](uninitialized=True)
    for i in range(6):
        data[i] = Int32(i * 10)

    var tile = TileTensor(data, row_major[2, 3]())
    var nullable = NullableTileTensor(tile)

    var unwrapped = nullable.value()
    assert_equal(unwrapped[(Idx(0), Idx(0))], 0)
    assert_equal(unwrapped[(Idx(0), Idx(1))], 10)
    assert_equal(unwrapped[(Idx(1), Idx(2))], 50)


def test_value_shares_memory_with_original() raises:
    """Value() returns a view — writes go to the same backing array."""
    var data = InlineArray[Int32, 4](fill=0)
    var tile = TileTensor(data, row_major[2, 2]())
    var nullable = NullableTileTensor(tile)

    nullable.value()[(Idx(0), Idx(1))] = 99
    assert_equal(tile[(Idx(0), Idx(1))], 99)


def test_layout_field_accessible() raises:
    """Layout field is accessible and holds the correct shape/stride info."""
    var data = InlineArray[Float32, 6](uninitialized=True)
    var tile = TileTensor(data, row_major[2, 3]())
    var nullable = NullableTileTensor(tile)

    # Shape and stride can be read from the layout field directly
    assert_equal(nullable.layout.shape[0]().value(), 2)
    assert_equal(nullable.layout.shape[1]().value(), 3)
    assert_equal(nullable.layout.stride[0]().value(), 3)
    assert_equal(nullable.layout.stride[1]().value(), 1)


def test_to_layout_tensor() raises:
    """To_layout_tensor() gives a LayoutTensor with matching shape."""
    var data = InlineArray[Float32, 6](uninitialized=True)
    for i in range(6):
        data[i] = Float32(i)

    var nullable = NullableTileTensor(TileTensor(data, row_major[2, 3]()))
    var lt = nullable.to_layout_tensor()

    assert_equal(lt.dim[0](), 2)
    assert_equal(lt.dim[1](), 3)
    assert_equal(lt[0, 0], Float32(0))
    assert_equal(lt[1, 2], Float32(5))
