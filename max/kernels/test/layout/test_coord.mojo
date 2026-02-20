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
"""Tests for the unified LayoutLike system."""

from buffer import Dim, DimList
from sys import size_of
from sys.intrinsics import _type_is_eq

from builtin.variadics import Variadic

from layout._coord import (
    ComptimeInt,
    CoordLike,
    Idx,
    Coord,
    RuntimeInt,
    coord_to_int_tuple,
    coord,
    idx2crd,
    _DimsToCoordLike,
    _Idx2CrdResultTypes,
)
from testing import assert_equal, assert_true, TestSuite


fn test_nested_layouts() raises:
    # Create nested layouts
    var inner = Coord(Idx[2](), Idx(Int(3)))
    var nested = Coord(inner, Idx[4]())
    assert_equal(inner[1].value(), 3)
    assert_equal(nested[0][0].value(), 2)
    assert_equal(nested[1].value(), 4)
    assert_equal(size_of[type_of(inner)](), size_of[Int]())
    assert_equal(size_of[type_of(nested)](), size_of[Int]())


fn test_int_tuple_conversion() raises:
    var t = Coord(Coord(Idx[2](), Idx(3)), Idx[4]())
    var t2 = coord_to_int_tuple(t)
    assert_equal(t2[0][0], 2)
    assert_equal(t2[0][1], 3)
    assert_equal(t2[1], 4)


fn test_list_literal_construction() raises:
    var t = Coord[ComptimeInt[2], RuntimeInt[DType.int]](
        Idx[2](),
        Idx(Int(3)),
    )
    assert_equal(t[0].value(), 2)
    assert_equal(t[1].value(), 3)


fn test_flatten_empty() raises:
    var t = Coord[]()
    assert_true(t.flatten() == t)


fn test_construction_from_int_variadic_empty() raises:
    var t = coord[]()
    assert_equal(len(t), 0)


fn test_construction_from_int_variadic() raises:
    var t = coord[1, 2, 3]()
    assert_equal(len(t), 3)
    assert_equal(t[0].value(), 1)
    assert_equal(t[1].value(), 2)
    assert_equal(t[2].value(), 3)


fn test_construction_from_int_variadic_list() raises:
    var t = coord[DType.int32]((1, 2, 3))
    assert_equal(len(t), 3)
    assert_equal(t[0].value(), 1)
    assert_equal(t[1].value(), 2)
    assert_equal(t[2].value(), 3)


fn test_static_product() raises:
    comptime p = coord[1, 2, 3]().static_product
    assert_equal(p, 6)


fn test_default_init() raises:
    var c = Coord[
        ComptimeInt[5],
        RuntimeInt[DType.int32],
        ComptimeInt[3],
        RuntimeInt[DType.int64],
    ]()
    assert_equal(c[0].value(), 5)
    assert_equal(c[1].value(), 0)
    assert_equal(c[2].value(), 3)
    assert_equal(c[3].value(), 0)


fn test_default_init_nested() raises:
    var c = Coord[
        ComptimeInt[5],
        Coord[
            RuntimeInt[DType.int32],
            ComptimeInt[3],
        ],
        RuntimeInt[DType.int64],
    ]()
    assert_equal(c[0].value(), 5)
    assert_equal(c[1][0].value(), 0)
    assert_equal(c[1][1].value(), 3)
    assert_equal(c[2].value(), 0)


def test_from_dimlist_empty():
    comptime dims = DimList()
    comptime coord = _DimsToCoordLike[DType.int32, dims]
    assert_equal(Variadic.size(coord), 0)


def test_from_dimlist():
    comptime dims = DimList(Dim(5), Dim(), Dim(3))
    comptime coord = _DimsToCoordLike[DType.int32, dims]
    assert_equal(Variadic.size(coord), 3)
    assert_true(_type_is_eq[coord[0], ComptimeInt[5]]())
    assert_true(_type_is_eq[coord[1], RuntimeInt[DType.int32]]())
    assert_true(_type_is_eq[coord[2], ComptimeInt[3]]())


fn test_idx2crd_basic() raises:
    """Test basic idx2crd correctness with row-major layout."""
    var shape = Coord(Idx[3](), Idx[4]())
    var stride = Coord(Idx[4](), Idx[1]())

    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0].value(), 0)
    assert_equal(c0[1].value(), 0)

    var c5 = idx2crd(5, shape, stride)
    assert_equal(c5[0].value(), 1)
    assert_equal(c5[1].value(), 1)

    var c11 = idx2crd(11, shape, stride)
    assert_equal(c11[0].value(), 2)
    assert_equal(c11[1].value(), 3)


fn test_idx2crd_static_shape_1() raises:
    """When a shape dim is statically 1, the coordinate is ComptimeInt[0]."""
    var shape = Coord(Idx[1](), Idx[4]())
    var stride = Coord(Idx[4](), Idx[1]())

    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0].value(), 0)
    assert_equal(c0[1].value(), 0)

    var c3 = idx2crd(3, shape, stride)
    assert_equal(c3[0].value(), 0)
    assert_equal(c3[1].value(), 3)

    # First dim should be ComptimeInt[0] (static shape 1).
    assert_true(_type_is_eq[type_of(c0[0]), ComptimeInt[0]]())
    # Second dim should be RuntimeInt.
    assert_true(_type_is_eq[type_of(c0[1]), RuntimeInt[DType.int64]]())


fn test_idx2crd_all_static_1() raises:
    """When all shape dims are 1, all coordinates are ComptimeInt[0]."""
    var shape = Coord(Idx[1](), Idx[1]())
    var stride = Coord(Idx[1](), Idx[1]())

    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0].value(), 0)
    assert_equal(c0[1].value(), 0)

    assert_true(_type_is_eq[type_of(c0[0]), ComptimeInt[0]]())
    assert_true(_type_is_eq[type_of(c0[1]), ComptimeInt[0]]())


fn test_idx2crd_mixed_static_dynamic() raises:
    """Shape (3, 1, 4): middle dim is statically 1, others are dynamic."""
    var shape = Coord(Idx[3](), Idx[1](), Idx[4]())
    var stride = Coord(Idx[4](), Idx[4](), Idx[1]())

    var c5 = idx2crd(5, shape, stride)
    assert_equal(c5[0].value(), 1)
    assert_equal(c5[1].value(), 0)
    assert_equal(c5[2].value(), 1)

    assert_true(_type_is_eq[type_of(c5[0]), RuntimeInt[DType.int64]]())
    assert_true(_type_is_eq[type_of(c5[1]), ComptimeInt[0]]())
    assert_true(_type_is_eq[type_of(c5[2]), RuntimeInt[DType.int64]]())


fn test_idx2crd_no_static_1() raises:
    """When no shape dim is 1, all coordinates are RuntimeInt."""
    var shape = Coord(Idx[3](), Idx[4]())
    var stride = Coord(Idx[4](), Idx[1]())

    var _c = idx2crd(0, shape, stride)
    assert_true(_type_is_eq[type_of(_c[0]), RuntimeInt[DType.int64]]())
    assert_true(_type_is_eq[type_of(_c[1]), RuntimeInt[DType.int64]]())


fn test_idx2crd_result_types_runtime_idx() raises:
    """No shape-1 dims with runtime idx: all RuntimeInt."""
    comptime shape = Variadic.types[T=CoordLike, ComptimeInt[3], ComptimeInt[4]]
    comptime stride = Variadic.types[
        T=CoordLike, ComptimeInt[4], ComptimeInt[1]
    ]
    comptime types = _Idx2CrdResultTypes[
        DType.int64, RuntimeInt[DType.int64], stride, shape
    ]
    assert_true(_type_is_eq[types[0], RuntimeInt[DType.int64]]())
    assert_true(_type_is_eq[types[1], RuntimeInt[DType.int64]]())


fn test_idx2crd_result_types_shape_1() raises:
    """Shape dim of 1 produces ComptimeInt[0], others RuntimeInt."""
    comptime shape = Variadic.types[T=CoordLike, ComptimeInt[1], ComptimeInt[4]]
    comptime stride = Variadic.types[
        T=CoordLike, ComptimeInt[4], ComptimeInt[1]
    ]
    comptime types = _Idx2CrdResultTypes[
        DType.int64, RuntimeInt[DType.int64], stride, shape
    ]
    assert_true(_type_is_eq[types[0], ComptimeInt[0]]())
    assert_true(_type_is_eq[types[1], RuntimeInt[DType.int64]]())


fn test_idx2crd_result_types_all_shape_1() raises:
    """All shape dims are 1: all ComptimeInt[0]."""
    comptime shape = Variadic.types[T=CoordLike, ComptimeInt[1], ComptimeInt[1]]
    comptime stride = Variadic.types[
        T=CoordLike, ComptimeInt[1], ComptimeInt[1]
    ]
    comptime types = _Idx2CrdResultTypes[
        DType.int64, RuntimeInt[DType.int64], stride, shape
    ]
    assert_true(_type_is_eq[types[0], ComptimeInt[0]]())
    assert_true(_type_is_eq[types[1], ComptimeInt[0]]())


fn test_idx2crd_result_types_runtime_shape() raises:
    """RuntimeInt shape dims always produce RuntimeInt result."""
    comptime shape = Variadic.types[
        T=CoordLike, RuntimeInt[DType.int], RuntimeInt[DType.int]
    ]
    comptime stride = Variadic.types[
        T=CoordLike, RuntimeInt[DType.int], RuntimeInt[DType.int]
    ]
    comptime types = _Idx2CrdResultTypes[
        DType.int64, RuntimeInt[DType.int64], stride, shape
    ]
    assert_true(_type_is_eq[types[0], RuntimeInt[DType.int64]]())
    assert_true(_type_is_eq[types[1], RuntimeInt[DType.int64]]())


fn test_idx2crd_result_types_all_static() raises:
    """All three static (idx=5, shape=(3,4), stride=(4,1)): compile-time results.
    """
    comptime shape = Variadic.types[T=CoordLike, ComptimeInt[3], ComptimeInt[4]]
    comptime stride = Variadic.types[
        T=CoordLike, ComptimeInt[4], ComptimeInt[1]
    ]
    comptime types = _Idx2CrdResultTypes[
        DType.int64, ComptimeInt[5], stride, shape
    ]
    # (5 // 4) % 3 = 1
    assert_true(_type_is_eq[types[0], ComptimeInt[1]]())
    # (5 // 1) % 4 = 1
    assert_true(_type_is_eq[types[1], ComptimeInt[1]]())


fn test_idx2crd_single_dim() raises:
    """Test idx2crd with a single (non-tuple) shape."""
    var c = idx2crd(7, Idx[10](), Idx[1]())
    assert_equal(c[0].value(), 7)
    assert_true(_type_is_eq[type_of(c[0]), RuntimeInt[DType.int64]]())

    # Single dim with shape 1 should produce ComptimeInt[0].
    var c1 = idx2crd(0, Idx[1](), Idx[1]())
    assert_equal(c1[0].value(), 0)
    assert_true(_type_is_eq[type_of(c1[0]), ComptimeInt[0]]())


fn test_idx2crd_col_major() raises:
    """Test idx2crd with col-major strides (which was broken with sequential algorithm).
    """
    # Shape (3, 4), col-major strides (1, 3)
    var shape = Coord(Idx[3](), Idx[4]())
    var stride = Coord(Idx[1](), Idx[3]())

    # idx=0 -> (0, 0)
    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0].value(), 0)
    assert_equal(c0[1].value(), 0)

    # idx=1 -> (1 // 1) % 3 = 1, (1 // 3) % 4 = 0 -> (1, 0)
    var c1 = idx2crd(1, shape, stride)
    assert_equal(c1[0].value(), 1)
    assert_equal(c1[1].value(), 0)

    # idx=3 -> (3 // 1) % 3 = 0, (3 // 3) % 4 = 1 -> (0, 1)
    var c3 = idx2crd(3, shape, stride)
    assert_equal(c3[0].value(), 0)
    assert_equal(c3[1].value(), 1)

    # idx=5 -> (5 // 1) % 3 = 2, (5 // 3) % 4 = 1 -> (2, 1)
    var c5 = idx2crd(5, shape, stride)
    assert_equal(c5[0].value(), 2)
    assert_equal(c5[1].value(), 1)

    # idx=11 -> (11 // 1) % 3 = 2, (11 // 3) % 4 = 3 -> (2, 3)
    var c11 = idx2crd(11, shape, stride)
    assert_equal(c11[0].value(), 2)
    assert_equal(c11[1].value(), 3)


fn test_idx2crd_comptime_idx() raises:
    """Test idx2crd with a compile-time index producing compile-time results."""
    var shape = Coord(Idx[3](), Idx[4]())
    var stride = Coord(Idx[4](), Idx[1]())

    # Compile-time idx=5 with static shape and stride should yield ComptimeInt results.
    var c5 = idx2crd(Idx[5](), shape, stride)
    # (5 // 4) % 3 = 1
    assert_equal(c5[0].value(), 1)
    # (5 // 1) % 4 = 1
    assert_equal(c5[1].value(), 1)
    # Both dimensions should be ComptimeInt.
    assert_true(_type_is_eq[type_of(c5[0]), ComptimeInt[1]]())
    assert_true(_type_is_eq[type_of(c5[1]), ComptimeInt[1]]())

    # Compile-time idx=0 should yield ComptimeInt[0] for both dims.
    var c0 = idx2crd(Idx[0](), shape, stride)
    assert_equal(c0[0].value(), 0)
    assert_equal(c0[1].value(), 0)
    assert_true(_type_is_eq[type_of(c0[0]), ComptimeInt[0]]())
    assert_true(_type_is_eq[type_of(c0[1]), ComptimeInt[0]]())


fn test_idx2crd_mixed_static_dynamic_idx() raises:
    """Test idx2crd with static idx but one runtime stride dimension."""
    # shape=(3, 4), stride=(RuntimeInt, ComptimeInt[1])
    var shape = Coord(Idx[3](), Idx[4]())
    var stride = Coord[RuntimeInt[DType.int], ComptimeInt[1]](
        Idx(Int(4)), Idx[1]()
    )

    # Static idx=5, but first stride is runtime -> first dim is RuntimeInt.
    # Second stride is static, shape is static, idx is static -> ComptimeInt.
    var c5 = idx2crd(Idx[5](), shape, stride)
    assert_equal(c5[0].value(), 1)
    assert_equal(c5[1].value(), 1)
    assert_true(_type_is_eq[type_of(c5[0]), RuntimeInt[DType.int64]]())
    # (5 // 1) % 4 = 1
    assert_true(_type_is_eq[type_of(c5[1]), ComptimeInt[1]]())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
