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
"""Step-by-step tests for lt_to_tt via _DimsToCoordLike (the working path)."""

from buffer import Dim, DimList
from layout import (
    Layout,
    LayoutTensor,
    UNKNOWN_VALUE,
    RuntimeLayout,
    TileTensor,
    lt_to_tt,
)
from layout.coord import (
    _DimsToCoordLike,
    Coord,
    CoordLike,
    ComptimeInt,
    RuntimeInt,
)
from layout.int_tuple import IntTuple
from layout.tile_layout import Layout as InternalLayout
from layout.tile_tensor import LTToTTLayout
from std.memory import UnsafePointer
from std.utils.index import Index
from std.testing import assert_equal


# ============================================================
# Step 1: _DimsToCoordLike with literal DimList (known to work)
# ============================================================


fn test_dims_to_coord_like_static() raises:
    print("--- test_dims_to_coord_like_static ---")
    comptime dims = DimList(4, 8)
    comptime CoordTypes = _DimsToCoordLike[DType.int64, dims]
    # Both dims are static, so both should be ComptimeInt
    comptime assert CoordTypes[0].is_static_value, "dim 0 should be static"
    comptime assert CoordTypes[1].is_static_value, "dim 1 should be static"
    comptime assert CoordTypes[0].static_value == 4, "dim 0 should be 4"
    comptime assert CoordTypes[1].static_value == 8, "dim 1 should be 8"
    print("  PASSED")


fn test_dims_to_coord_like_dynamic() raises:
    print("--- test_dims_to_coord_like_dynamic ---")
    comptime dims = DimList(Dim(), Dim(8))
    comptime CoordTypes = _DimsToCoordLike[DType.int64, dims]
    # dim 0 is dynamic -> RuntimeInt, dim 1 is static -> ComptimeInt
    comptime assert not CoordTypes[0].is_static_value, "dim 0 should be dynamic"
    comptime assert CoordTypes[1].is_static_value, "dim 1 should be static"
    comptime assert CoordTypes[1].static_value == 8, "dim 1 should be 8"
    print("  PASSED")


# ============================================================
# Step 2: Build a TileTensor type from DimList
# ============================================================


fn test_tiletensor_type_from_dims() raises:
    print("--- test_tiletensor_type_from_dims ---")
    comptime dims = DimList(4, 8)
    comptime shape_types = _DimsToCoordLike[DType.int64, dims]
    # For row-major stride (8, 1):
    comptime stride_dims = DimList(8, 1)
    comptime stride_types = _DimsToCoordLike[DType.int64, stride_dims]
    comptime TTLayout = InternalLayout[shape_types, stride_types]
    comptime TTType = TileTensor[DType.float32, TTLayout, MutAnyOrigin]
    comptime assert TTType.rank == 2, "rank should be 2"
    print("  rank:", TTType.rank)
    print("  PASSED")


# ============================================================
# Step 3: Build DimList from a public Layout's shape
# ============================================================


fn test_dimlist_from_layout_shape() raises:
    print("--- test_dimlist_from_layout_shape ---")

    # Public layout with static dims
    comptime lt_layout = Layout.row_major(4, 8)

    # Build DimList from IntTuple shape
    # For a flat 2D layout: shape[0] and shape[1] are Int values
    comptime shape_dim0 = lt_layout.shape[0].value()
    comptime shape_dim1 = lt_layout.shape[1].value()

    # Convert to Dim: UNKNOWN_VALUE (-1) becomes Dim(), others become Dim(N)
    comptime d0 = Dim(shape_dim0) if shape_dim0 != UNKNOWN_VALUE else Dim()
    comptime d1 = Dim(shape_dim1) if shape_dim1 != UNKNOWN_VALUE else Dim()
    comptime shape_dims = DimList(d0, d1)

    comptime CoordTypes = _DimsToCoordLike[DType.int64, shape_dims]
    comptime assert CoordTypes[0].is_static_value, "dim 0 should be static"
    comptime assert CoordTypes[1].is_static_value, "dim 1 should be static"
    comptime assert CoordTypes[0].static_value == 4, "dim 0 should be 4"
    comptime assert CoordTypes[1].static_value == 8, "dim 1 should be 8"
    print("  PASSED")


# ============================================================
# Step 4: DimList from a DYNAMIC public Layout shape
# ============================================================


fn test_dimlist_from_dynamic_layout_shape() raises:
    print("--- test_dimlist_from_dynamic_layout_shape ---")

    # Public layout with one dynamic dim
    comptime lt_layout = Layout.row_major[dims=DimList(Dim(), Dim(8))]()

    comptime shape_dim0 = lt_layout.shape[0].value()
    comptime shape_dim1 = lt_layout.shape[1].value()

    comptime d0 = Dim(shape_dim0) if shape_dim0 != UNKNOWN_VALUE else Dim()
    comptime d1 = Dim(shape_dim1) if shape_dim1 != UNKNOWN_VALUE else Dim()
    comptime shape_dims = DimList(d0, d1)

    comptime CoordTypes = _DimsToCoordLike[DType.int64, shape_dims]
    comptime assert not CoordTypes[0].is_static_value, "dim 0 should be dynamic"
    comptime assert CoordTypes[1].is_static_value, "dim 1 should be static"
    comptime assert CoordTypes[1].static_value == 8, "dim 1 should be 8"
    print("  PASSED")


# ============================================================
# Step 5: Build full TileTensor type from a public Layout
# ============================================================


fn test_tiletensor_type_from_public_layout() raises:
    print("--- test_tiletensor_type_from_public_layout ---")

    comptime lt_layout = Layout.row_major[dims=DimList(Dim(), Dim(8))]()

    # Helper: convert IntTuple element to Dim
    @parameter
    fn _int_to_dim(value: Int) -> Dim:
        if value != UNKNOWN_VALUE:
            return Dim(value)
        return Dim()

    comptime shape_dims = DimList(
        _int_to_dim(lt_layout.shape[0].value()),
        _int_to_dim(lt_layout.shape[1].value()),
    )
    comptime stride_dims = DimList(
        _int_to_dim(lt_layout.stride[0].value()),
        _int_to_dim(lt_layout.stride[1].value()),
    )

    comptime shape_types = _DimsToCoordLike[DType.int64, shape_dims]
    comptime stride_types = _DimsToCoordLike[DType.int64, stride_dims]
    comptime TTLayout = InternalLayout[shape_types, stride_types]
    comptime TTType = TileTensor[DType.float32, TTLayout, MutAnyOrigin]

    comptime assert TTType.rank == 2, "rank should be 2"
    # dim 0 is dynamic, dim 1 is static
    comptime assert not shape_types[
        0
    ].is_static_value, "shape 0 should be dynamic"
    comptime assert shape_types[1].is_static_value, "shape 1 should be static"
    print("  rank:", TTType.rank)
    print("  PASSED")


# ============================================================
# Step 6: lt_to_tt function using _DimsToCoordLike path
# ============================================================


fn test_lt_to_tt_function() raises:
    print("--- test_lt_to_tt_function ---")

    @parameter
    fn _int_to_dim(value: Int) -> Dim:
        if value != UNKNOWN_VALUE:
            return Dim(value)
        return Dim()

    fn lt_to_tt_2d[
        dtype: DType,
        lt_layout: Layout,
    ](lt: LayoutTensor[dtype, lt_layout, ...]) -> TileTensor[
        dtype,
        InternalLayout[
            _DimsToCoordLike[
                DType.int64,
                DimList(
                    _int_to_dim(lt_layout.shape[0].value()),
                    _int_to_dim(lt_layout.shape[1].value()),
                ),
            ],
            _DimsToCoordLike[
                DType.int64,
                DimList(
                    _int_to_dim(lt_layout.stride[0].value()),
                    _int_to_dim(lt_layout.stride[1].value()),
                ),
            ],
        ],
        lt.origin,
    ]:
        comptime ShapeTypes = _DimsToCoordLike[
            DType.int64,
            DimList(
                _int_to_dim(lt_layout.shape[0].value()),
                _int_to_dim(lt_layout.shape[1].value()),
            ),
        ]
        comptime StrideTypes = _DimsToCoordLike[
            DType.int64,
            DimList(
                _int_to_dim(lt_layout.stride[0].value()),
                _int_to_dim(lt_layout.stride[1].value()),
            ),
        ]
        var shape = Coord[*ShapeTypes]()
        var stride = Coord[*StrideTypes]()

        comptime for i in range(2):
            comptime if not shape.element_types[i].is_static_value:
                shape[i] = rebind[shape.element_types[i]](
                    Scalar[DType.int64](lt.runtime_layout.shape.value[i])
                )

            comptime if not stride.element_types[i].is_static_value:
                stride[i] = rebind[stride.element_types[i]](
                    Scalar[DType.int64](lt.runtime_layout.stride.value[i])
                )

        comptime ResultLayout = InternalLayout[ShapeTypes, StrideTypes]
        var ptr = UnsafePointer[Scalar[dtype], lt.origin](
            unsafe_from_address=Int(lt.ptr)
        )
        return TileTensor[dtype, ResultLayout, lt.origin](
            ptr=ptr,
            layout=ResultLayout(shape, stride),
        )

    # Test with static layout
    comptime static_layout = Layout.row_major(4, 8)
    var array1 = InlineArray[Float32, 32](fill=1.0)
    var lt1 = LayoutTensor[DType.float32, static_layout](array1.unsafe_ptr())
    var tt1 = lt_to_tt_2d(lt1)
    _ = tt1
    print("  static: rank =", tt1.rank)

    # Test with dynamic layout
    comptime dynamic_layout = Layout.row_major[dims=DimList(Dim(), Dim(8))]()
    var array2 = InlineArray[Float32, 32](fill=2.0)
    var lt2 = LayoutTensor[DType.float32, dynamic_layout](
        array2.unsafe_ptr(),
        RuntimeLayout[dynamic_layout].row_major(Index(4, 8)),
    )
    var tt2 = lt_to_tt_2d(lt2)
    _ = tt2
    print("  dynamic: rank =", tt2.rank)
    print("  PASSED")


# ============================================================
# Step 7: Generalized lt_to_tt from layout.tile_tensor
# ============================================================


fn test_generalized_lt_to_tt() raises:
    print("--- test_generalized_lt_to_tt ---")

    # --- 2D static layout ---
    comptime static_2d = Layout.row_major(4, 8)
    var arr_2d = InlineArray[Float32, 32](fill=1.0)
    var lt_2d = LayoutTensor[DType.float32, static_2d](arr_2d.unsafe_ptr())
    var tt_2d = lt_to_tt(lt_2d)
    assert_equal(tt_2d.rank, 2)
    assert_equal(Int(tt_2d.dim[0]()), 4)
    assert_equal(Int(tt_2d.dim[1]()), 8)
    print("  2D static: PASSED")

    # --- 2D dynamic layout ---
    comptime dynamic_2d = Layout.row_major[dims=DimList(Dim(), Dim(8))]()
    var arr_2d_dyn = InlineArray[Float32, 24](fill=2.0)
    var lt_2d_dyn = LayoutTensor[DType.float32, dynamic_2d](
        arr_2d_dyn.unsafe_ptr(),
        RuntimeLayout[dynamic_2d].row_major(Index(3, 8)),
    )
    var tt_2d_dyn = lt_to_tt(lt_2d_dyn)
    assert_equal(tt_2d_dyn.rank, 2)
    assert_equal(Int(tt_2d_dyn.dim[0]()), 3)
    assert_equal(Int(tt_2d_dyn.dim[1]()), 8)
    print("  2D dynamic: PASSED")

    # --- 1D dynamic layout ---
    comptime dynamic_1d = Layout(UNKNOWN_VALUE)
    var arr_1d = InlineArray[UInt32, 5](fill=42)
    var lt_1d = LayoutTensor[DType.uint32, dynamic_1d](
        arr_1d.unsafe_ptr(),
        RuntimeLayout[dynamic_1d](Index(5), Index(1)),
    )
    var tt_1d = lt_to_tt(lt_1d)
    assert_equal(tt_1d.rank, 1)
    assert_equal(Int(tt_1d.dim[0]()), 5)
    print("  1D dynamic: PASSED")

    # --- 4D mixed layout (matching kv_cache pattern) ---
    # Shape: (UNKNOWN, UNKNOWN, 8, 128), strides row-major from known dims
    comptime shape_4d = IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, 8, 128)
    comptime layout_4d = Layout.row_major(shape_4d)
    # Verify the strides we expect from row_major with unknown leading dims:
    #   stride[3] = 1, stride[2] = 128, stride[1] = 1024, stride[0] = UNKNOWN
    comptime assert layout_4d.stride[3].value() == 1
    comptime assert layout_4d.stride[2].value() == 128
    comptime assert layout_4d.stride[1].value() == 1024

    # Verify the TileTensor layout types
    comptime TTLayout4D = LTToTTLayout[layout_4d]
    comptime assert TTLayout4D.rank == 4
    # dim 0 and 1 are dynamic (UNKNOWN_VALUE in shape)
    comptime assert not TTLayout4D._shape_types[0].is_static_value
    comptime assert not TTLayout4D._shape_types[1].is_static_value
    # dim 2 and 3 are static
    comptime assert TTLayout4D._shape_types[2].is_static_value
    comptime assert TTLayout4D._shape_types[3].is_static_value
    comptime assert TTLayout4D._shape_types[2].static_value == 8
    comptime assert TTLayout4D._shape_types[3].static_value == 128
    # stride 0 is dynamic, strides 1-3 are static
    comptime assert not TTLayout4D._stride_types[0].is_static_value
    comptime assert TTLayout4D._stride_types[1].is_static_value
    comptime assert TTLayout4D._stride_types[1].static_value == 1024
    comptime assert TTLayout4D._stride_types[2].static_value == 128
    comptime assert TTLayout4D._stride_types[3].static_value == 1

    # Construct a 4D LayoutTensor and convert
    comptime num_blocks = 2
    comptime max_seq_len = 16
    comptime num_elements = num_blocks * max_seq_len * 8 * 128
    var arr_4d = InlineArray[Float32, num_elements](fill=3.0)
    var lt_4d = LayoutTensor[DType.float32, layout_4d](
        arr_4d.unsafe_ptr(),
        RuntimeLayout[layout_4d](
            Index(num_blocks, max_seq_len, 8, 128),
            Index(max_seq_len * 8 * 128, 8 * 128, 128, 1),
        ),
    )
    var tt_4d = lt_to_tt(lt_4d)
    assert_equal(tt_4d.rank, 4)
    assert_equal(Int(tt_4d.dim[0]()), num_blocks)
    assert_equal(Int(tt_4d.dim[1]()), max_seq_len)
    assert_equal(Int(tt_4d.dim[2]()), 8)
    assert_equal(Int(tt_4d.dim[3]()), 128)
    print("  4D mixed: PASSED")

    print("  ALL PASSED")


fn main() raises:
    test_dims_to_coord_like_static()
    test_dims_to_coord_like_dynamic()
    test_tiletensor_type_from_dims()
    test_dimlist_from_layout_shape()
    test_dimlist_from_dynamic_layout_shape()
    test_tiletensor_type_from_public_layout()
    test_lt_to_tt_function()
    test_generalized_lt_to_tt()
    print("=== ALL TESTS PASSED ===")
