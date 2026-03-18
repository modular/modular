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
"""Provides a mixed compile-time/runtime layout system for tensor memory mapping.

This module provides a layout system where some dimensions can be known at
compile time and others determined at runtime, enabling ergonomic layout
definitions while maintaining performance through compile-time specialization.

Key components:

- [`TensorLayout`](/mojo/kernels/layout/tile_layout/TensorLayout): Trait
  defining the interface for all mixed layouts.
- [`Layout`](/mojo/kernels/layout/tile_layout/Layout): Primary struct
  implementing a layout with mixed compile-time and runtime dimensions.
- [`row_major`](/mojo/kernels/layout/tile_layout/row_major): Create a
  row-major layout from a shape.
- [`col_major`](/mojo/kernels/layout/tile_layout/col_major): Create a
  column-major layout from a shape.
- [`blocked_product`](/mojo/kernels/layout/tile_layout/blocked_product):
  Create a hierarchical blocked layout from block and tiler layouts.
- [`zipped_divide`](/mojo/kernels/layout/tile_layout/zipped_divide): Divide
  a layout into inner and outer components by a tile shape.

You can import these APIs from the `layout` package:

```mojo
from layout.tile_layout import Layout, TensorLayout, row_major, col_major
```
"""

from std.os import abort
from std.sys.intrinsics import _type_is_eq

from std.builtin.variadics import (
    Variadic,
    VariadicPack,
    _MapVariadicAndIdxToType,
    _ReduceVariadicAndIdxToVariadic,
    _ReduceVariadicAndIdxToValue,
)

from .coord import (
    ComptimeInt,
    Idx,
    Coord,
    CoordLike,
    RuntimeInt,
    DynamicCoord,
    crd2idx,
    idx2crd,
    coord_to_int_tuple,
    _IntToComptimeInt,
    _CoordToDynamic,
    _Divide,
    _Multiply,
    _MultiplyByScalar,
    _Flattened,
)
from .int_tuple import IntTuple
from .layout import Layout as LegacyLayout


trait TensorLayout(TrivialRegisterPassable):
    """Trait defining the interface for mixed compile-time/runtime layouts.

    Implementors map logical multi-dimensional coordinates to linear memory
    indices, with support for dimensions that are known at compile time or
    determined at runtime.
    """

    comptime rank: Int
    """The number of dimensions in the layout."""
    comptime flat_rank: Int
    """The number of dimensions after flattening nested coordinates."""
    comptime shape_known: Bool
    """Whether all shape dimensions are known at compile time."""
    comptime stride_known: Bool
    """Whether all stride dimensions are known at compile time."""
    comptime all_dims_known: Bool = Self.shape_known and Self.stride_known
    """Whether all shape and stride dimensions are known at compile time."""
    comptime static_shape[i: Int]: Int
    """Returns the compile-time value of the i-th shape dimension.

    Parameters:
        i: The dimension index.
    """
    comptime static_stride[i: Int]: Int
    """Returns the compile-time value of the i-th stride dimension.

    Parameters:
        i: The dimension index.
    """

    comptime static_product: Int
    """The compile-time product of all shape dimensions."""

    comptime static_cosize: Int
    """The compile-time size of the memory region spanned by the layout."""

    comptime _shape_types: Variadic.TypesOfTrait[CoordLike]
    comptime _stride_types: Variadic.TypesOfTrait[CoordLike]

    def shape[i: Int](self) -> Self._shape_types[i]:
        """Returns the i-th shape dimension.

        Parameters:
            i: The dimension index.

        Returns:
            The shape value for dimension `i`.
        """
        ...

    def stride[i: Int](self) -> Self._stride_types[i]:
        """Returns the i-th stride dimension.

        Parameters:
            i: The dimension index.

        Returns:
            The stride value for dimension `i`.
        """
        ...

    def product(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        Returns:
            The product of all shape dimensions.
        """
        ...

    def size(self) -> Int:
        """Returns the total number of elements. Alias for `product()`.

        Returns:
            The product of all shape dimensions.
        """
        ...

    def __call__[
        index_type: CoordLike,
        *,
        linear_idx_type: DType = DType.int64,
    ](self, index: index_type) -> Scalar[linear_idx_type]:
        """Maps a logical coordinate to a linear memory index.

        Parameters:
            index_type: The coordinate type.
            linear_idx_type: The data type for the returned linear index.

        Args:
            index: The logical coordinates to map.

        Returns:
            The linear memory index corresponding to the given coordinates.
        """
        ...

    def idx2crd[
        *,
        out_dtype: DType = DType.int64,
    ](self, idx: Int) -> DynamicCoord[out_dtype, Self.rank]:
        """Maps a linear memory index back to logical coordinates.

        This is the inverse of `__call__` (crd2idx). Given a linear index,
        it computes the corresponding multi-dimensional coordinates.

        Parameters:
            out_dtype: The data type for the output coordinate values.

        Args:
            idx: The linear memory index to convert to coordinates.

        Returns:
            A Coord containing the logical coordinates corresponding to
            the linear index.

        Examples:
            For a layout with shape (3, 4) and row-major strides:
            - layout.idx2crd(0) returns (0, 0).
            - layout.idx2crd(5) returns (1, 1).
            - layout.idx2crd(11) returns (2, 3).
        """
        ...

    def shape_coord(self) -> Coord[*Self._shape_types]:
        """Returns the full shape as a `Coord`.

        Returns:
            A Coord containing all shape dimensions.
        """
        ...

    def stride_coord(self) -> Coord[*Self._stride_types]:
        """Returns the full stride as a `Coord`.

        Returns:
            A Coord containing all stride dimensions.
        """
        ...

    def transpose(
        self,
    ) -> Layout[
        Variadic.reverse[*Self._shape_types],
        Variadic.reverse[*Self._stride_types],
    ]:
        """Transposes the layout by reversing the order of dimensions.

        For an n-dimensional layout, this reverses the order of both shapes
        and strides. For 2D layouts, this swaps rows and columns.

        Returns:
            A new Layout with transposed dimensions.
        """
        ...

    def make_dynamic[
        dtype: DType
    ](self) -> Layout[
        _CoordToDynamic[dtype, *Self._shape_types],
        _CoordToDynamic[dtype, *Self._stride_types],
    ]:
        """Converts all dimensions to runtime values of the given dtype.

        Parameters:
            dtype: The data type for the resulting `RuntimeInt` values.

        Returns:
            A new Layout with all dimensions as `RuntimeInt[dtype]`.
        """
        ...


comptime RowMajorLayout[*shape_types: CoordLike] = Layout[
    shape_types, _RowMajor[*shape_types]
]
"""A `Layout` with row-major (C-order) strides computed from the shape.

Parameters:
    shape_types: The types for the shape dimensions.
"""


struct Layout[
    shape_types: Variadic.TypesOfTrait[CoordLike],
    stride_types: Variadic.TypesOfTrait[CoordLike],
](ImplicitlyCopyable, TensorLayout, TrivialRegisterPassable):
    """A layout that supports mixed compile-time and runtime dimensions.

    This layout provides a unified interface for layouts where some dimensions
    are known at compile time and others are determined at runtime. It enables
    more ergonomic layout definitions while maintaining performance.

    A Layout's shape and strides must be non-negative.

    Parameters:
        shape_types: The types for the shape dimensions.
        stride_types: The types for the stride dimensions.
    """

    var _shape: Coord[*Self.shape_types]
    """The shape of the layout as a Coord."""

    var _stride: Coord[*Self.stride_types]
    """The stride of the layout as a Coord."""

    comptime rank = Variadic.size(Self.shape_types)
    """The number of dimensions in the layout."""
    comptime flat_rank = Variadic.size(_Flattened[*Self.shape_types])
    """The number of dimensions after flattening nested coordinates."""
    comptime shape_known = Coord[*Self.shape_types].all_dims_known
    """Whether all shape dimensions are known at compile time."""
    comptime stride_known = Coord[*Self.stride_types].all_dims_known
    """Whether all stride dimensions are known at compile time."""
    comptime _flat_shape_types = _Flattened[*Self.shape_types]
    comptime _flat_stride_types = _Flattened[*Self.stride_types]
    comptime static_shape[i: Int]: Int = Self._flat_shape_types[i].static_value
    """Returns the compile-time value of the i-th flattened shape dimension.

    Parameters:
        i: The dimension index.
    """
    comptime static_stride[i: Int]: Int = Self._flat_stride_types[
        i
    ].static_value
    """Returns the compile-time value of the i-th flattened stride dimension.

    Parameters:
        i: The dimension index.
    """
    comptime _shape_types: Variadic.TypesOfTrait[CoordLike] = Self.shape_types
    comptime _stride_types: Variadic.TypesOfTrait[CoordLike] = Self.stride_types

    comptime static_product = Coord[*Self.shape_types].static_product
    """The compile-time product of all shape dimensions."""

    comptime static_cosize = _StaticCosize[
        Self._flat_shape_types, Self._flat_stride_types
    ]
    """The compile-time size of the memory region spanned by the layout."""

    def __init__(
        out self,
        shape: Coord[*Self.shape_types],
        stride: Coord[*Self.stride_types],
    ):
        """Initialize a layout with shape and stride.

        Args:
            shape: The shape as a Coord.
            stride: The stride as a Coord.
        """
        comptime assert (
            type_of(shape).__len__() == type_of(stride).__len__()
        ), String(
            (
                "Shape and stride must have the same length, but got shape"
                " length: "
            ),
            type_of(shape).__len__(),
            " stride length: ",
            type_of(stride).__len__(),
        )
        self._shape = shape
        self._stride = stride

    def __call__[
        index_type: CoordLike,
        *,
        linear_idx_type: DType = DType.int64,
    ](self, index: index_type) -> Scalar[linear_idx_type]:
        """Maps a logical coordinate to a linear memory index.

        Supports hierarchical indexing where the coordinate structure can
        differ from the shape structure. For a layout with shape (4, (3, 2)):

        - (1, (1, 1)): exact structure match, each element maps directly.
        - (1, 1): rank-matching, the scalar 1 is decomposed within the
          nested (3, 2) sub-dimension.
        - (1): scalar index decomposed across all dimensions.

        Parameters:
            index_type: The coordinate type.
            linear_idx_type: The data type for the returned linear index.

        Args:
            index: The logical coordinates to map.

        Returns:
            The linear memory index corresponding to the given coordinates.
        """
        comptime index_len = index_type.__len__()

        comptime if index_len == Self.rank:
            # Hierarchical: each coord element maps to one shape dimension.
            # If a shape dimension is nested (e.g., (3, 2)) and the
            # corresponding coord element is a scalar, crd2idx decomposes
            # the scalar within that sub-dimension.
            return crd2idx[out_type=linear_idx_type](
                index, self._shape, self._stride
            )
        elif index_type.is_tuple and index_len > Self.rank:
            # More coord elements than shape dimensions: flatten the coord
            # and strides, then compute a direct element-wise dot product.
            var flat_idx = index.tuple().flatten()
            var flat_stride = self._stride.flatten()
            var result: Scalar[linear_idx_type] = 0

            comptime flat_len = type_of(flat_idx).__len__()
            comptime for i in range(flat_len):
                result += Scalar[linear_idx_type](
                    flat_idx[i].value() * flat_stride[i].value()
                )

            return result
        else:
            # Scalar or single-element coord: decompose against full shape.
            return crd2idx[out_type=linear_idx_type](
                index, self._shape, self._stride
            )

    def idx2crd[
        *,
        out_dtype: DType = DType.int64,
    ](self, idx: Int) -> DynamicCoord[out_dtype, Self.rank]:
        """Maps a linear memory index back to logical coordinates.

        This is the inverse of `__call__` (crd2idx). Given a linear index,
        it computes the corresponding multi-dimensional coordinates using
        the per-element formula: ``coord[i] = (idx // stride[i]) % shape[i]``.

        Parameters:
            out_dtype: The data type for the output coordinate values.

        Args:
            idx: The linear memory index to convert to coordinates.

        Returns:
            A Coord containing the logical coordinates corresponding to the linear index.

        Examples:
            For a layout with shape (3, 4) and row-major strides:
            - layout.idx2crd(0) returns (0, 0).
            - layout.idx2crd(5) returns (1, 1).
            - layout.idx2crd(11) returns (2, 3).
        """
        comptime ResultType = DynamicCoord[out_dtype, Self.rank]
        var result = ResultType()
        var shape_t = self._shape.tuple()
        var stride_t = self._stride.tuple()

        comptime for i in range(Self.rank):
            var coord_val = Int(
                (UInt(idx) // UInt(stride_t[i].value()))
                % UInt(shape_t[i].value())
            )
            UnsafePointer(to=result[i]).init_pointee_copy(
                rebind[ResultType.element_types[i]](
                    RuntimeInt[out_dtype](Scalar[out_dtype](coord_val))
                )
            )
        return result

    def product(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        For a layout with shape (m, n), this returns m * n, representing
        the total number of valid coordinates in the layout.

        Returns:
            The total number of elements in the layout.
        """
        return self._shape.product()

    def size(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        Alias for `product()`. Compatible with the legacy Layout API.

        Returns:
            The total number of elements in the layout.
        """
        return self.product()

    def cosize[
        linear_idx_type: DType = DType.int64
    ](self) -> Scalar[linear_idx_type]:
        """Returns the size of the memory region spanned by the layout.

        For a layout with shape `(m, n)` and stride `(r, s)`, this returns
        `(m-1)*r + (n-1)*s + 1`, representing the memory footprint.

        Parameters:
            linear_idx_type: The data type for the returned size value.

        Returns:
            The size of the memory region required by the layout.
        """
        return (
            self[linear_idx_type=linear_idx_type](Idx(self.product() - 1)) + 1
        )

    def to_layout(self) -> LegacyLayout:
        """Converts this mixed layout to a legacy `Layout` using `IntTuple`.

        Returns:
            A legacy `Layout` with the same shape and stride.
        """
        return LegacyLayout(
            coord_to_int_tuple(self._shape),
            coord_to_int_tuple(self._stride),
        )

    @always_inline("nodebug")
    def reverse(
        self,
    ) -> Layout[
        Variadic.reverse[*Self.shape_types],
        Variadic.reverse[*Self.stride_types],
    ]:
        """Reverse the order of dimensions in the layout.

        Turns row-major into column-major ordering where the stride-1
        dimension comes first, enabling coalesced scalar iteration.

        Returns:
            A new Layout with shape and stride Coords reversed.
        """
        return Layout(self._shape.reverse(), self._stride.reverse())

    @always_inline("nodebug")
    def transpose(
        self,
    ) -> Layout[
        Variadic.reverse[*Self.shape_types],
        Variadic.reverse[*Self.stride_types],
    ]:
        """Transposes the layout by reversing the order of dimensions.

        For an n-dimensional layout, this reverses the order of both shapes
        and strides. For 2D layouts, this swaps rows and columns, converting
        row-major to column-major and vice versa.

        Returns:
            A new Layout with transposed dimensions.

        Example:

        ```mojo
        from layout.tile_layout import row_major

        var layout = row_major[3, 4]()  # shape (3,4), stride (4,1)
        var transposed = layout.transpose()  # shape (4,3), stride (1,4)
        ```
        """
        return self.reverse()

    @always_inline("nodebug")
    def make_dynamic[
        dtype: DType
    ](self) -> Layout[
        _CoordToDynamic[dtype, *Self.shape_types],
        _CoordToDynamic[dtype, *Self.stride_types],
    ]:
        """Convert all elements in shape and stride to RuntimeInt[dtype].

        Parameters:
            dtype: The data type for the resulting RuntimeInt values.

        Returns:
            A new Layout where all elements in shape and stride are
            converted to RuntimeInt[dtype].

        Examples:
            ```mojo
            from layout.tile_layout import row_major
            var layout = row_major[3, 4]()  # All compile-time
            var dynamic = layout.make_dynamic[DType.int64]()
            # dynamic has RuntimeInt[DType.int64] for all dimensions
            ```
        """
        return Layout(
            self._shape.make_dynamic[dtype](),
            self._stride.make_dynamic[dtype](),
        )

    def shape[i: Int](self) -> Self._shape_types[i]:
        """Returns the i-th shape dimension.

        Parameters:
            i: The dimension index.

        Returns:
            The shape value for dimension `i`.
        """
        return self._shape[i]

    def stride[i: Int](self) -> Self._stride_types[i]:
        """Returns the i-th stride dimension.

        Parameters:
            i: The dimension index.

        Returns:
            The stride value for dimension `i`.
        """
        return self._stride[i]

    def shape_coord(self) -> Coord[*Self._shape_types]:
        """Returns the full shape as a `Coord`.

        Returns:
            A Coord containing all shape dimensions.
        """
        return self._shape

    def stride_coord(self) -> Coord[*Self._stride_types]:
        """Returns the full stride as a `Coord`.

        Returns:
            A Coord containing all stride dimensions.
        """
        return self._stride


comptime _StaticCosizeReducer[
    Strides: Variadic.TypesOfTrait[CoordLike],
    Prev: Variadic.ValuesOfType[Int],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.values[
    (From[idx].static_value - 1) * Strides[idx].static_value + Prev[0]
]


comptime _StaticCosize[
    Shapes: Variadic.TypesOfTrait[CoordLike],
    Strides: Variadic.TypesOfTrait[CoordLike],
] = _ReduceVariadicAndIdxToValue[
    BaseVal=Variadic.values[1],
    VariadicType=Shapes,
    Reducer=_StaticCosizeReducer[Strides=Strides, ...],
][
    0
]


comptime _RowMajor[*element_types: CoordLike] = _ReduceVariadicAndIdxToVariadic[
    BaseVal=Variadic.empty_of_trait[CoordLike],
    VariadicType=Variadic.reverse[*element_types],
    Reducer=_RowMajorMapper,
]


comptime _RowMajorMapper[
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.concat_types[
    Variadic.types[T=CoordLike, ComptimeInt[1]] if idx
    == 0 else (
        Variadic.types[
            T=CoordLike,
            RuntimeInt[
                From[idx - 1]
                .DTYPE if not From[idx - 1]
                .is_static_value else Prev[0]
                .DTYPE
            ],
        ] if not From[idx - 1].is_static_value
        or not Prev[0].is_static_value else Variadic.types[
            T=CoordLike,
            ComptimeInt[From[idx - 1].static_value * Prev[0].static_value],
        ]
    ),
    Prev,
]


@always_inline
def row_major(var shape: Coord) -> RowMajorLayout[*shape.element_types]:
    """Creates a row-major layout from a shape `Coord`.

    Row-major means the rightmost dimension has stride 1, and each preceding
    dimension has stride equal to the product of all following dimensions.

    Args:
        shape: The shape as a Coord.

    Returns:
        A Layout with row-major strides.
    """
    # Flatten the shape and compute row-major strides on the flattened representation
    # For now, we keep both shape and strides flat (not nested)

    comptime RowMajorTypes = _RowMajor[*shape.element_types]
    comptime rank = Variadic.size(shape.element_types)

    var strides = Tuple[*RowMajorTypes]()

    # Compute row-major strides on the flattened shape
    # Row-major means rightmost dimension has stride 1,
    # and each preceding dimension has stride equal to the product of all following dimensions
    comptime for i in range(rank):
        comptime idx = rank - 1 - i  # Process in reverse order
        var stride_ptr = UnsafePointer(to=strides[idx])

        comptime if i == 0:
            # Rightmost dimension always has stride 1
            comptime StrideType = RowMajorTypes[idx]
            stride_ptr.init_pointee_copy(rebind[StrideType](Idx[1]()))
        else:
            # Calculate stride as product of shape[idx+1] * stride[idx+1]
            comptime StrideType = RowMajorTypes[idx]

            comptime if StrideType.is_static_value:
                # Stride is compile-time known (both shape and prev stride are compile-time)
                comptime stride_val = StrideType.static_value
                stride_ptr.init_pointee_copy(
                    rebind[StrideType](Idx[stride_val]())
                )
            else:
                # At least one is runtime, compute at runtime
                var stride_val = (
                    shape[idx + 1].value() * strides[idx + 1].value()
                )
                stride_ptr.init_pointee_copy(
                    rebind[StrideType](
                        RuntimeInt[StrideType.DTYPE](
                            Scalar[StrideType.DTYPE](stride_val)
                        )
                    )
                )

    return Layout(shape, Coord(strides^))


@always_inline("nodebug")
def row_major[*idxs: Int]() -> RowMajorLayout[*_IntToComptimeInt[*idxs]]:
    """Creates a row-major layout from compile-time shape dimensions.

    Parameters:
        idxs: The shape dimensions as compile-time integers.

    Returns:
        A Layout with row-major strides.
    """
    var shape = Coord[*_IntToComptimeInt[*idxs]]()
    return row_major(shape)


@always_inline("nodebug")
def row_major(
    idx: ComptimeInt[...],
) -> Layout[
    shape_types=Variadic.types[type_of(idx)],
    stride_types=Variadic.types[ComptimeInt[1]],
]:
    """Creates a 1D row-major layout from a compile-time dimension.

    Args:
        idx: The shape dimension as a `ComptimeInt`.

    Returns:
        A 1D Layout with stride 1.
    """
    return Layout(Coord(idx), Coord(Idx[1]()))


@always_inline("nodebug")
def row_major(
    idx: RuntimeInt[...],
) -> Layout[
    shape_types=Variadic.types[type_of(idx)],
    stride_types=Variadic.types[ComptimeInt[1]],
]:
    """Creates a 1D row-major layout from a runtime dimension.

    Args:
        idx: The shape dimension as a `RuntimeInt`.

    Returns:
        A 1D Layout with stride 1.
    """
    return Layout(Coord(idx), Coord(Idx[1]()))


# ===----------------------------------------------------------------------=== #
# Column Major Layout
# ===----------------------------------------------------------------------=== #


comptime ColMajorLayout[*shape_types: CoordLike] = Layout[
    shape_types, _ColMajor[*shape_types]
]
"""A `Layout` with column-major (Fortran-order) strides computed from the shape.

Parameters:
    shape_types: The types for the shape dimensions.
"""


comptime _ColMajor[*element_types: CoordLike] = _ReduceVariadicAndIdxToVariadic[
    BaseVal=Variadic.empty_of_trait[CoordLike],
    VariadicType=Variadic.types[*element_types],  # Process in forward order
    Reducer=_ColMajorMapper,
]


comptime _ColMajorMapper[
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.concat_types[
    Prev,
    Variadic.types[T=CoordLike, ComptimeInt[1]] if idx
    == 0 else (
        Variadic.types[
            T=CoordLike,
            RuntimeInt[
                From[idx - 1]
                .DTYPE if not From[idx - 1]
                .is_static_value else Prev[idx - 1]
                .DTYPE
            ],
        ] if not From[idx - 1].is_static_value
        or not Prev[idx - 1].is_static_value else Variadic.types[
            T=CoordLike,
            ComptimeInt[
                From[idx - 1].static_value * Prev[idx - 1].static_value
            ],
        ]
    ),
]


@always_inline
def col_major(var shape: Coord) -> ColMajorLayout[*shape.element_types]:
    """Create a column-major layout from a shape.

    Column-major means the first dimension has stride 1, and each subsequent
    dimension has stride equal to the product of all previous dimensions.

    For shape (M, N, K):
    - row_major strides: (N*K, K, 1)
    - col_major strides: (1, M, M*N)

    Args:
        shape: The shape as a Coord.

    Returns:
        A Layout with column-major strides.
    """
    comptime ColMajorTypes = _ColMajor[*shape.element_types]
    comptime rank = Variadic.size(shape.element_types)

    var strides = Tuple[*ColMajorTypes]()

    # Compute column-major strides on the shape
    # Column-major means leftmost dimension has stride 1,
    # and each subsequent dimension has stride = product of all previous dimensions
    comptime for i in range(rank):
        var stride_ptr = UnsafePointer(to=strides[i])

        comptime if i == 0:
            # Leftmost dimension always has stride 1
            comptime StrideType = ColMajorTypes[i]
            stride_ptr.init_pointee_copy(rebind[StrideType](Idx[1]()))
        else:
            # Calculate stride as product of shape[i-1] * stride[i-1]
            comptime StrideType = ColMajorTypes[i]

            comptime if StrideType.is_static_value:
                # Stride is compile-time known
                comptime stride_val = StrideType.static_value
                stride_ptr.init_pointee_copy(
                    rebind[StrideType](Idx[stride_val]())
                )
            else:
                # At least one is runtime, compute at runtime
                var stride_val = shape[i - 1].value() * strides[i - 1].value()
                stride_ptr.init_pointee_copy(
                    rebind[StrideType](
                        RuntimeInt[StrideType.DTYPE](
                            Scalar[StrideType.DTYPE](stride_val)
                        )
                    )
                )

    return Layout(shape, Coord(strides^))


@always_inline("nodebug")
def col_major[*idxs: Int]() -> ColMajorLayout[*_IntToComptimeInt[*idxs]]:
    """Create a column-major layout from compile-time shape dimensions.

    Parameters:
        idxs: The shape dimensions as compile-time integers.

    Returns:
        A Layout with column-major strides.

    Example:

    ```mojo
    from layout.tile_layout import col_major

    var layout = col_major[3, 4]()
    # shape: (3, 4), stride: (1, 3)
    ```
    """
    var shape = Coord[*_IntToComptimeInt[*idxs]]()
    return col_major(shape)


@always_inline("nodebug")
def col_major(
    idx: ComptimeInt[...],
) -> Layout[
    shape_types=Variadic.types[type_of(idx)],
    stride_types=Variadic.types[ComptimeInt[1]],
]:
    """Creates a 1D column-major layout from a compile-time dimension.

    Args:
        idx: The shape dimension as a `ComptimeInt`.

    Returns:
        A 1D Layout with stride 1.
    """
    return Layout(Coord(idx), Coord(Idx[1]()))


@always_inline("nodebug")
def col_major(
    idx: RuntimeInt[...],
) -> Layout[
    shape_types=Variadic.types[type_of(idx)],
    stride_types=Variadic.types[ComptimeInt[1]],
]:
    """Creates a 1D column-major layout from a runtime dimension.

    Args:
        idx: The shape dimension as a `RuntimeInt`.

    Returns:
        A 1D Layout with stride 1.
    """
    return Layout(Coord(idx), Coord(Idx[1]()))


def zipped_divide[
    LayoutType: TensorLayout, //, tile: Coord
](layout: LayoutType) -> ZippedDivideLayout[
    LayoutType._shape_types,
    LayoutType._stride_types,
    tile.element_types,
]:
    """Divides a layout into inner (tile) and outer (number-of-tiles) parts.

    Given a layout and a tile shape, produces a hierarchical layout where the
    inner component has the tile shape with the original strides, and the outer
    component has shape = original_shape / tile with scaled strides.

    Parameters:
        LayoutType: The type of the input layout.
        tile: The tile shape to divide by.

    Args:
        layout: The layout to divide.

    Returns:
        A `ZippedDivideLayout` with inner and outer components.
    """
    var shape = layout.shape_coord()
    var outer_shape = Coord[
        *_Divide[LayoutType._shape_types, tile.element_types]
    ]()
    var outer_stride = Coord[
        *_Multiply[LayoutType._stride_types, tile.element_types]
    ]()
    var inner_shape = tile
    var inner_stride = layout.stride_coord()

    comptime for i in range(outer_shape.rank):
        comptime if (
            outer_shape.element_types[i].is_value
            and not outer_shape.element_types[i].is_static_value
        ):
            outer_shape[i] = rebind[outer_shape.element_types[i]](
                Scalar[outer_shape.element_types[i].DTYPE](
                    shape[i].value() // tile[i].value()
                )
            )

        comptime if (
            outer_stride.element_types[i].is_value
            and not outer_stride.element_types[i].is_static_value
        ):
            outer_stride[i] = rebind[outer_stride.element_types[i]](
                Scalar[outer_stride.element_types[i].DTYPE](
                    inner_stride[i].value() * tile[i].value()
                )
            )
    var out_layout = Layout(
        Coord(inner_shape, outer_shape), Coord(inner_stride, outer_stride)
    )
    return out_layout


comptime ZippedDivideLayout[
    shape_types: Variadic.TypesOfTrait[CoordLike],
    stride_types: Variadic.TypesOfTrait[CoordLike],
    tile: Variadic.TypesOfTrait[CoordLike],
] = Layout[
    Variadic.types[
        T=CoordLike,
        Coord[*tile],  # inner_shape = tile
        Coord[*_Divide[shape_types, tile]],  # outer_shape = shape / tile
    ],
    Variadic.types[
        T=CoordLike,
        Coord[*stride_types],  # inner_stride = original stride
        Coord[*_Multiply[stride_types, tile]],  # outer_stride = stride * tile
    ],
]
"""Type alias for the result of `zipped_divide`.

Splits a layout into inner (tile-sized) and outer (number-of-tiles)
components.

Parameters:
    shape_types: Shape types of the original layout.
    stride_types: Stride types of the original layout.
    tile: Shape types of the tile used to divide the layout.
"""


# ===----------------------------------------------------------------------=== #
# Blocked Product
# ===----------------------------------------------------------------------=== #


comptime BlockedProductLayout[
    block_shape_types: Variadic.TypesOfTrait[CoordLike],
    block_stride_types: Variadic.TypesOfTrait[CoordLike],
    tiler_shape_types: Variadic.TypesOfTrait[CoordLike],
    tiler_stride_types: Variadic.TypesOfTrait[CoordLike],
] = Layout[
    Variadic.types[
        T=CoordLike,
        Coord[*block_shape_types],  # inner_shape = block shape
        Coord[*tiler_shape_types],  # outer_shape = tiler shape
    ],
    Variadic.types[
        T=CoordLike,
        Coord[*block_stride_types],  # inner_stride = block stride
        Coord[
            *_MultiplyByScalar[
                tiler_stride_types,
                # Multiply tiler stride by block cosize (product of block shape)
                Coord[*block_shape_types].static_product,
            ]
        ],  # outer_stride = block.cosize * tiler.stride
    ],
]
"""Type alias for blocked product layout.

Creates a hierarchical layout by combining a block (inner) layout with a
tiler (outer) layout. This is useful for creating tiled memory access patterns.

Parameters:
    block_shape_types: Shape types for the inner block.
    block_stride_types: Stride types for the inner block.
    tiler_shape_types: Shape types for the outer tiler (number of blocks).
    tiler_stride_types: Stride types for the outer tiler.

The result is a layout where:
- inner_shape = block.shape (dimensions within each tile)
- outer_shape = tiler.shape (how many tiles in each dimension)
- inner_stride = block.stride (stride within a tile)
- outer_stride = block.cosize * tiler.stride (stride between tiles)
"""


def blocked_product[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
    //,
](block: BlockLayoutType, tiler: TilerLayoutType) -> BlockedProductLayout[
    BlockLayoutType._shape_types,
    BlockLayoutType._stride_types,
    TilerLayoutType._shape_types,
    TilerLayoutType._stride_types,
]:
    """Creates a blocked layout by combining a block and tiler layout.

    This function creates a hierarchical blocked layout where each element
    of the tiler layout is replaced by a block. This is useful for creating
    tiled layouts for efficient cache utilization.

    Parameters:
        BlockLayoutType: The type of the block layout.
        TilerLayoutType: The type of the tiler layout.

    Args:
        block: The inner layout defining the structure of each tile.
        tiler: The outer layout defining the arrangement of tiles.

    Returns:
        A new layout representing the blocked structure.

    Example:

    ```mojo
    from layout.tile_layout import row_major, blocked_product

    # Create a 2x2 block layout
    var block = row_major[2, 2]()
    # Create a 2x3 tiler (2 rows, 3 cols of blocks)
    var tiler = row_major[2, 3]()
    # Create blocked layout
    var blocked = blocked_product(block, tiler)
    # Result: shape ((2,2), (2,3)), stride ((2,1), (12,4))
    ```
    """
    comptime BlockShape = Coord[*BlockLayoutType._shape_types]
    comptime BlockStride = Coord[*BlockLayoutType._stride_types]
    comptime TilerShape = Coord[*TilerLayoutType._shape_types]
    comptime TilerStride = Coord[*TilerLayoutType._stride_types]
    comptime OuterStrideTypes = _MultiplyByScalar[
        TilerLayoutType._stride_types,
        BlockShape.static_product,
    ]

    # Build inner shape/stride from block layout
    var inner_shape = block.shape_coord()
    var inner_stride = block.stride_coord()

    # Build outer shape from tiler layout
    var outer_shape = tiler.shape_coord()

    # Build outer stride = block.cosize * tiler.stride
    # For row-major block, cosize = product of shape
    # We compute this by multiplying tiler stride by the row-major strides of block shape
    var outer_stride = Coord[*OuterStrideTypes]()

    comptime for i in range(TilerShape.rank):
        comptime if OuterStrideTypes[i].is_static_value:
            # Compile-time known
            UnsafePointer(to=outer_stride[i]).init_pointee_copy(
                rebind[OuterStrideTypes[i]](
                    ComptimeInt[OuterStrideTypes[i].static_value]()
                )
            )
        else:
            # Runtime computation needed
            # outer_stride[i] = tiler.stride[i] * block.cosize
            # For row-major, block.cosize = product of block shape
            var block_cosize = block.shape_coord().product()
            UnsafePointer(to=outer_stride[i]).init_pointee_copy(
                rebind[OuterStrideTypes[i]](
                    RuntimeInt[OuterStrideTypes[i].DTYPE](
                        Scalar[OuterStrideTypes[i].DTYPE](
                            tiler.stride_coord()[i].value() * block_cosize
                        )
                    )
                )
            )

    var result_shape = Coord(inner_shape, outer_shape)
    var result_stride = Coord(inner_stride, outer_stride)

    return Layout(result_shape, result_stride)
