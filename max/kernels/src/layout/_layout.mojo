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
"""Mixed layout implementation that unifies compile-time and runtime indices."""

from os import abort
from sys.intrinsics import _type_is_eq

from builtin.variadics import (
    Variadic,
    VariadicPack,
    _MapVariadicAndIdxToType,
    _ReduceVariadicAndIdxToVariadic,
)

from ._coord import (
    ComptimeInt,
    Idx,
    Coord,
    CoordLike,
    RuntimeInt,
    crd2idx,
    idx2crd,
    coord_to_int_tuple,
    _IntToComptimeInt,
    _CoordToDynamic,
    DynamicCoord,
)
from .int_tuple import IntTuple
from .layout import Layout as LegacyLayout


trait TensorLayout(TrivialRegisterType):
    comptime rank: Int
    comptime shape_known: Bool
    comptime stride_known: Bool
    comptime all_dims_known: Bool = Self.shape_known and Self.stride_known
    comptime static_shape[i: Int]: Int
    comptime static_stride[i: Int]: Int
    comptime _shape_types: Variadic.TypesOfTrait[CoordLike]
    comptime _stride_types: Variadic.TypesOfTrait[CoordLike]

    fn shape[i: Int](self) -> Self._shape_types[i]:
        ...

    fn stride[i: Int](self) -> Self._stride_types[i]:
        ...

    fn product(self) -> Int:
        ...

    fn __call__[
        index_type: CoordLike,
        *,
        linear_idx_type: DType = DType.int64,
    ](self, index: index_type) -> Scalar[linear_idx_type]:
        """Maps a logical coordinate to a linear memory index.

        Args:
            index: An IntTuple representing the logical coordinates to map.

        Returns:
            The linear memory index corresponding to the given coordinates.
        """
        ...

    fn idx2crd[
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
            A Coord containing the logical coordinates corresponding to the linear index.

        Examples:
            For a layout with shape (3, 4) and row-major strides:
            - layout.idx2crd(0) returns (0, 0).
            - layout.idx2crd(5) returns (1, 1).
            - layout.idx2crd(11) returns (2, 3).
        """
        ...

    fn shape_coord(self) -> Coord[*Self._shape_types]:
        ...

    fn stride_coord(self) -> Coord[*Self._stride_types]:
        ...


struct Layout[
    shape_types: Variadic.TypesOfTrait[CoordLike],
    stride_types: Variadic.TypesOfTrait[CoordLike],
](ImplicitlyCopyable, TensorLayout, TrivialRegisterType):
    """A layout that supports mixed compile-time and runtime dimensions.

    This layout provides a unified interface for layouts where some dimensions
    are known at compile time and others are determined at runtime. It enables
    more ergonomic layout definitions while maintaining performance.

    Parameters:
        shape_types: The types for the shape dimensions.
        stride_types: The types for the stride dimensions.
    """

    var _shape: Coord[*Self.shape_types]
    """The shape of the layout as a Coord."""

    var _stride: Coord[*Self.stride_types]
    """The stride of the layout as a Coord."""

    comptime rank = Variadic.size(Self.shape_types)
    comptime shape_known = Coord[*Self.shape_types].all_dims_known
    comptime stride_known = Coord[*Self.stride_types].all_dims_known
    comptime static_shape[i: Int]: Int = Self.shape_types[i].static_value
    comptime static_stride[i: Int]: Int = Self.stride_types[i].static_value
    comptime _shape_types: Variadic.TypesOfTrait[CoordLike] = Self.shape_types
    comptime _stride_types: Variadic.TypesOfTrait[CoordLike] = Self.stride_types

    comptime static_product = Coord[*Self.shape_types].static_product

    fn __init__(
        out self,
        shape: Coord[*Self.shape_types],
        stride: Coord[*Self.stride_types],
    ):
        """Initialize a layout with shape and stride.

        Args:
            shape: The shape as a Coord.
            stride: The stride as a Coord.
        """
        __comptime_assert (
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

    fn __call__[
        index_type: CoordLike,
        *,
        linear_idx_type: DType = DType.int64,
    ](self, index: index_type) -> Scalar[linear_idx_type]:
        """Maps a logical coordinate to a linear memory index.

        Args:
            index: An IntTuple representing the logical coordinates to map.

        Returns:
            The linear memory index corresponding to the given coordinates.
        """
        return crd2idx[out_type=linear_idx_type](
            index, self._shape, self._stride
        )

    fn idx2crd[
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
            A Coord containing the logical coordinates corresponding to the linear index.

        Examples:
            For a layout with shape (3, 4) and row-major strides:
            - layout.idx2crd(0) returns (0, 0).
            - layout.idx2crd(5) returns (1, 1).
            - layout.idx2crd(11) returns (2, 3).
        """
        comptime Shape = Coord[*Self.shape_types]
        comptime Stride = Coord[*Self.stride_types]
        return rebind[DynamicCoord[out_dtype, Self.rank]](
            idx2crd[Shape, Stride, out_dtype](idx, self._shape, self._stride)
        )

    fn product(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        For a layout with shape (m, n), this returns m * n, representing
        the total number of valid coordinates in the layout.

        Returns:
            The total number of elements in the layout.
        """
        return self._shape.product()

    fn cosize[
        linear_idx_type: DType = DType.int64
    ](self) -> Scalar[linear_idx_type]:
        """Returns the size of the memory region spanned by the layout.

        For a layout with shape `(m, n)` and stride `(r, s)`, this returns
        `(m-1)*r + (n-1)*s + 1`, representing the memory footprint.

        Returns:
            The size of the memory region required by the layout.
        """
        return (
            self[linear_idx_type=linear_idx_type](Idx(self.product() - 1)) + 1
        )

    fn to_layout(self) -> LegacyLayout:
        return LegacyLayout(
            coord_to_int_tuple(self._shape),
            coord_to_int_tuple(self._stride),
        )

    @always_inline("nodebug")
    fn make_dynamic[
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
            from layout._layout import row_major
            var layout = row_major[3, 4]()  # All compile-time
            var dynamic = layout.make_dynamic[DType.int64]()
            # dynamic has RuntimeInt[DType.int64] for all dimensions
            ```
        """
        return Layout(
            self._shape.make_dynamic[dtype](),
            self._stride.make_dynamic[dtype](),
        )

    fn shape[i: Int](self) -> Self._shape_types[i]:
        return self._shape[i]

    fn stride[i: Int](self) -> Self._stride_types[i]:
        return self._stride[i]

    fn shape_coord(self) -> Coord[*Self._shape_types]:
        return self._shape

    fn stride_coord(self) -> Coord[*Self._stride_types]:
        return self._stride


comptime _RowMajor[*element_types: CoordLike] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType = Variadic.reverse[*element_types],
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
fn row_major(
    var shape: Coord,
) -> Layout[shape.element_types, _RowMajor[*shape.element_types]]:
    # Flatten the shape and compute row-major strides on the flattened representation
    # For now, we keep both shape and strides flat (not nested)

    comptime RowMajorTypes = _RowMajor[*shape.element_types]
    comptime rank = Variadic.size(shape.element_types)

    var strides = Tuple[*RowMajorTypes]()

    # Compute row-major strides on the flattened shape
    # Row-major means rightmost dimension has stride 1,
    # and each preceding dimension has stride equal to the product of all following dimensions
    @parameter
    for i in range(rank):
        comptime idx = rank - 1 - i  # Process in reverse order
        var stride_ptr = UnsafePointer(to=strides[idx])

        @parameter
        if i == 0:
            # Rightmost dimension always has stride 1
            comptime StrideType = RowMajorTypes[idx]
            stride_ptr.init_pointee_copy(rebind[StrideType](Idx[1]()))
        else:
            # Calculate stride as product of shape[idx+1] * stride[idx+1]
            comptime StrideType = RowMajorTypes[idx]

            @parameter
            if StrideType.is_static_value:
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
fn row_major[
    *idxs: Int
]() -> Layout[
    shape_types = _IntToComptimeInt[*idxs],
    stride_types = _RowMajor[*_IntToComptimeInt[*idxs]],
]:
    var shape = Coord[*_IntToComptimeInt[*idxs]]()
    return row_major(shape)


@always_inline("nodebug")
fn row_major(
    idx: ComptimeInt[...],
) -> Layout[
    shape_types = Variadic.types[type_of(idx)],
    stride_types = Variadic.types[ComptimeInt[1]],
]:
    return Layout(Coord(idx), Coord(Idx[1]()))


@always_inline("nodebug")
fn row_major(
    idx: RuntimeInt[...],
) -> Layout[
    shape_types = Variadic.types[type_of(idx)],
    stride_types = Variadic.types[ComptimeInt[1]],
]:
    return Layout(Coord(idx), Coord(Idx[1]()))
