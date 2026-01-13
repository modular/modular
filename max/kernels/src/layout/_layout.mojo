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
    _Splatted,
)
from .int_tuple import IntTuple
from .layout import Layout as LegacyLayout


struct Layout[
    shape_types: Variadic.TypesOfTrait[CoordLike],
    stride_types: Variadic.TypesOfTrait[CoordLike],
](ImplicitlyCopyable):
    """A layout that supports mixed compile-time and runtime dimensions.

    This layout provides a unified interface for layouts where some dimensions
    are known at compile time and others are determined at runtime. It enables
    more ergonomic layout definitions while maintaining performance.

    Parameters:
        shape_types: The types for the shape dimensions.
        stride_types: The types for the stride dimensions.
    """

    var shape: Coord[*Self.shape_types]
    """The shape of the layout as a Coord."""

    var stride: Coord[*Self.stride_types]
    """The stride of the layout as a Coord."""

    comptime rank = Variadic.size(Self.shape_types)
    comptime ALL_DIMS_KNOWN = Coord[*Self.shape_types].ALL_DIMS_KNOWN and Coord[
        *Self.stride_types
    ].ALL_DIMS_KNOWN
    comptime STATIC_PRODUCT = Coord[*Self.shape_types].STATIC_PRODUCT

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
        self.shape = shape
        self.stride = stride

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
        return crd2idx[out_type=linear_idx_type](index, self.shape, self.stride)

    fn idx2crd[
        *,
        out_dtype: DType = DType.int64,
    ](self, idx: Int) -> Coord[*_Splatted[RuntimeInt[out_dtype], Self.rank]]:
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
        return rebind[Coord[*_Splatted[RuntimeInt[out_dtype], Self.rank]]](
            idx2crd[Shape, Stride, out_dtype](idx, self.shape, self.stride)
        )

    fn size(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        For a layout with shape (m, n), this returns m * n, representing
        the total number of valid coordinates in the layout.

        Returns:
            The total number of elements in the layout.
        """
        return self.shape.product()

    fn cosize[
        linear_idx_type: DType = DType.int64
    ](self) -> Scalar[linear_idx_type]:
        """Returns the size of the memory region spanned by the layout.

        For a layout with shape `(m, n)` and stride `(r, s)`, this returns
        `(m-1)*r + (n-1)*s + 1`, representing the memory footprint.

        Returns:
            The size of the memory region required by the layout.
        """
        return self[linear_idx_type=linear_idx_type](Idx(self.size() - 1)) + 1

    fn to_layout(self) -> LegacyLayout:
        return LegacyLayout(
            coord_to_int_tuple(self.shape),
            coord_to_int_tuple(self.stride),
        )


comptime _RowMajor[*element_types: CoordLike] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType = Variadic.reverse[*element_types],
    Reducer=_RowMajorMapper,
]


comptime _RowMajorMapper[
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.concat[
    Variadic.types[T=CoordLike, ComptimeInt[1]] if idx
    == 0 else (
        Variadic.types[
            T=CoordLike,
            RuntimeInt[
                From[idx - 1]
                .DTYPE if not From[idx - 1]
                .IS_STATIC_VALUE else Prev[0]
                .DTYPE
            ],
        ] if not From[idx - 1].IS_STATIC_VALUE
        or not Prev[0].IS_STATIC_VALUE else Variadic.types[
            T=CoordLike,
            ComptimeInt[From[idx - 1].STATIC_VALUE * Prev[0].STATIC_VALUE],
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

    var strides: Tuple[*RowMajorTypes]

    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(strides))

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
            if StrideType.IS_STATIC_VALUE:
                # Stride is compile-time known (both shape and prev stride are compile-time)
                comptime stride_val = StrideType.STATIC_VALUE
                stride_ptr.init_pointee_copy(
                    rebind[StrideType](Idx[stride_val]())
                )
            else:
                # At least one is runtime, compute at runtime
                var stride_val = (
                    shape[idx + 1].value() * strides[idx + 1].value()
                )
                stride_ptr.init_pointee_copy(
                    rebind[StrideType](RuntimeInt[StrideType.DTYPE](stride_val))
                )

    return Layout(shape^, Coord(strides^))


fn row_major[
    *idxs: Int
]() -> Layout[
    shape_types = _IntToComptimeInt[*idxs],
    stride_types = _RowMajor[*_IntToComptimeInt[*idxs]],
]:
    var shape: Coord[*_IntToComptimeInt[*idxs]]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(shape))
    return row_major(shape^)
