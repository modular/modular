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
    VariadicOf,
    VariadicPack,
    Concatenated,
    Reversed,
    MakeVariadic,
    EmptyVariadic,
    variadic_size,
    _MapVariadicAndIdxToType,
    _ReduceVariadicAndIdxToVariadic,
)

from ._mixed_tuple import (
    ComptimeInt,
    Idx,
    MixedTuple,
    MixedTupleLike,
    RuntimeInt,
    _Flattened,
    crd2idx,
    mixed_int_tuple_to_int_tuple,
    _FlattenedOffsets,
)
from .int_tuple import IntTuple
from .layout import LayoutTrait


struct MixedLayout[
    shape_types: VariadicOf[MixedTupleLike],
    stride_types: VariadicOf[MixedTupleLike],
](ImplicitlyCopyable, Movable):
    """A layout that supports mixed compile-time and runtime dimensions.

    This layout provides a unified interface for layouts where some dimensions
    are known at compile time and others are determined at runtime. It enables
    more ergonomic layout definitions while maintaining performance.

    Parameters:
        shape_types: The types for the shape dimensions.
        stride_types: The types for the stride dimensions.
    """

    var shape: MixedTuple[*Self.shape_types]
    """The shape of the layout as a mixed tuple."""

    var stride: MixedTuple[*Self.stride_types]
    """The stride of the layout as a mixed tuple."""

    comptime rank = variadic_size(Self.shape_types)

    fn __init__(
        out self,
        shape: MixedTuple[*Self.shape_types],
        stride: MixedTuple[*Self.stride_types],
    ):
        """Initialize a mixed layout with shape and stride.

        Args:
            shape: The shape as a MixedTuple.
            stride: The stride as a MixedTuple.
        """
        constrained[
            type_of(shape).__len__() == type_of(stride).__len__(),
            String(
                (
                    "Shape and stride must have the same length, but got shape"
                    " length: "
                ),
                type_of(shape).__len__(),
                " stride length: ",
                type_of(stride).__len__(),
            ),
        ]()
        self.shape = shape
        self.stride = stride

    fn __call__[
        index_type: MixedTupleLike,
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

    fn to_layout(self) -> Layout:
        return Layout(
            mixed_int_tuple_to_int_tuple(self.shape),
            mixed_int_tuple_to_int_tuple(self.stride),
        )


comptime _RowMajor[
    *element_types: MixedTupleLike
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = EmptyVariadic[MixedTupleLike],
    Variadic = Reversed[*_Flattened[*element_types]],
    Reducer=_RowMajorMapper,
]


comptime _RowMajorMapper[
    Prev: VariadicOf[MixedTupleLike],
    From: VariadicOf[MixedTupleLike],
    idx: Int,
] = Concatenated[
    MakeVariadic[T=MixedTupleLike, ComptimeInt[1]] if idx
    == 0 else (
        MakeVariadic[
            T=MixedTupleLike,
            RuntimeInt[
                From[idx - 1].DTYPE if From[idx - 1].STATIC_VALUE
                == -1 else Prev[0].DTYPE
            ],
        ] if From[idx - 1].STATIC_VALUE
        == -1
        or Prev[0].STATIC_VALUE
        == -1 else MakeVariadic[
            T=MixedTupleLike,
            ComptimeInt[From[idx - 1].STATIC_VALUE * Prev[0].STATIC_VALUE],
        ]
    ),
    Prev,
]


fn unflatten[
    structure_types: VariadicOf[MixedTupleLike],
    flat_types: VariadicOf[MixedTupleLike],
](flat_tuple: MixedTuple[*flat_types]) -> MixedTuple[*structure_types]:
    """Unflatten a flat tuple back to match a nested structure.

    This reconstructs a nested tuple structure by extracting elements from a
    flat tuple based on the structure template. The values come from flat_tuple,
    but the nesting structure matches structure_types.

    Parameters:
        structure_types: The desired nested structure (values ignored, only structure used).
        flat_types: The types of the flat tuple elements.

    Args:
        flat_tuple: The flat tuple containing the actual values.

    Returns:
        A nested tuple with the structure of structure_types and values from flat_tuple.
    """
    comptime Offsets = _FlattenedOffsets[*structure_types]
    var result_tuple: Tuple[*structure_types]

    __mlir_op.`lit.ownership.mark_initialized`(
        __get_mvalue_as_litref(result_tuple)
    )

    @parameter
    for i in range(variadic_size(structure_types)):
        var result_ptr = UnsafePointer(to=result_tuple[i])
        comptime T = structure_types[i]
        comptime offset = Offsets[i].STATIC_VALUE

        @parameter
        if T.IS_TUPLE:
            # For nested tuples, we need to recursively unflatten
            # Extract the slice of flat_tuple that corresponds to this nested element
            comptime count = variadic_size(_Flattened[*T.VariadicType])

            # Build a tuple containing just the elements for this nested structure
            var nested_flat: Tuple[*_Flattened[*T.VariadicType]]
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(nested_flat)
            )

            @parameter
            for j in range(count):
                var nested_flat_ptr = UnsafePointer(to=nested_flat[j])
                nested_flat_ptr.init_pointee_copy(
                    rebind[type_of(nested_flat).element_types[j]](
                        flat_tuple[offset + j]
                    )
                )

            var nested = unflatten[T.VariadicType, _Flattened[*T.VariadicType]](
                MixedTuple(nested_flat^)
            )
            result_ptr.init_pointee_move(rebind[T](nested^))
        else:
            # For leaf values, copy directly from flat_tuple
            result_ptr.init_pointee_copy(rebind[T](flat_tuple[offset]))

    return MixedTuple(result_tuple^)


@always_inline
fn row_major(
    var tuple: MixedTuple,
) -> MixedLayout[
    _Flattened[*tuple.element_types], _RowMajor[*tuple.element_types]
]:
    # Flatten the shape and compute row-major strides on the flattened representation
    # For now, we keep both shape and strides flat (not nested)

    var flat_shape = tuple.flatten()
    comptime FlatTypes = _Flattened[*tuple.element_types]
    comptime RowMajorTypes = _RowMajor[*tuple.element_types]
    comptime flat_rank = variadic_size(FlatTypes)

    var flat_strides: Tuple[*RowMajorTypes]

    __mlir_op.`lit.ownership.mark_initialized`(
        __get_mvalue_as_litref(flat_strides)
    )

    # Compute row-major strides on the flattened shape
    # Row-major means rightmost dimension has stride 1,
    # and each preceding dimension has stride equal to the product of all following dimensions
    @parameter
    for i in range(flat_rank):
        comptime idx = flat_rank - 1 - i  # Process in reverse order
        var stride_ptr = UnsafePointer(to=flat_strides[idx])

        @parameter
        if i == 0:
            # Rightmost dimension always has stride 1
            comptime StrideType = RowMajorTypes[idx]
            stride_ptr.init_pointee_copy(rebind[StrideType](Idx[1]()))
        else:
            # Calculate stride as product of shape[idx+1] * stride[idx+1]
            comptime StrideType = RowMajorTypes[idx]

            @parameter
            if StrideType.STATIC_VALUE != -1:
                # Stride is compile-time known (both shape and prev stride are compile-time)
                comptime stride_val = StrideType.STATIC_VALUE
                stride_ptr.init_pointee_copy(
                    rebind[StrideType](Idx[stride_val]())
                )
            else:
                # At least one is runtime, compute at runtime
                var stride_val = (
                    flat_shape[idx + 1].value() * flat_strides[idx + 1].value()
                )
                stride_ptr.init_pointee_copy(
                    rebind[StrideType](RuntimeInt[StrideType.DTYPE](stride_val))
                )

    return MixedLayout(flat_shape^, MixedTuple(flat_strides^))
