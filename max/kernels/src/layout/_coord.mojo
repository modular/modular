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
"""Unified layout system for mixed compile-time and runtime indices."""

from os import abort
from sys.intrinsics import _type_is_eq

from builtin.variadics import (
    Variadic,
    VariadicPack,
    _ReduceVariadicAndIdxToVariadic,
    _ReduceValueAndIdxToVariadic,
    _ReduceVariadicAndIdxToValue,
)
from sys.intrinsics import _type_is_eq_parse_time


trait CoordLike(ImplicitlyCopyable, Representable):
    """Trait for unified layout handling of compile-time and runtime indices."""

    comptime VariadicType: Variadic.TypesOfTrait[CoordLike]
    comptime STATIC_VALUE: Int
    comptime IS_STATIC_VALUE = False
    comptime IS_TUPLE = False
    comptime IS_VALUE = not Self.IS_TUPLE
    comptime DTYPE = DType.invalid

    # Note that unlike the __len__() from Sized, this is a static method.
    @staticmethod
    fn __len__() -> Int:
        """Get the number of elements in this type.

        Returns:
            The number of elements (1 for single values, >1 for tuples).
        """
        ...

    fn __repr__(self) -> String:
        """Get the string representation of this type."""
        ...

    fn value(self) -> Int:
        """Get the value of this type.
        Only valid for value types.
        """
        ...

    fn tuple(var self) -> Coord[*Self.VariadicType]:
        """Get the value of this type.
        Only valid for tuple types.
        """
        ...

    fn product(self) -> Int:
        """Calculate the product of all elements.

        Returns:
            The product of all elements.
        """
        ...

    fn sum(self) -> Int:
        """Calculate the sum of all elements.

        Returns:
            The sum of all elements.
        """
        ...


@register_passable("trivial")
struct ComptimeInt[val: Int](CoordLike):
    """Compile-time known index value.

    Parameters:
        val: The compile-time integer value.
    """

    comptime VariadicType: Variadic.TypesOfTrait[CoordLike] = Tuple[
        Self
    ].element_types
    comptime STATIC_VALUE: Int = Self.val
    comptime DTYPE = DType.int
    comptime IS_STATIC_VALUE = True

    fn __init__(out self):
        """Initialize a compile-time integer with the specified value."""
        pass

    @staticmethod
    @always_inline("nodebug")
    fn __len__() -> Int:
        return 1

    fn __repr__(self) -> String:
        return String("ComptimeInt[", self.value(), "]()")

    @always_inline("nodebug")
    fn product(self) -> Int:
        return self.value()

    @always_inline("nodebug")
    fn sum(self) -> Int:
        return self.value()

    @always_inline("nodebug")
    fn value(self) -> Int:
        return Self.val

    @always_inline("nodebug")
    fn tuple(var self) -> Coord[*Self.VariadicType]:
        constrained[False, "ComptimeInt is not a tuple type"]()
        return rebind[Coord[*Self.VariadicType]](self)


@register_passable("trivial")
struct RuntimeInt[dtype: DType = DType.int](CoordLike):
    """Runtime index value with configurable precision.

    Parameters:
        dtype: The data type for the runtime integer value. Defaults to `DType.int`.
    """

    comptime VariadicType: Variadic.TypesOfTrait[CoordLike] = Tuple[
        Self
    ].element_types
    comptime STATIC_VALUE: Int = -1
    comptime DTYPE = Self.dtype

    var val: Scalar[Self.dtype]
    """The runtime scalar value."""

    fn __init__(out self, value: Scalar[Self.dtype]):
        """Initialize a runtime integer with the given value.

        Args:
            value: The scalar value to store.
        """
        self.val = value

    @staticmethod
    @always_inline("nodebug")
    fn __len__() -> Int:
        return 1

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        return String("RuntimeInt(", self.value(), ")")

    @always_inline("nodebug")
    fn product(self) -> Int:
        return self.value()

    @always_inline("nodebug")
    fn sum(self) -> Int:
        return self.value()

    @always_inline("nodebug")
    fn value(self) -> Int:
        return Int(self.val)

    @always_inline("nodebug")
    fn tuple(var self) -> Coord[*Self.VariadicType]:
        constrained[False, "RuntimeInt is not a tuple type"]()
        return rebind[Coord[*Self.VariadicType]](self)


fn Idx(value: Int) -> RuntimeInt[DType.int]:
    """Helper to create runtime indices.

    Args:
        value: The integer value for the runtime index.

    Returns:
        A `RuntimeInt` instance with the specified value.

    Usage: Idx(5) creates a RuntimeInt with value 5.
    """
    return RuntimeInt[DType.int](value)


fn Idx[value: Int]() -> ComptimeInt[value]:
    """Helper to create compile-time indices.

    Parameters:
        value: The compile-time integer value.

    Returns:
        A `ComptimeInt` instance with the specified compile-time value.

    Usage: Idx[5]() creates a ComptimeInt with value 5.
    """
    return ComptimeInt[value]()


@fieldwise_init("implicit")
struct Coord[*element_types: CoordLike](CoordLike, Sized):
    """A struct representing tuple-like data with compile-time and runtime elements.

    Parameters:
        element_types: The variadic pack of element types that implement `CoordLike`.
    """

    comptime VariadicType: Variadic.TypesOfTrait[CoordLike] = Self.element_types
    comptime STATIC_VALUE: Int = -1
    comptime IS_TUPLE = True
    comptime ALL_DIMS_KNOWN = _AllStatic[*Self.element_types]
    comptime STATIC_PRODUCT = _StaticProduct[*Self.element_types]
    comptime rank = Variadic.size(Self.element_types)

    var _storage: Tuple[*Self.element_types]
    """The underlying MLIR storage for the tuple elements."""

    fn __init__(out self) where Self.ALL_DIMS_KNOWN:
        """
        Empty initialize a tensor with static dims.
        """
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    fn __init__[
        rank: Int, dtype: DType
    ](
        out self: Coord[*_Splatted[RuntimeInt[dtype], rank]],
        index_list: std.utils.IndexList[rank, element_type=dtype],
    ):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

        @parameter
        for i in range(rank):
            UnsafePointer(to=self[i]).init_pointee_copy(
                rebind[type_of(self[i])](RuntimeInt[dtype](index_list[i]))
            )

    @staticmethod
    @always_inline("nodebug")
    fn size() -> Int:
        """Get the total number of elements including nested ones.

        Returns:
            The total count of all elements.
        """
        var count = 0

        @parameter
        for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            count += T.__len__()

        return count

    @staticmethod
    fn __len__() -> Int:
        """Get the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """

        comptime result = Variadic.size(Self.element_types)
        return result

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        var result = String("Coord(")

        @parameter
        for i in range(Self.__len__()):
            result += self[i].__repr__()
            if i < Self.__len__() - 1:
                result += String(", ")
        return result + String(")")

    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """
        return Self.__len__()

    @always_inline("nodebug")
    fn __init__(out self, var *args: * Self.element_types):
        """Construct tuple from variadic arguments.

        Args:
            args: Values for each element.
        """
        self = Self(storage=args^)

    @always_inline("nodebug")
    fn __init__(
        out self,
        *,
        var storage: VariadicPack[_, CoordLike, *Self.element_types],
    ):
        """Construct from a low-level variadic pack.

        Args:
            storage: The variadic pack storage to construct from.
        """
        var t = Tuple(
            storage=rebind_var[
                VariadicPack[
                    elt_is_mutable = type_of(storage).elt_is_mutable,
                    origin = type_of(storage).origin,
                    type_of(storage).is_owned,
                    Movable,
                    *Self.element_types,
                ]
            ](storage^)
        )

        self._storage = rebind[Tuple[*Self.element_types]](t^)

    fn __del__(deinit self):
        """Destructor that destroys all elements."""

        @parameter
        for i in range(Self.__len__()):
            UnsafePointer(to=self[i]).destroy_pointee()

    @always_inline("nodebug")
    fn __getitem__[
        idx: Int
    ](ref self) -> ref [self._storage] Self.element_types[idx]:
        """Get a reference to an element in the tuple.

        Parameters:
            idx: The element index to access.

        Returns:
            A reference to the specified element.
        """
        return self._storage[idx]

    @always_inline("nodebug")
    fn product(self) -> Int:
        var result = 1

        @parameter
        for i in range(Self.__len__()):
            result *= self[i].product()

        return result

    @always_inline("nodebug")
    fn sum(self) -> Int:
        var result = 0

        @parameter
        for i in range(Self.__len__()):
            result += self[i].sum()

        return result

    @always_inline("nodebug")
    fn value(self) -> Int:
        constrained[False, "Coord is not a value type"]()
        abort()

    @always_inline("nodebug")
    fn inner_product(self, t: IntTuple) -> Int:
        """Calculate the inner product with an IntTuple.

        Args:
            t: The other value to compute inner product with.

        Returns:
            The inner product of the two values.
        """

        var result = 0
        debug_assert(
            Self.__len__() == t.__len__(),
            "Length of Coord (",
            Self.__len__(),
            ") and IntTuple (",
            t.__len__(),
            ") must match",
        )

        @parameter
        for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            var t_elem = t[i]

            @parameter
            if T.IS_TUPLE:
                debug_assert(
                    t_elem.is_tuple(),
                    "Type mismatch: expected tuple in t[",
                    i,
                    "] but got value",
                )
                result += Coord(self[i]).inner_product(t_elem)
            else:
                debug_assert(
                    not t_elem.is_tuple(),
                    "Type mismatch: expected value in t[",
                    i,
                    "] but got tuple",
                )
                result += self[i].value() * t_elem.value()
        return result

    @always_inline("nodebug")
    fn inner_product[
        *other_types: CoordLike
    ](self, other: Coord[*other_types]) -> Int:
        """Calculate the inner product with another CoordLike.

        Parameters:
            other_types: The types of the other value.

        Args:
            other: The other value to compute inner product with.

        Returns:
            The inner product of the two values.
        """
        __comptime_assert Self.__len__() == Coord[*other_types].__len__(), (
            "Length of Coord ("
            + String(Self.__len__())
            + ") and Coord[*other_types] ("
            + String(Coord[*other_types].__len__())
            + ") must match"
        )
        var result = 0

        @parameter
        for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            comptime U = other_types[i]

            @parameter
            if T.IS_TUPLE and U.IS_TUPLE:
                result += Coord(self[i]).inner_product(Coord(other[i]))
            elif T.IS_VALUE and U.IS_VALUE:
                result += self[i].value() * other[i].value()
            else:
                constrained[
                    False,
                    String(
                        "Element ",
                        i,
                        " of Coord must both be a tuple or both be a value",
                    ),
                ]()

        return result

    @always_inline("nodebug")
    fn __eq__[
        *other_types: CoordLike
    ](self, other: Coord[*other_types]) -> Bool:
        """Check if this tuple's elements are equal to the other tuple's elements.
        """

        __comptime_assert Self.__len__() == Coord[*other_types].__len__(), (
            "Length of Coord ("
            + String(Self.__len__())
            + ") and Coord[*other_types] ("
            + String(Coord[*other_types].__len__())
            + ") must match"
        )

        @parameter
        for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            comptime U = other_types[i]

            @parameter
            if T.IS_TUPLE and U.IS_TUPLE:
                if Coord(self[i]) != Coord(other[i]):
                    return False
            elif T.IS_VALUE and U.IS_VALUE:
                if self[i].value() != other[i].value():
                    return False
            else:
                constrained[
                    False,
                    String(
                        "Element ",
                        i,
                        " of Coord must both be a tuple or both be",
                        " a value",
                    ),
                ]()

        return True

    @always_inline("nodebug")
    fn __ne__[
        *other_types: CoordLike
    ](self, other: Coord[*other_types]) -> Bool:
        return not self == other

    @always_inline("nodebug")
    fn tuple(var self) -> Coord[*Self.VariadicType]:
        return rebind[Coord[*Self.VariadicType]](self)

    @always_inline("nodebug")
    fn reverse(var self) -> Coord[*Variadic.reverse[*Self.element_types]]:
        return Coord[*Variadic.reverse[*Self.element_types]](
            rebind[Tuple[*Variadic.reverse[*Self.element_types]]](
                self._storage.reverse()
            )
        )

    @always_inline("nodebug")
    fn concat[
        *other_element_types: CoordLike
    ](var self, var other: Coord[*other_element_types]) -> Coord[
        *Variadic.concat[Self.element_types, other_element_types]
    ]:
        return Coord[*Variadic.concat[Self.element_types, other_element_types]](
            rebind[
                Tuple[*Variadic.concat[Self.element_types, other_element_types]]
            ](self._storage.concat(other._storage))
        )

    @always_inline("nodebug")
    fn flatten(var self) -> Coord[*_Flattened[*Self.element_types]]:
        """Convert a nested Coord to a flattened Coord.


        Returns:
            A flattened Coord containing all leaf values in order.

        Examples:
            ```mojo
            from layout._coord import Coord, Idx
            var nested = Coord(
                Idx[5](),
                Coord(Idx[3](), Idx[2]()),
                Idx(7)
            )
            var flat = nested.flatten()
            # flat is Coord(Idx[5](), Idx[3](), Idx[2](), Idx(7))
            ```
        """
        comptime FlatTypes = _Flattened[*Self.element_types]
        comptime flat_size = Variadic.size(FlatTypes)

        var flat_tuple: Tuple[*FlatTypes]

        # Mark the tuple as initialized so we can work on it
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(flat_tuple)
        )

        # TODO: Implement fully generic flatten for deeply nested Coords
        # For now, this only works for non-nested or single-level nested tuples
        # For the test cases in test_mixed_layout (which are all non-nested), this works
        # Deep nesting like Coord(A, Coord(B, Coord(C))) is not yet supported

        __comptime_assert flat_size == Self.__len__(), (
            "flatten() currently only supports non-nested Coords -"
            " nested tuple flattening not yet implemented"
        )

        # For non-nested tuples, just copy elements directly
        @parameter
        for i in range(flat_size):
            UnsafePointer(to=flat_tuple[i]).init_pointee_copy(
                rebind[FlatTypes[i]](self[i])
            )

        return Coord(flat_tuple^)


# Implementation based off runtime_tuple.mojo's crd2idx.
fn crd2idx[
    Index: CoordLike,
    Shape: CoordLike,
    Stride: CoordLike,
    out_type: DType = DType.int64,
](crd: Index, shape: Shape, stride: Stride) -> Scalar[out_type]:
    """Calculate the index from a coordinate tuple."""
    comptime shape_len = Shape.__len__()
    comptime stride_len = Stride.__len__()
    comptime crd_len = Index.__len__()

    @parameter
    if Shape.IS_TUPLE and Stride.IS_TUPLE and shape_len == stride_len:
        var shape_t = shape.tuple()
        var stride_t = stride.tuple()

        var result: Scalar[out_type] = 0

        @parameter
        if crd_len > 1:  # tuple tuple tuple
            var crd_t = crd.tuple()

            @parameter
            for i in range(shape_len):
                result += crd2idx[out_type=out_type](
                    crd_t[i], shape_t[i], stride_t[i]
                )

            return result
        else:  # "int" tuple tuple
            var crd_int: Int

            @parameter
            if Index.IS_TUPLE:
                crd_int = 0 if crd_len == 0 else crd.tuple()[0].value()
            else:
                crd_int = 0 if crd_len == 0 else crd.value()

            comptime last_elem_idx = shape_len - 1

            @parameter
            for i in range(last_elem_idx):
                var quotient, remainder = divmod(crd_int, shape_t[i].product())
                result += crd2idx[out_type=out_type](
                    Idx(remainder), shape_t[i], stride_t[i]
                )
                crd_int = quotient
            return result + crd2idx[out_type=out_type](
                Idx(crd_int), shape_t[last_elem_idx], stride_t[last_elem_idx]
            )
    else:

        @parameter
        if crd_len > 1:
            abort("crd is a tuple but shape and stride are not")
        else:
            return crd.value() * stride.value()


# Implementation based off crd2idx - computes the inverse operation
fn idx2crd[
    Shape: CoordLike,
    Stride: CoordLike,
    out_dtype: DType = DType.int64,
](idx: Int, shape: Shape, stride: Stride) -> Coord[
    *_Splatted[RuntimeInt[out_dtype], Shape.__len__()]
]:
    """Calculate the coordinate tuple from a linear index.

    This is the inverse of crd2idx - given a linear index, shape, and stride,
    it computes the multi-dimensional coordinates.

    Parameters:
        Shape: The shape type (must be CoordLike).
        Stride: The stride type (must be CoordLike).
        out_dtype: The output data type for coordinate values.

    Args:
        idx: The linear index to convert.
        shape: The shape of the tensor.
        stride: The stride of the tensor.

    Returns:
        A Coord containing the coordinate values for each dimension.

    Examples:
        For a 2D tensor with shape (3, 4) and row-major strides (4, 1):

        - idx2crd(0, shape, stride) returns (0, 0).
        - idx2crd(5, shape, stride) returns (1, 1).
        - idx2crd(11, shape, stride) returns (2, 3).
    """
    comptime shape_len = Shape.__len__()
    comptime stride_len = Stride.__len__()

    debug_assert(
        shape_len == stride_len,
        "Shape length (",
        shape_len,
        ") must match stride length (",
        stride_len,
        ")",
    )

    comptime Result = Coord[*_Splatted[RuntimeInt[out_dtype], shape_len]]
    var result: Result
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(result))

    @parameter
    if Shape.IS_TUPLE and Stride.IS_TUPLE and shape_len == stride_len:
        var stride_t = stride.tuple()
        var remaining_idx = idx

        # Process dimensions in order of decreasing stride
        # For each dimension, compute coordinate = remaining_idx // stride
        # then update remaining_idx = remaining_idx % stride
        @parameter
        for i in range(shape_len):
            var stride_val = stride_t[i].value()
            var coord_val = remaining_idx // stride_val
            remaining_idx = remaining_idx % stride_val
            UnsafePointer(to=result[i]).init_pointee_copy(
                rebind[Result.element_types[i]](
                    RuntimeInt[out_dtype](Scalar[out_dtype](coord_val))
                )
            )
    else:
        # Single dimension case
        var coord_val = idx // stride.value()

        @parameter
        for i in range(shape_len):
            UnsafePointer(to=result[i]).init_pointee_copy(
                rebind[Result.element_types[i]](
                    RuntimeInt[out_dtype](Scalar[out_dtype](coord_val))
                )
            )

    return result^


fn coord_to_int_tuple[
    *element_types: CoordLike
](value: Coord[*element_types]) -> IntTuple:
    """Convert a Coord to an IntTuple, preserving the nested structure.

    This function recursively traverses the Coord and converts each element:
    - Value elements (ComptimeInt, RuntimeInt) become integer values in the IntTuple
    - Tuple elements (nested Coord) become nested IntTuples

    Parameters:
        element_types: The variadic pack of element types in the Coord.

    Args:
        value: The Coord to convert.

    Returns:
        An IntTuple with the same structure and values as the input Coord.
    """
    var result = IntTuple()

    @parameter
    for i in range(Coord[*element_types].__len__()):
        comptime T = element_types[i]

        @parameter
        if T.IS_TUPLE:
            # Recursively convert nested tuples
            result.append(coord_to_int_tuple(value[i].tuple()))
        else:
            # Convert value elements to integers
            result.append(IntTuple(value[i].value()))

    return result


@always_inline
fn coord_to_index_list[
    *element_types: CoordLike
](value: Coord[*element_types]) -> std.utils.IndexList[value.rank]:
    """Convert a flat Coord to an IndexList.

    Parameters:
        element_types: The variadic pack of element types in the Coord.

    Args:
        value: The Coord to convert.

    Returns:
        An IndexList with the same rank and values as the input Coord.
    """
    var result = std.utils.IndexList[value.rank]()

    @parameter
    for i in range(Coord[*element_types].__len__()):
        result[i] = value[i].value()

    return result


fn coord_to_int_tuple[*element_types: CoordLike]() -> IntTuple:
    """Convert a Coord to an IntTuple, preserving the nested structure.

    This function recursively traverses the Coord and converts each element:
    - Value elements (ComptimeInt, RuntimeInt) become integer values in the IntTuple
    - Tuple elements (nested Coord) become nested IntTuples

    Parameters:
        element_types: The variadic pack of element types in the Coord.

    Returns:
        An IntTuple with the same structure and values as the input Coord.
    """
    var result = IntTuple()

    @parameter
    for i in range(Variadic.size(element_types)):
        comptime T = element_types[i]

        @parameter
        if T.IS_TUPLE:
            # Recursively convert nested tuples
            result.append(coord_to_int_tuple[element_types[i]]())
        else:

            @parameter
            if T.IS_STATIC_VALUE:
                result.append(IntTuple(T.STATIC_VALUE))
            else:
                result.append(layout.UNKNOWN_VALUE)

    return result


fn coord[
    dtype: DType, *element_types: Movable
](var values: Tuple[*element_types]) -> Coord[
    *_Splatted[RuntimeInt[dtype], type_of(values).__len__()]
] where _AllEqual[Int, *element_types]:
    """Helper to create a Coord from a variadic pack of integers.
    Parameters:
        dtype: The data type for the runtime integer values.
        rank: The number of elements in the tuple.
    Args:
        values: The run-time integer values.
    Returns:
        A `Coord` instance containing `ComptimeInt` elements for each value.
    Usage: coord[5, 3, 2]() creates Coord(ComptimeInt[5](), ComptimeInt[3](), ComptimeInt[2]()).
    """
    var tuple: Coord[*_Splatted[RuntimeInt[dtype], type_of(values).__len__()]]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(tuple))

    @parameter
    for i in range(type_of(values).__len__()):
        UnsafePointer(to=tuple[i]).init_pointee_copy(
            rebind[type_of(tuple[i])](
                RuntimeInt[dtype](Scalar[dtype](rebind[Int](values[i])))
            )
        )
    return tuple^


fn coord[*values: Int]() -> Coord[*_IntToComptimeInt[*values]]:
    """Helper to create a Coord from a variadic pack of integers.
    Parameters:
        values: The compile-time integer values.
    Returns:
        A `Coord` instance containing `ComptimeInt` elements for each value.
    Usage: coord[5, 3, 2]() creates Coord(ComptimeInt[5](), ComptimeInt[3](), ComptimeInt[2]()).
    """
    # values is a ZST since all elements are comptime
    var tuple: Coord[*_IntToComptimeInt[*values]]
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(tuple))
    return tuple^


comptime _FlattenReducer[
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.concat[
    Prev,
    From[idx]
    .VariadicType if From[idx]
    .IS_TUPLE else Variadic.types[T=CoordLike, From[idx]],
]


comptime _Flattened[
    *element_types: CoordLike
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType=element_types,
    Reducer=_FlattenReducer,
]

comptime _NextOffset[
    prev_offset: Int,
    element_type: CoordLike,
] = prev_offset + (
    1 if element_type.IS_VALUE else Variadic.size(
        _Flattened[*element_type.VariadicType]
    )
)


comptime _FlattenOffsetReducer[
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.concat[
    Prev,
    Variadic.types[
        T=CoordLike,
        ComptimeInt[
            0 if idx
            == 0 else _NextOffset[
                Prev[Variadic.size(Prev) - 1].STATIC_VALUE,
                From[idx - 1],
            ]
        ],
    ],
]


comptime _FlattenedOffsets[
    *element_types: CoordLike
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType=element_types,
    Reducer=_FlattenOffsetReducer,
]


fn _get_flattened_helper[
    flat_idx: Int,
    current_offset: Int,
    i: Int,
    *element_types: CoordLike,
](tuple: Coord[*element_types]) -> Int:
    """Helper function to recursively access flattened elements."""

    @parameter
    if i >= Coord[*element_types].__len__():
        constrained[False, "flat_idx out of bounds"]()
        abort()

    comptime T = element_types[i]

    @parameter
    if T.IS_TUPLE:
        comptime count = Variadic.size(_Flattened[*T.VariadicType])

        @parameter
        if flat_idx >= current_offset and flat_idx < current_offset + count:
            return _get_flattened[flat_idx - current_offset](tuple[i].tuple())
        else:
            return _get_flattened_helper[
                flat_idx, current_offset + count, i + 1
            ](tuple)
    else:

        @parameter
        if flat_idx == current_offset:
            return tuple[i].value()
        else:
            return _get_flattened_helper[flat_idx, current_offset + 1, i + 1](
                tuple
            )


fn _get_flattened[
    flat_idx: Int, *element_types: CoordLike
](tuple: Coord[*element_types]) -> Int:
    """Access an element from a nested Coord using a flat index.

    Parameters:
        flat_idx: The index into the flattened representation.
        element_types: The variadic element types of the tuple.

    Args:
        tuple: The nested Coord to access.

    Returns:
        The value at the given flat index.

    Examples:
        For tuple = Coord(Idx[5](), Coord(Idx[3](), Idx[2]()), Idx(7)):
        - get_flattened[0](tuple) returns 5  (first element)
        - get_flattened[1](tuple) returns 3  (first element of nested tuple)
        - get_flattened[2](tuple) returns 2  (second element of nested tuple)
        - get_flattened[3](tuple) returns 7  (third top-level element)
    """
    return _get_flattened_helper[flat_idx, 0, 0](tuple)


comptime _AllStaticReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = (Variadic.values[From[idx].IS_STATIC_VALUE and Prev[0]])


comptime _AllStatic[*element_types: CoordLike] = _ReduceVariadicAndIdxToValue[
    BaseVal = Variadic.values[True],
    VariadicType=element_types,
    Reducer=_AllStaticReducer,
][0]

comptime _AllEqualReducer[
    T: AnyType,
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = (
    Variadic.values[
        _type_is_eq_parse_time[From[idx], T]() and (Prev[0] or idx == 0)
    ]
)


comptime _AllEqual[
    T: AnyType, *element_types: AnyType
] = _ReduceVariadicAndIdxToValue[
    BaseVal = Variadic.values[False],
    VariadicType=element_types,
    Reducer = _AllEqualReducer[T],
][
    0
]

comptime _StaticProductReducer[
    Prev: Variadic.ValuesOfType[Int],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = (Variadic.values[From[idx].STATIC_VALUE * Prev[0]])


comptime _StaticProduct[
    *element_types: CoordLike
] = _ReduceVariadicAndIdxToValue[
    BaseVal = Variadic.values[1],
    VariadicType=element_types,
    Reducer=_StaticProductReducer,
][
    0
]

comptime _IntToComptimeIntMapper[
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.ValuesOfType[Int],
    idx: Int,
] = Variadic.concat[Prev, Variadic.types[ComptimeInt[From[idx]]]]


comptime _IntToComptimeInt[*values: Int] = _ReduceValueAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType=values,
    Reducer=_IntToComptimeIntMapper,
]

comptime _Splatted[T: CoordLike, count: Int] = __mlir_attr[
    `#kgen.variadic.splat<`,
    T,
    `,`,
    count._mlir_value,
    `> : `,
    Variadic.TypesOfTrait[type_of(T)],
]
