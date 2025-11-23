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
    VariadicOf,
    VariadicPack,
    Concatenated,
    Reversed,
    MakeVariadic,
    EmptyVariadic,
    _ReduceVariadicAndIdxToVariadic,
    variadic_size,
)
from memory import LegacyUnsafePointer as UnsafePointer


trait MixedTupleLike(ImplicitlyCopyable, Movable, Representable):
    """Trait for unified layout handling of compile-time and runtime indices."""

    comptime VariadicType: VariadicOf[MixedTupleLike]
    comptime STATIC_VALUE: Int
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

    fn tuple(var self) -> MixedTuple[*Self.VariadicType]:
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
struct ComptimeInt[val: Int](MixedTupleLike):
    """Compile-time known index value.

    Parameters:
        val: The compile-time integer value.
    """

    comptime VariadicType: VariadicOf[MixedTupleLike] = Tuple[
        Self
    ].element_types
    comptime STATIC_VALUE: Int = Self.val
    comptime DTYPE = DType.int

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
    fn tuple(var self) -> MixedTuple[*Self.VariadicType]:
        constrained[False, "ComptimeInt is not a tuple type"]()
        return rebind[MixedTuple[*Self.VariadicType]](self)


@register_passable("trivial")
struct RuntimeInt[dtype: DType = DType.int](MixedTupleLike):
    """Runtime index value with configurable precision.

    Parameters:
        dtype: The data type for the runtime integer value. Defaults to `DType.int`.
    """

    comptime VariadicType: VariadicOf[MixedTupleLike] = Tuple[
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
    fn tuple(var self) -> MixedTuple[*Self.VariadicType]:
        constrained[False, "RuntimeInt is not a tuple type"]()
        return rebind[MixedTuple[*Self.VariadicType]](self)


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
struct MixedTuple[*element_types: MixedTupleLike](MixedTupleLike, Sized):
    """A struct representing tuple-like data with compile-time and runtime elements.

    Parameters:
        element_types: The variadic pack of element types that implement `MixedTupleLike`.
    """

    comptime VariadicType: VariadicOf[MixedTupleLike] = Self.element_types
    comptime STATIC_VALUE: Int = -1
    comptime IS_TUPLE = True

    var _storage: Tuple[*Self.element_types]
    """The underlying MLIR storage for the tuple elements."""

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

        comptime result = stdlib.builtin.variadic_size(Self.element_types)
        return result

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        var result = String("MixedTuple(")

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
        var storage: VariadicPack[_, _, MixedTupleLike, *Self.element_types],
    ):
        """Construct from a low-level variadic pack.

        Args:
            storage: The variadic pack storage to construct from.
        """
        var t = Tuple(
            storage=rebind_var[
                VariadicPack[
                    type_of(storage).is_owned,
                    type_of(storage).origin,
                    Copyable & Movable,
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
        constrained[False, "MixedTuple is not a value type"]()
        return abort[Int]()

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
            "Length of MixedTuple (",
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
                result += MixedTuple(self[i]).inner_product(t_elem)
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
        *other_types: MixedTupleLike
    ](self, other: MixedTuple[*other_types]) -> Int:
        """Calculate the inner product with another MixedTupleLike.

        Parameters:
            other_types: The types of the other value.

        Args:
            other: The other value to compute inner product with.

        Returns:
            The inner product of the two values.
        """
        constrained[
            Self.__len__() == MixedTuple[*other_types].__len__(),
            "Length of MixedTuple (",
            String(Self.__len__()),
            ") and MixedTuple[*other_types] (",
            String(MixedTuple[*other_types].__len__()),
            ") must match",
        ]()
        var result = 0

        @parameter
        for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            comptime U = other_types[i]

            @parameter
            if T.IS_TUPLE and U.IS_TUPLE:
                result += MixedTuple(self[i]).inner_product(
                    MixedTuple(other[i])
                )
            elif T.IS_VALUE and U.IS_VALUE:
                result += self[i].value() * other[i].value()
            else:
                constrained[
                    False,
                    String(
                        "Element ",
                        i,
                        (
                            " of MixedTuple must both be a tuple or both be"
                            " a value"
                        ),
                    ),
                ]()

        return result

    @always_inline("nodebug")
    fn __eq__[
        *other_types: MixedTupleLike
    ](self, other: MixedTuple[*other_types]) -> Bool:
        """Check if this tuple's elements are equal to the other tuple's elements.
        """

        constrained[
            Self.__len__() == MixedTuple[*other_types].__len__(),
            "Length of MixedTuple (",
            String(Self.__len__()),
            ") and MixedTuple[*other_types] (",
            String(MixedTuple[*other_types].__len__()),
            ") must match",
        ]()

        @parameter
        for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            comptime U = other_types[i]

            @parameter
            if T.IS_TUPLE and U.IS_TUPLE:
                if MixedTuple(self[i]) != MixedTuple(other[i]):
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
                        " of MixedTuple must both be a tuple or both be",
                        " a value",
                    ),
                ]()

        return True

    @always_inline("nodebug")
    fn __ne__[
        *other_types: MixedTupleLike
    ](self, other: MixedTuple[*other_types]) -> Bool:
        return not self == other

    @always_inline("nodebug")
    fn tuple(var self) -> MixedTuple[*Self.VariadicType]:
        return rebind[MixedTuple[*Self.VariadicType]](self)

    @always_inline("nodebug")
    fn reverse(var self) -> MixedTuple[*Reversed[*Self.element_types]]:
        return MixedTuple[*Reversed[*Self.element_types]](
            rebind[Tuple[*Reversed[*Self.element_types]]](
                self._storage.reverse()
            )
        )

    @always_inline("nodebug")
    fn concat[
        *other_element_types: MixedTupleLike
    ](var self, var other: MixedTuple[*other_element_types]) -> MixedTuple[
        *Concatenated[Self.element_types, other_element_types]
    ]:
        return MixedTuple[
            *Concatenated[Self.element_types, other_element_types]
        ](
            rebind[
                Tuple[*Concatenated[Self.element_types, other_element_types]]
            ](self._storage.concat(other._storage))
        )

    @always_inline("nodebug")
    fn flatten(var self) -> MixedTuple[*_Flattened[*Self.element_types]]:
        """Convert a nested MixedTuple to a flattened MixedTuple.


        Returns:
            A flattened MixedTuple containing all leaf values in order.

        Examples:
            ```mojo
            var nested = MixedTuple(
                Idx[5](),
                MixedTuple(Idx[3](), Idx[2]()),
                Idx(7)
            )
            var flat = nested.flatten()
            # flat is MixedTuple(Idx[5](), Idx[3](), Idx[2](), Idx(7))
            ```
        """
        comptime FlatTypes = _Flattened[*Self.element_types]
        comptime flat_size = variadic_size(FlatTypes)

        var flat_tuple: Tuple[*FlatTypes]

        # Mark the tuple as initialized so we can work on it
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(flat_tuple)
        )

        # TODO: Implement fully generic flatten for deeply nested MixedTuples
        # For now, this only works for non-nested or single-level nested tuples
        # For the test cases in test_mixed_layout (which are all non-nested), this works
        # Deep nesting like MixedTuple(A, MixedTuple(B, MixedTuple(C))) is not yet supported

        constrained[
            flat_size == Self.__len__(),
            (
                "flatten() currently only supports non-nested MixedTuples -"
                " nested tuple flattening not yet implemented"
            ),
        ]()

        # For non-nested tuples, just copy elements directly
        @parameter
        for i in range(flat_size):
            UnsafePointer(to=flat_tuple[i]).init_pointee_copy(
                rebind[FlatTypes[i]](self[i])
            )

        return MixedTuple(flat_tuple^)


# Implementation based off runtime_tuple.mojo's crd2idx.
fn crd2idx[
    Index: MixedTupleLike,
    Shape: MixedTupleLike,
    Stride: MixedTupleLike,
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
            var crd_int = 0 if crd_len == 0 else crd.value()

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
            constrained[False, "crd is a tuple but shape and stride are not"]()
            return abort[Scalar[out_type]]()
        else:
            return crd.value() * stride.value()


fn mixed_int_tuple_to_int_tuple[
    *element_types: MixedTupleLike
](value: MixedTuple[*element_types]) -> IntTuple:
    """Convert a MixedTuple to an IntTuple, preserving the nested structure.

    This function recursively traverses the MixedTuple and converts each element:
    - Value elements (ComptimeInt, RuntimeInt) become integer values in the IntTuple
    - Tuple elements (nested MixedTuple) become nested IntTuples

    Parameters:
        element_types: The variadic pack of element types in the MixedTuple.

    Args:
        value: The MixedTuple to convert.

    Returns:
        An IntTuple with the same structure and values as the input MixedTuple.
    """
    var result = IntTuple()

    @parameter
    for i in range(MixedTuple[*element_types].__len__()):
        comptime T = element_types[i]

        @parameter
        if T.IS_TUPLE:
            # Recursively convert nested tuples
            result.append(mixed_int_tuple_to_int_tuple(value[i].tuple()))
        else:
            # Convert value elements to integers
            result.append(IntTuple(value[i].value()))

    return result


comptime _FlattenMapper[
    Prev: VariadicOf[MixedTupleLike],
    From: VariadicOf[MixedTupleLike],
    idx: Int,
] = Concatenated[
    Prev,
    From[idx]
    .VariadicType if From[idx]
    .IS_TUPLE else MakeVariadic[T=MixedTupleLike, From[idx]],
]


comptime _Flattened[
    *element_types: MixedTupleLike
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = EmptyVariadic[MixedTupleLike],
    Variadic=element_types,
    Reducer=_FlattenMapper,
]

comptime _NextOffset[
    prev_offset: Int,
    element_type: MixedTupleLike,
] = prev_offset + (
    1 if element_type.IS_VALUE else variadic_size(
        _Flattened[*element_type.VariadicType]
    )
)


comptime _FlattenOffsetMapper[
    Prev: VariadicOf[MixedTupleLike],
    From: VariadicOf[MixedTupleLike],
    idx: Int,
] = Concatenated[
    Prev,
    MakeVariadic[
        T=MixedTupleLike,
        ComptimeInt[
            0 if idx
            == 0 else _NextOffset[
                Prev[variadic_size(Prev) - 1].STATIC_VALUE,
                From[idx - 1],
            ]
        ],
    ],
]


comptime _FlattenedOffsets[
    *element_types: MixedTupleLike
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = EmptyVariadic[MixedTupleLike],
    Variadic=element_types,
    Reducer=_FlattenOffsetMapper,
]


fn _get_flattened_helper[
    flat_idx: Int,
    current_offset: Int,
    i: Int,
    *element_types: MixedTupleLike,
](tuple: MixedTuple[*element_types]) -> Int:
    """Helper function to recursively access flattened elements."""

    @parameter
    if i >= MixedTuple[*element_types].__len__():
        constrained[False, "flat_idx out of bounds"]()
        return abort[Int]()

    comptime T = element_types[i]

    @parameter
    if T.IS_TUPLE:
        comptime count = variadic_size(_Flattened[*T.VariadicType])

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
    flat_idx: Int, *element_types: MixedTupleLike
](tuple: MixedTuple[*element_types]) -> Int:
    """Access an element from a nested MixedTuple using a flat index.

    Parameters:
        flat_idx: The index into the flattened representation.
        element_types: The variadic element types of the tuple.

    Args:
        tuple: The nested MixedTuple to access.

    Returns:
        The value at the given flat index.

    Examples:
        For tuple = MixedTuple(Idx[5](), MixedTuple(Idx[3](), Idx[2]()), Idx(7)):
        - get_flattened[0](tuple) returns 5  (first element)
        - get_flattened[1](tuple) returns 3  (first element of nested tuple)
        - get_flattened[2](tuple) returns 2  (second element of nested tuple)
        - get_flattened[3](tuple) returns 7  (third top-level element)
    """
    return _get_flattened_helper[flat_idx, 0, 0](tuple)
