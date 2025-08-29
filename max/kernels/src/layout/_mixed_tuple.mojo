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

from builtin.variadics import VariadicOf, VariadicPack
from memory import UnsafePointer
from os import abort
from sys.intrinsics import _type_is_eq


trait MixedTupleLike(Copyable, Movable, Representable):
    """Trait for unified layout handling of compile-time and runtime indices."""

    # Note that unlike the __len__() from Sized, this is a static method.
    @staticmethod
    fn __len__() -> Int:
        """Get the number of elements in this type.

        Returns:
            The number of elements (1 for single values, >1 for tuples).
        """
        ...

    @staticmethod
    fn is_tuple() -> Bool:
        """Check if this type is a tuple."""
        ...

    @staticmethod
    fn is_value() -> Bool:
        """Check if this type is a value."""
        ...

    fn __repr__(self) -> String:
        """Get the string representation of this type."""
        ...

    fn value(self) -> Int:
        """Get the value of this type.
        Only valid for value types.
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

    # TODO(MOCO-2274): This method allows rebinding from MixedTupleLike to a
    # MixedTuple by retrieving the variadic pack. It is a workaround for
    # rebind causing variadic parameters to be erased.
    @staticmethod
    fn _get_variadic_pack() -> VariadicOf[MixedTupleLike]:
        ...


@register_passable("trivial")
struct ComptimeInt[val: Int](MixedTupleLike):
    """Compile-time known index value.

    Parameters:
        val: The compile-time integer value.
    """

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

    @staticmethod
    @always_inline("nodebug")
    fn is_tuple() -> Bool:
        return False

    @staticmethod
    @always_inline("nodebug")
    fn is_value() -> Bool:
        return True

    @always_inline("nodebug")
    fn value(self) -> Int:
        return val

    @staticmethod
    fn _get_variadic_pack() -> VariadicOf[MixedTupleLike]:
        constrained[False, "ComptimeInt does not have a variadic pack"]()
        return abort[VariadicOf[MixedTupleLike]]()


@register_passable("trivial")
struct RuntimeInt[dtype: DType = DType.index](MixedTupleLike):
    """Runtime index value with configurable precision.

    Parameters:
        dtype: The data type for the runtime integer value. Defaults to `DType.index`.
    """

    var val: Scalar[dtype]
    """The runtime scalar value."""

    fn __init__(out self, value: Scalar[dtype]):
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

    @staticmethod
    @always_inline("nodebug")
    fn is_tuple() -> Bool:
        return False

    @staticmethod
    @always_inline("nodebug")
    fn is_value() -> Bool:
        return True

    @always_inline("nodebug")
    fn value(self) -> Int:
        return Int(self.val)

    @staticmethod
    fn _get_variadic_pack() -> VariadicOf[MixedTupleLike]:
        constrained[False, "RuntimeInt does not have a variadic pack"]()
        return abort[VariadicOf[MixedTupleLike]]()


# Note that `to_mixed_int_tuple` isn't a method on MixedTupleLike because it
# calls T._get_variadic_pack(). Putting this in the return type for Compile and
# RuntimeInt be illegal, since the function is constrained False for those types.


@always_inline("nodebug")
fn to_mixed_int_tuple[
    T: MixedTupleLike
](value: T) -> MixedTuple[*T._get_variadic_pack()]:
    """Convert a MixedTupleLike value to its corresponding MixedTuple type.

    This is a convenience function that performs rebind internally, making the code cleaner
    when working with nested MixedTuple types.

    Parameters:
        T: The MixedTupleLike type to convert.

    Args:
        value: The value to convert.

    Returns:
        The value rebound as a MixedTuple with the appropriate variadic pack.
    """
    return rebind[MixedTuple[*T._get_variadic_pack()]](value)


fn Idx(value: Int) -> RuntimeInt[DType.index]:
    """Helper to create runtime indices.

    Args:
        value: The integer value for the runtime index.

    Returns:
        A `RuntimeInt` instance with the specified value.

    Usage: Idx(5) creates a RuntimeInt with value 5.
    """
    return RuntimeInt[DType.index](value)


fn Idx[value: Int]() -> ComptimeInt[value]:
    """Helper to create compile-time indices.

    Parameters:
        value: The compile-time integer value.

    Returns:
        A `ComptimeInt` instance with the specified compile-time value.

    Usage: Idx[5]() creates a ComptimeInt with value 5.
    """
    return ComptimeInt[value]()


struct MixedTuple[*element_types: MixedTupleLike](MixedTupleLike, Sized):
    """A struct representing tuple-like data with compile-time and runtime elements.

    Parameters:
        element_types: The variadic pack of element types that implement `MixedTupleLike`.
    """

    # TODO(MOCO-1565): Use a Tuple[*element_types] instead of directly using a variadic pack,
    # therefore eliminating most of the code below.

    alias _mlir_type = __mlir_type[
        `!kgen.pack<:`,
        VariadicOf[MixedTupleLike],
        element_types,
        `>`,
    ]

    var storage: Self._mlir_type
    """The underlying MLIR storage for the tuple elements."""

    @staticmethod
    fn _get_variadic_pack() -> VariadicOf[MixedTupleLike]:
        return element_types

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
            alias T = element_types[i]
            count += T.__len__()

        return count

    @staticmethod
    fn __len__() -> Int:
        """Get the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """

        alias result = stdlib.builtin.variadic_size(element_types)
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
    fn __init__(out self, var *args: *element_types, __list_literal__: () = ()):
        """Construct tuple from variadic arguments.

        Args:
            args: Values for each element.
            __list_literal__: List literal marker (unused).
        """
        self = Self(storage=args^)

    @always_inline("nodebug")
    fn __init__(
        out self,
        *,
        var storage: VariadicPack[_, _, MixedTupleLike, *element_types],
    ):
        """Construct from a low-level variadic pack.

        Args:
            storage: The variadic pack storage to construct from.
        """
        # Mark our storage as initialized
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )

        # Move each element into the tuple storage.
        @parameter
        fn init_elt[idx: Int](var elt: element_types[idx]):
            UnsafePointer(to=self[idx]).init_pointee_move(elt^)

        storage^.consume_elements[init_elt]()

    fn __del__(deinit self):
        """Destructor that destroys all elements."""

        @parameter
        for i in range(Self.__len__()):
            UnsafePointer(to=self[i]).destroy_pointee()

    @always_inline("nodebug")
    fn __getitem__[idx: Int](ref self) -> ref [self] element_types[idx]:
        """Get a reference to an element in the tuple.

        Parameters:
            idx: The element index to access.

        Returns:
            A reference to the specified element.
        """
        var storage_ptr = UnsafePointer(to=self.storage).address

        # Get pointer to the element
        var elt_ptr = __mlir_op.`kgen.pack.gep`[index = idx._mlir_value](
            storage_ptr
        )
        # Return as reference, propagating mutability
        return UnsafePointer(elt_ptr)[]

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

    @staticmethod
    @always_inline("nodebug")
    fn is_tuple() -> Bool:
        return True

    @staticmethod
    @always_inline("nodebug")
    fn is_value() -> Bool:
        return False

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
            alias T = element_types[i]
            var t_elem = t[i]

            @parameter
            if T.is_tuple():
                debug_assert(
                    t_elem.is_tuple(),
                    "Type mismatch: expected tuple in t[",
                    i,
                    "] but got value",
                )
                result += to_mixed_int_tuple(self[i]).inner_product(t_elem)
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
            alias T = element_types[i]
            alias U = other_types[i]

            @parameter
            if T.is_tuple() and U.is_tuple():
                result += to_mixed_int_tuple(self[i]).inner_product(
                    to_mixed_int_tuple(other[i])
                )
            elif T.is_value() and U.is_value():
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
            alias T = element_types[i]
            alias U = other_types[i]

            @parameter
            if T.is_tuple() and U.is_tuple():
                if to_mixed_int_tuple(self[i]) != to_mixed_int_tuple(other[i]):
                    return False
            elif T.is_value() and U.is_value():
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


# Implementation based off runtime_tuple.mojo's crd2idx.
fn crd2idx[
    Index: MixedTupleLike,
    Shape: MixedTupleLike,
    Stride: MixedTupleLike,
    out_type: DType = DType.index,
](crd: Index, shape: Shape, stride: Stride) -> Scalar[out_type]:
    """Calculate the index from a coordinate tuple."""
    alias shape_len = Shape.__len__()
    alias stride_len = Stride.__len__()
    alias crd_len = Index.__len__()

    @parameter
    if Shape.is_tuple() and Stride.is_tuple() and shape_len == stride_len:
        var shape_t = to_mixed_int_tuple(shape)
        var stride_t = to_mixed_int_tuple(stride)

        var result: Scalar[out_type] = 0

        @parameter
        if crd_len > 1:  # tuple tuple tuple
            var crd_t = to_mixed_int_tuple(crd)

            @parameter
            for i in range(shape_len):
                result += crd2idx[out_type=out_type](
                    crd_t[i], shape_t[i], stride_t[i]
                )

            return result
        else:  # "int" tuple tuple
            var crd_int = 0 if crd_len == 0 else crd.value()

            alias last_elem_idx = shape_len - 1

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
        alias T = element_types[i]

        @parameter
        if T.is_tuple():
            # Recursively convert nested tuples
            result.append(
                mixed_int_tuple_to_int_tuple(to_mixed_int_tuple(value[i]))
            )
        else:
            # Convert value elements to integers
            result.append(IntTuple(value[i].value()))

    return result
