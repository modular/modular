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


trait MixedIntTupleLike(Copyable, Movable):
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

    # TODO(MOCO-2274): This method allows rebinding from MixedIntTupleLike to a
    # MixedIntTuple by retrieving the variadic pack. It is a workaround for
    # rebind causing variadic parameters to be erased.
    @staticmethod
    fn _get_variadic_pack() -> VariadicOf[MixedIntTupleLike]:
        ...


@register_passable("trivial")
struct ComptimeInt[val: Int](MixedIntTupleLike):
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
    fn _get_variadic_pack() -> VariadicOf[MixedIntTupleLike]:
        constrained[False, "ComptimeInt does not have a variadic pack"]()
        return abort[VariadicOf[MixedIntTupleLike]]()


@register_passable("trivial")
struct RuntimeInt[dtype: DType = DType.index](MixedIntTupleLike):
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
    fn _get_variadic_pack() -> VariadicOf[MixedIntTupleLike]:
        constrained[False, "RuntimeInt does not have a variadic pack"]()
        return abort[VariadicOf[MixedIntTupleLike]]()


@always_inline("nodebug")
fn to_mixed_int_tuple[
    T: MixedIntTupleLike
](value: T) -> MixedIntTuple[*T._get_variadic_pack()]:
    """Convert a MixedIntTupleLike value to its corresponding MixedIntTuple type.

    This is a convenience function that performs rebind internally, making the code cleaner
    when working with nested MixedIntTuple types.

    Parameters:
        T: The MixedIntTupleLike type to convert.

    Args:
        value: The value to convert.

    Returns:
        The value rebound as a MixedIntTuple with the appropriate variadic pack.
    """
    return rebind[MixedIntTuple[*T._get_variadic_pack()]](value)


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


struct MixedIntTuple[*element_types: MixedIntTupleLike](
    MixedIntTupleLike, Sized
):
    """A struct representing tuple-like data with compile-time and runtime elements.

    Parameters:
        element_types: The variadic pack of element types that implement `MixedIntTupleLike`.
    """

    # TODO(MOCO-1565): Use a Tuple[*element_types] instead of directly using a variadic pack,
    # therefore eliminating most of the code below.

    alias _mlir_type = __mlir_type[
        `!kgen.pack<:`,
        VariadicOf[MixedIntTupleLike],
        element_types,
        `>`,
    ]

    var storage: Self._mlir_type
    """The underlying MLIR storage for the tuple elements."""

    @staticmethod
    fn _get_variadic_pack() -> VariadicOf[MixedIntTupleLike]:
        return element_types

    @staticmethod
    fn __len__() -> Int:
        """Get the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """

        @parameter
        fn variadic_size(x: VariadicOf[MixedIntTupleLike]) -> Int:
            return __mlir_op.`pop.variadic.size`(x)

        return variadic_size(element_types)

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
        var storage: VariadicPack[_, _, MixedIntTupleLike, *element_types],
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
    fn __getitem__[idx: Int](ref self) -> ref [self] element_types[idx.value]:
        """Get a reference to an element in the tuple.

        Parameters:
            idx: The element index to access.

        Returns:
            A reference to the specified element.
        """
        var storage_ptr = UnsafePointer(to=self.storage).address

        # Get pointer to the element
        var elt_ptr = __mlir_op.`kgen.pack.gep`[index = idx.value](storage_ptr)
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
        constrained[False, "MixedIntTuple is not a value type"]()
        return abort[Int]()

    @always_inline("nodebug")
    fn size(self) -> Int:
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
            "Length of MixedIntTuple (",
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
        *other_types: MixedIntTupleLike
    ](self, other: MixedIntTuple[*other_types]) -> Int:
        """Calculate the inner product with another MixedIntTupleLike.

        Parameters:
            other_types: The types of the other value.

        Args:
            other: The other value to compute inner product with.

        Returns:
            The inner product of the two values.
        """
        constrained[
            Self.__len__() == MixedIntTuple[*other_types].__len__(),
            "Length of MixedIntTuple (",
            String(Self.__len__()),
            ") and MixedIntTuple[*other_types] (",
            String(MixedIntTuple[*other_types].__len__()),
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
                            " of MixedIntTuple must both be a tuple or both be"
                            " a value"
                        ),
                    ),
                ]()

        return result

    @always_inline("nodebug")
    fn __eq__[
        *other_types: MixedIntTupleLike
    ](self, other: MixedIntTuple[*other_types]) -> Bool:
        """Check if this tuple's elements are equal to the other tuple's elements.
        """

        constrained[
            Self.__len__() == MixedIntTuple[*other_types].__len__(),
            "Length of MixedIntTuple (",
            String(Self.__len__()),
            ") and MixedIntTuple[*other_types] (",
            String(MixedIntTuple[*other_types].__len__()),
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
                        " of MixedIntTuple must both be a tuple or both be",
                        " a value",
                    ),
                ]()

        return True

    @always_inline("nodebug")
    fn __ne__[
        *other_types: MixedIntTupleLike
    ](self, other: MixedIntTuple[*other_types]) -> Bool:
        return not self == other


# Implementation based off runtime_tuple.mojo's crd2idx.
fn crd2idx[
    Index: MixedIntTupleLike,
    Shape: MixedIntTupleLike,
    Stride: MixedIntTupleLike,
    out_type: DType = DType.index,
](crd: Index, shape: Shape, stride: Stride) -> Scalar[out_type]:
    """Calculate the index from a coordinate tuple."""
    alias shape_len = Shape.__len__()
    alias stride_len = Stride.__len__()
    alias index_len = Index.__len__()

    @parameter
    if Shape.is_tuple() and Stride.is_tuple() and shape_len == stride_len:
        var result: Scalar[out_type] = 0

        @parameter
        if index_len > 1:  # tuple tuple tuple
            var index_t = to_mixed_int_tuple(crd)
            var shape_t = to_mixed_int_tuple(shape)
            var stride_t = to_mixed_int_tuple(stride)

            @parameter
            for i in range(shape_len):
                result += crd2idx[out_type=out_type](
                    index_t[i], shape_t[i], stride_t[i]
                )

            return result
        else:  # "int" tuple tuple
            var int_crd: Scalar[out_type] = 0 if index_len == 0 else crd.value()

            var shape_t = to_mixed_int_tuple(shape)
            var stride_t = to_mixed_int_tuple(stride)

            alias last_elem_idx = shape_len - 1

            @parameter
            for i in range(last_elem_idx):
                var quotient, remainder = divmod(
                    Int(int_crd), shape_t[i].product()
                )
                result += crd2idx[out_type=out_type](
                    remainder, shape_t[i], stride_t[i]
                )
                int_crd = quotient
            return result + crd2idx[out_type=out_type](
                Int(int_crd), shape_t[last_elem_idx], stride_t[last_elem_idx]
            )
    else:

        @parameter
        if index_len > 1:
            constrained[False, "crd is a tuple but shape and stride are not"]()
            return abort[Scalar[out_type]]()
        else:
            return crd.value() * stride.value()


fn crd2idx[
    Shape: MixedIntTupleLike,
    Stride: MixedIntTupleLike,
    out_type: DType = DType.index,
](crd: IntTuple, shape: Shape, stride: Stride) -> Scalar[out_type]:
    """Calculate the index from a coordinate tuple."""
    alias shape_len = Shape.__len__()
    alias stride_len = Stride.__len__()

    if crd.is_tuple():
        var result: Scalar[out_type] = 0

        @parameter  # tuple tuple tuple
        if Shape.is_tuple() and Stride.is_tuple() and shape_len == stride_len:
            var shape_t = to_mixed_int_tuple(shape)
            var stride_t = to_mixed_int_tuple(stride)

            @parameter
            for i in range(shape_len):
                result += crd2idx[out_type=out_type](
                    crd[i], shape_t[i], stride_t[i]
                )

            return result
        else:
            return abort[Scalar[out_type]](
                String(
                    (
                        "Shape and stride tuple must have same length and have"
                        " length greater than 1 but got shape with length: "
                    ),
                    shape_len,
                    " and stride with length: ",
                    stride_len,
                )
            )
    else:
        var int_crd: Scalar[out_type] = 0 if len(crd) == 0 else crd.value()

        @parameter
        if Shape.is_tuple() and Stride.is_tuple() and shape_len == stride_len:
            # "int" tuple tuple
            var result: Scalar[out_type] = 0

            var shape_t = to_mixed_int_tuple(shape)
            var stride_t = to_mixed_int_tuple(stride)

            alias last_elem_idx = shape_len - 1

            @parameter
            for i in range(last_elem_idx):
                var quotient, remainder = divmod(
                    Int(int_crd), shape_t[i].product()
                )
                result += crd2idx[out_type=out_type](
                    remainder, shape_t[i], stride_t[i]
                )
                int_crd = quotient
            return result + crd2idx[out_type=out_type](
                Int(int_crd), shape_t[last_elem_idx], stride_t[last_elem_idx]
            )
        elif Shape.is_tuple() or Stride.is_tuple():
            constrained[
                False,
                String(
                    (
                        "Shape and stride must both be tuples with same length"
                        " but got shape with length: "
                    ),
                    shape_len,
                    " and stride with length: ",
                    stride_len,
                ),
            ]()
            return abort[Scalar[out_type]]()
        else:  # "int" "int" "int"
            return int_crd * stride.value()
