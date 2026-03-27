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
"""Implements the Tuple type.

These are Mojo built-ins, so you don't need to import them.
"""

from std.builtin.constrained import _constrained_conforms_to
from std.format._utils import (
    write_sequence_to,
    TypeNames,
    FormatStruct,
)
from std.hashlib.hasher import Hasher
from std.reflection.traits import (
    AllCopyable,
    AllEquatable,
    AllHashable,
    AllImplicitlyCopyable,
    AllRegisterPassable,
    AllWritable,
)
from std.sys.intrinsics import _type_is_eq

from std.reflection.type_info import _unqualified_type_name

from std.utils._visualizers import lldb_formatter_wrapping_type

# ===-----------------------------------------------------------------------===#
# Tuple
# ===-----------------------------------------------------------------------===#


@lldb_formatter_wrapping_type
struct Tuple[*element_types: Movable](
    Copyable where AllCopyable[*element_types],
    Equatable where AllEquatable[*element_types],
    Hashable where AllHashable[*element_types],
    # TODO(MOCO-3421): AllImplicitlyCopyable implies AllCopyable since
    # ImplicitlyCopyable refines Copyable, but the compiler can't infer
    # parent trait constraints from derived ones yet. Remove AllCopyable
    # from this where clause once that's fixed.
    ImplicitlyCopyable where (
        AllImplicitlyCopyable[*element_types] and AllCopyable[*element_types]
    ),
    # ImplicitlyDestructible and Movable are listed explicitly because
    # conditional conformances require all conformances to be stated.
    ImplicitlyDestructible,
    Movable,
    RegisterPassable where AllRegisterPassable[*element_types],
    Sized,
    Writable where AllWritable[*element_types],
):
    """The type of a literal tuple expression.

    A tuple consists of zero or more values, separated by commas.

    Parameters:
        element_types: The elements type.
    """

    comptime _mlir_type = __mlir_type[
        `!kgen.pack<:`,
        Variadic.TypesOfTrait[Movable],
        Self.element_types,
        `>`,
    ]

    var _mlir_value: Self._mlir_type
    """The underlying storage for the tuple."""

    # Overload that crushes down IR generated on the caller side.
    @always_inline("nodebug")
    def __init__(out self: Tuple[]):
        """Construct an empty tuple."""
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

    @always_inline("nodebug")
    def __init__(out self, var *args: * Self.element_types):
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self = Self(storage=args^)

    @always_inline("nodebug")
    def __init__(
        out self,
        *,
        var storage: VariadicPack[_, Movable, *Self.element_types],
    ):
        """Construct the tuple from a low-level internal representation.

        Args:
            storage: The variadic pack storage to construct from.
        """

        # Mark 'self._mlir_value' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        # Move each element into the tuple storage.
        @parameter
        def init_elt[idx: Int](var elt: Self.element_types[idx]):
            UnsafePointer(to=self[idx]).init_pointee_move(elt^)

        storage^.consume_elements[init_elt]()

    def __del__(deinit self):
        """Destructor that destroys all of the elements."""

        # Run the destructor on each member, the destructor of !kgen.pack is
        # trivial and won't do anything.
        comptime for i in range(Self.__len__()):
            comptime TUnknown = Self.element_types[i]
            _constrained_conforms_to[
                conforms_to(TUnknown, ImplicitlyDestructible),
                Parent=Self,
                Element=TUnknown,
                ParentConformsTo="ImplicitlyDestructible",
            ]()
            UnsafePointer(
                to=trait_downcast[ImplicitlyDestructible](self[i])
            ).destroy_pointee()

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """Copy construct the tuple.

        Args:
            copy: The value to copy from.
        """
        # Mark '_mlir_value' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        comptime for i in range(Self.__len__()):
            # TODO: We should not use self[i] as this returns a reference to
            # uninitialized memory.
            UnsafePointer(
                to=trait_downcast[Copyable](self[i])
            ).init_pointee_copy(trait_downcast[Copyable](copy[i]))

    @always_inline("nodebug")
    def __init__(out self, *, deinit take: Self):
        """Move construct the tuple.

        Args:
            take: The value to move from.
        """
        # Mark '_mlir_value' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        comptime for i in range(Self.__len__()):
            # TODO: We should not use self[i] as this returns a reference to
            # uninitialized memory.
            UnsafePointer(to=self[i]).init_pointee_move_from(
                UnsafePointer(to=take[i])
            )
        # Note: The destructor on `take` is auto-disabled in a moveinit.

    @always_inline("builtin")
    @staticmethod
    def __len__() -> Int:
        """Return the number of elements in the tuple.

        Returns:
            The tuple length.
        """

        comptime result = Variadic.size(Self.element_types)
        return result

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Get the number of elements in the tuple.

        Returns:
            The tuple length.
        """
        return Self.__len__()

    @always_inline("nodebug")
    def __getitem_param__[
        idx: Int
    ](ref self) -> ref[self] Self.element_types[idx]:
        """Get a reference to an element in the tuple.

        Parameters:
            idx: The element to return.

        Returns:
            A reference to the specified element.
        """
        # Return a reference to an element at the specified index, propagating
        # mutability of self.
        var storage_kgen_ptr = UnsafePointer(to=self._mlir_value).address

        # KGenPointer to the element.
        var elt_kgen_ptr = __mlir_op.`kgen.pack.gep`[
            index=idx.__mlir_index__()
        ](storage_kgen_ptr)
        return UnsafePointer[_, origin_of(self)](elt_kgen_ptr)[]

    @always_inline("nodebug")
    def __contains__[T: Equatable](self, value: T) -> Bool:
        """Return whether the tuple contains the specified value.

        For example:

        ```mojo
        var t = Tuple(True, 1, 2.5)
        if 1 in t:
            print("t contains 1")
        ```

        Args:
            value: The value to search for.

        Parameters:
            T: The type of the value.

        Returns:
            True if the value is in the tuple, False otherwise.
        """

        comptime for i in range(type_of(self).__len__()):
            comptime if _type_is_eq[Self.element_types[i], T]():
                if rebind[T](self[i]) == value:
                    return True

        return False

    @always_inline("nodebug")
    def __init__[
        *elt_types: Movable & Defaultable
    ](out self: Tuple[*elt_types]):
        """Construct a tuple with default-initialized elements.

        Parameters:
            elt_types: The types of the elements contained in the Tuple.
        """

        # Mark 'self._mlir_value' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        comptime for i in range(type_of(self).__len__()):
            UnsafePointer(to=self[i]).init_pointee_move(elt_types[i]())

    @always_inline
    def __eq__(
        self, other: Self
    ) -> Bool where AllEquatable[*Self.element_types]:
        """Compare this tuple to another tuple using equality comparison.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is equal to the other tuple, False otherwise.
        """
        comptime for i in range(type_of(self).__len__()):
            if trait_downcast[Equatable](self[i]) != trait_downcast[Equatable](
                other[i]
            ):
                return False
        return True

    @always_inline
    def __eq__[
        self_elt_types: Variadic.TypesOfTrait[Movable & Equatable],
        other_elt_types: Variadic.TypesOfTrait[Movable & Equatable],
    ](self: Tuple[*self_elt_types], other: Tuple[*other_elt_types]) -> Bool:
        """Compare this tuple to another tuple of a potentially different type.

        This overload enables cross-type and cross-length comparisons such as
        `(1, "a") == (1, "b")` or `(1, 2, 3) == (1, 2)`.

        Parameters:
            self_elt_types: The types of the elements in this tuple.
            other_elt_types: The types of the elements in the other tuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if the tuples are equal, False otherwise.
        """
        comptime self_len = type_of(self).__len__()
        comptime other_len = type_of(other).__len__()

        comptime if self_len != other_len:
            return False

        comptime for i in range(type_of(self).__len__()):
            comptime self_type = type_of(self[i])
            comptime other_type = type_of(other[i])
            comptime assert _type_is_eq[
                self_type, other_type
            ](), "Tuple elements must be of the same type to compare."
            if self[i] != rebind[self_type](other[i]):
                return False
        return True

    @always_inline
    def __ne__[
        self_elt_types: Variadic.TypesOfTrait[Movable & Equatable],
        other_elt_types: Variadic.TypesOfTrait[Movable & Equatable],
    ](self: Tuple[*self_elt_types], other: Tuple[*other_elt_types]) -> Bool:
        """Compare this tuple to another tuple of a potentially different type.

        Parameters:
            self_elt_types: The types of the elements in this tuple.
            other_elt_types: The types of the elements in the other tuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if the tuples are not equal, False otherwise.
        """
        return not self == other

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where AllHashable[*Self.element_types]:
        """Hashes the tuple using the given hasher.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        comptime for i in range(type_of(self).__len__()):
            trait_downcast[Hashable](self[i]).__hash__(hasher)

    @no_inline
    def _write_tuple_to[
        *, is_repr: Bool
    ](self, mut writer: Some[Writer]) where AllWritable[*Self.element_types]:
        """Write this tuple's elements to a writer.

        Parameters:
            is_repr: Whether to use repr formatting for elements.

        Args:
            writer: The writer to write to.
        """

        @parameter
        def elements[i: Int](mut writer: Some[Writer]):
            comptime if is_repr:
                trait_downcast[Writable](self[i]).write_repr_to(writer)
            else:
                trait_downcast[Writable](self[i]).write_to(writer)

        write_sequence_to[
            size=Self.__len__(),
            ElementFn=elements,
        ](writer, open="", close="")

        comptime if Self.__len__() == 1:
            writer.write_string(",")

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where AllWritable[*Self.element_types]:
        """Write this tuple's text representation to a writer.

        Elements are formatted using their `write_to()` representation.
        Single-element tuples include a trailing comma: `(1,)`.

        Args:
            writer: The writer to write to.
        """
        writer.write_string("(")
        self._write_tuple_to[is_repr=False](writer)
        writer.write_string(")")

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where AllWritable[*Self.element_types]:
        """Write this tuple's debug representation to a writer.

        Outputs the type name and parameters followed by elements formatted
        using their `write_repr_to()` representation. For example,
        `Tuple[Int, String](Int(0), 'hello')`.

        Args:
            writer: The writer to write to.
        """

        @parameter
        def fields(mut w: Some[Writer]):
            self._write_tuple_to[is_repr=True](w)

        FormatStruct(writer, "Tuple").params(
            TypeNames[*Self.element_types]()
        ).fields[
            FieldsFn=fields,
        ]()

    @always_inline
    def _compare[
        self_elt_types: Variadic.TypesOfTrait[Movable & Comparable],
        other_elt_types: Variadic.TypesOfTrait[Movable & Comparable],
    ](self: Tuple[*self_elt_types], other: Tuple[*other_elt_types]) -> Int:
        comptime self_len = type_of(self).__len__()
        comptime other_len = type_of(other).__len__()

        comptime if other_len == 0:
            return 1 if self_len > 0 else 0

        comptime min_length = min(self_len, other_len)

        comptime for i in range(min_length):
            comptime self_type = type_of(self[i])
            comptime other_type = type_of(other[i])
            comptime assert _type_is_eq[self_type, other_type](), String(
                "Mismatch between tuple elements at index ",
                i,
                " must be of the same type to compare.",
            )
            if self[i] < rebind[self_type](other[i]):
                return -1
            if rebind[self_type](other[i]) < self[i]:
                return 1

        comptime if self_len < other_len:
            return -1
        elif self_len > other_len:
            return 1
        else:
            return 0

    @always_inline
    def __lt__[
        self_elt_types: Variadic.TypesOfTrait[Movable & Comparable],
        other_elt_types: Variadic.TypesOfTrait[Movable & Comparable],
        //,
    ](self: Tuple[*self_elt_types], other: Tuple[*other_elt_types]) -> Bool:
        """Compare this tuple to another tuple using less than comparison.

        Parameters:
            self_elt_types: The types of the elements contained in the Tuple.
            other_elt_types: The types of the elements contained in the other Tuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is less than the other tuple, False otherwise.
        """
        return self._compare(other) < 0

    @always_inline
    def __le__[
        self_elt_types: Variadic.TypesOfTrait[Movable & Comparable],
        other_elt_types: Variadic.TypesOfTrait[Movable & Comparable],
        //,
    ](self: Tuple[*self_elt_types], other: Tuple[*other_elt_types]) -> Bool:
        """Compare this tuple to another tuple using less than or equal to comparison.

        Parameters:
            self_elt_types: The types of the elements contained in the Tuple.
            other_elt_types: The types of the elements contained in the other Tuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is less than or equal to the other tuple, False otherwise.
        """
        return self._compare(other) <= 0

    @always_inline
    def __gt__[
        self_elt_types: Variadic.TypesOfTrait[Movable & Comparable],
        other_elt_types: Variadic.TypesOfTrait[Movable & Comparable],
        //,
    ](self: Tuple[*self_elt_types], other: Tuple[*other_elt_types]) -> Bool:
        """Compare this tuple to another tuple using greater than comparison.

        Parameters:
            self_elt_types: The types of the elements contained in the Tuple.
            other_elt_types: The types of the elements contained in the other
                Tuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is greater than the other tuple, False otherwise.
        """

        return self._compare(other) > 0

    @always_inline
    def __ge__[
        self_elt_types: Variadic.TypesOfTrait[Movable & Comparable],
        other_elt_types: Variadic.TypesOfTrait[Movable & Comparable],
        //,
    ](self: Tuple[*self_elt_types], other: Tuple[*other_elt_types]) -> Bool:
        """Compare this tuple to another tuple using greater than or equal to comparison.

        Parameters:
            self_elt_types: The types of the elements contained in the Tuple.
            other_elt_types: The types of the elements contained in the other Tuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is greater than or equal to the other tuple, False otherwise.
        """

        return self._compare(other) >= 0

    @always_inline("nodebug")
    def reverse(
        deinit self, out result: Tuple[*Variadic.reverse[*Self.element_types]]
    ):
        """Return a new tuple with the elements in reverse order.

        Returns:
            A new tuple with the elements in reverse order.

        Usage:

        ```mojo
        image_coords = Tuple[Int, Int](100, 200) # row-major indexing
        screen_coords = image_coords.reverse() # (col, row) for x,y display
        print(screen_coords[0], screen_coords[1]) # output: 200, 100
        ```
        """
        # Mark 'result' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime for i in range(type_of(result).__len__()):
            UnsafePointer(to=result[i]).init_pointee_move_from(
                rebind[UnsafePointer[type_of(result[i]), origin_of(self)]](
                    UnsafePointer(
                        to=self[Variadic.size(Self.element_types) - 1 - i]
                    )
                )
            )

    @always_inline("nodebug")
    def concat[
        *other_element_types: Movable
    ](
        deinit self,
        deinit other: Tuple[*other_element_types],
        out result: Tuple[
            *Variadic.concat_types[Self.element_types, other_element_types]
        ],
    ):
        """Return a new tuple that concatenates this tuple with another.

        Args:
            other: The other tuple to concatenate.

        Parameters:
            other_element_types: The types of the elements contained in the other Tuple.

        Returns:
            A new tuple with the concatenated elements.

        Usage:

        ```
        var rgb = Tuple[Int, Int, Int](0xFF, 0xF0, 0x0)
        var rgba = rgb.concat(Tuple[Int](0xFF)) # Adds alpha channel
        print(rgba[0], rgba[1], rgba[2], rgba[3]) # 255 240 0 255
        ```
        """
        # Mark 'result' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime self_len = Self.__len__()

        comptime for i in range(self_len):
            UnsafePointer(to=result[i]).init_pointee_move_from(
                rebind[UnsafePointer[type_of(result[i]), origin_of(self)]](
                    UnsafePointer(to=self[i])
                )
            )

        comptime for i in range(type_of(other).__len__()):
            UnsafePointer(to=result[self_len + i]).init_pointee_move_from(
                rebind[
                    UnsafePointer[
                        type_of(result[self_len + i]), origin_of(other)
                    ]
                ](UnsafePointer(to=other[i]))
            )
