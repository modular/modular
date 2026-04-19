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
"""Implements StaticTuple, a statically-sized uniform container.

You can import these APIs from the `utils` package. For example:

```mojo
from std.utils import StaticTuple
```
"""

from std.builtin.device_passable import DevicePassable
from std.compile import get_type_name

# ===----------------------------------------------------------------------=== #
# StaticTuple
# ===----------------------------------------------------------------------=== #

comptime _StaticTupleTraits = ImplicitlyCopyable & ImplicitlyDestructible & RegisterPassable
"""The required trait conformances for a StaticTuple's element type."""


def _static_tuple_construction_checks[T: _StaticTupleTraits, size: Int]():
    """Checks if the properties in `StaticTuple` are valid.

    Validity right now is just ensuring the number of elements is > 0.

    Parameters:
      T: The StaticTuple's element type.
      size: The number of elements.
    """
    comptime assert (
        T.__move_ctor_is_trivial
        and T.__copy_ctor_is_trivial
        and T.__del__is_trivial
    ), String(
        (
            "`StaticTuple` element type must have a trivial move/copy"
            " constructor and destructor: "
        ),
        get_type_name[T](),
    )
    comptime assert (
        size >= 0
    ), "number of elements in `StaticTuple` must be >= 0"


struct StaticTuple[element_type: _StaticTupleTraits, size: Int](
    Defaultable, DevicePassable, Sized, TrivialRegisterPassable
):
    """A statically sized tuple type which contains elements of homogeneous types.

    Parameters:
        element_type: The type of the elements in the tuple.
        size: The size of the tuple.
    """

    comptime _mlir_type = __mlir_type[
        `!pop.array<`, Self.size._int_mlir_index(), `, `, Self.element_type, `>`
    ]
    comptime device_type: AnyType = Self
    """The device-side type for this `StaticTuple`."""

    var _mlir_value: Self._mlir_type
    """The underlying storage for the static tuple."""

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        """Get the human-readable type name for this `StaticTuple`.

        Returns:
            A string representation of the type, e.g. "StaticTuple[Int, 3]".
        """
        return String(
            "StaticTuple[",
            get_type_name[Self.element_type](),
            ", ",
            Self.size,
            "]",
        )

    @always_inline
    def __init__(out self):
        """Constructs an empty (undefined) tuple."""
        _static_tuple_construction_checks[Self.element_type, Self.size]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline
    def __init__(out self, *, mlir_value: Self._mlir_type):
        """Constructs from an array type.

        Args:
            mlir_value: Underlying MLIR array type.
        """
        _static_tuple_construction_checks[Self.element_type, Self.size]()
        self._mlir_value = mlir_value

    @always_inline
    def __init__(out self, *, fill: Self.element_type):
        """Constructs a static tuple given a fill value.

        Args:
            fill: The value to fill the tuple with.
        """
        _static_tuple_construction_checks[Self.element_type, Self.size]()
        self._mlir_value = __mlir_op.`pop.array.repeat`[
            _type=__mlir_type[
                `!pop.array<`,
                Self.size._int_mlir_index(),
                `, `,
                Self.element_type,
                `>`,
            ]
        ](fill)

    @always_inline
    def __init__(out self, *elems: Self.element_type):
        """Constructs a static tuple given a set of arguments.

        Args:
            elems: The element types.
        """
        _static_tuple_construction_checks[Self.element_type, Self.size]()
        if len(elems) == 1:
            return Self(fill=elems[0])

        assert Self.size == len(elems), "mismatch in the number of elements"
        self = Self()
        comptime for idx in range(Self.size):
            self[idx] = elems[idx]

    @always_inline
    def __init__[*values: Self.element_type](out self):
        """Creates a tuple constant using the specified values.

        Parameters:
            values: The list of values.
        """
        _static_tuple_construction_checks[Self.element_type, Self.size]()

        comptime num_values = values.size
        if num_values == 1:
            return Self(fill=values[0])

        comptime assert (
            Self.size == num_values
        ), "mismatch in the number of elements"
        self = Self()
        comptime for idx in range(Self.size):
            self[idx] = values[idx]

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Returns the length of the array. This is a known constant value.

        Returns:
            The size of the list.
        """
        return Self.size

    @always_inline("nodebug")
    def __getitem__[I: Indexer, //](self, idx: I) -> Self.element_type:
        """Returns the value of the tuple at the given dynamic index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            idx: The index into the tuple.

        Returns:
            The value at the specified position.
        """
        assert Self.size > index(idx), "index must be within bounds"
        return self._unsafe_ref(index(idx))

    @always_inline("nodebug")
    def __setitem__[I: Indexer, //](mut self, idx: I, val: Self.element_type):
        """Stores a single value into the tuple at the specified dynamic index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            idx: The index into the tuple.
            val: The value to store.
        """
        assert Self.size > index(idx), "index must be within bounds"
        self._unsafe_ref(index(idx)) = val

    @always_inline("nodebug")
    def get[index: Int](self) -> Self.element_type:
        """Returns the value of the tuple at the given index.

        Parameters:
            index: The index into the tuple.

        Returns:
            The value at the specified position.
        """
        comptime assert index < Self.size
        var val = __mlir_op.`pop.array.get`[
            _type=Self.element_type,
            index=index._int_mlir_index(),
        ](self._mlir_value)
        return val

    @always_inline("nodebug")
    def _unsafe_ref(ref self, idx: Int) -> ref[self] Self.element_type:
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer(to=self._mlir_value).address, idx._int_mlir_index()
        )
        return UnsafePointer[origin=origin_of(self)](ptr)[]

    @always_inline("nodebug")
    def _replace[idx: Int](self, val: Self.element_type) -> Self:
        """Replaces the value at the specified index.

        Parameters:
            idx: The index into the tuple.

        Args:
            val: The value to store.

        Returns:
            A new tuple with the specified element value replaced.
        """
        comptime assert idx < Self.size

        var array = __mlir_op.`pop.array.replace`[
            _type=__mlir_type[
                `!pop.array<`,
                Self.size._int_mlir_index(),
                `, `,
                Self.element_type,
                `>`,
            ],
            index=idx._int_mlir_index(),
        ](val, self._mlir_value)

        return Self(mlir_value=array)

    @always_inline
    def __eq__[
        _E: Equatable & TrivialRegisterPassable, //
    ](
        self: StaticTuple[_E, Self.size], other: StaticTuple[_E, Self.size]
    ) -> Bool:
        """Returns `True` if all elements are equal.

        Parameters:
            _E: The element type, must be `Equatable` and
                `TrivialRegisterPassable`.

        Args:
            other: The tuple to compare with.

        Returns:
            True if all corresponding elements are equal.
        """
        comptime for i in range(Self.size):
            if self[i] != other[i]:
                return False
        return True

    @always_inline
    def __ne__[
        _E: Equatable & TrivialRegisterPassable, //
    ](
        self: StaticTuple[_E, Self.size], other: StaticTuple[_E, Self.size]
    ) -> Bool:
        """Returns `True` if any element differs.

        Parameters:
            _E: The element type, must be `Equatable` and
                `TrivialRegisterPassable`.

        Args:
            other: The tuple to compare with.

        Returns:
            True if any corresponding elements differ.
        """
        return not (self == other)

    @always_inline
    def __lt__[
        _E: Comparable & TrivialRegisterPassable, //
    ](
        self: StaticTuple[_E, Self.size], other: StaticTuple[_E, Self.size]
    ) -> Bool:
        """Returns `True` if `self` is lexicographically less than `other`.

        Parameters:
            _E: The element type, must be `Comparable` and
                `TrivialRegisterPassable`.

        Args:
            other: The tuple to compare with.

        Returns:
            True if `self` is lexicographically less than `other`.
        """
        comptime for i in range(Self.size):
            if self[i] < other[i]:
                return True
            if self[i] != other[i]:
                return False
        return False

    @always_inline
    def __le__[
        _E: Comparable & TrivialRegisterPassable, //
    ](
        self: StaticTuple[_E, Self.size], other: StaticTuple[_E, Self.size]
    ) -> Bool:
        """Returns `True` if `self` is lexicographically less than or equal to
        `other`.

        Parameters:
            _E: The element type, must be `Comparable` and
                `TrivialRegisterPassable`.

        Args:
            other: The tuple to compare with.

        Returns:
            True if `self` is lexicographically less than or equal to `other`.
        """
        return not (other < self)

    @always_inline
    def __gt__[
        _E: Comparable & TrivialRegisterPassable, //
    ](
        self: StaticTuple[_E, Self.size], other: StaticTuple[_E, Self.size]
    ) -> Bool:
        """Returns `True` if `self` is lexicographically greater than `other`.

        Parameters:
            _E: The element type, must be `Comparable` and
                `TrivialRegisterPassable`.

        Args:
            other: The tuple to compare with.

        Returns:
            True if `self` is lexicographically greater than `other`.
        """
        return other < self

    @always_inline
    def __ge__[
        _E: Comparable & TrivialRegisterPassable, //
    ](
        self: StaticTuple[_E, Self.size], other: StaticTuple[_E, Self.size]
    ) -> Bool:
        """Returns `True` if `self` is lexicographically greater than or equal
        to `other`.

        Parameters:
            _E: The element type, must be `Comparable` and
                `TrivialRegisterPassable`.

        Args:
            other: The tuple to compare with.

        Returns:
            True if `self` is lexicographically greater than or equal to
            `other`.
        """
        return not (self < other)
