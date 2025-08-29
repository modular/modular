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
"""Implements StaticTuple, a statically-sized uniform container.

You can import these APIs from the `utils` package. For example:

```mojo
from utils import StaticTuple
```
"""


# ===-----------------------------------------------------------------------===#
# StaticTuple
# ===-----------------------------------------------------------------------===#


fn _static_tuple_construction_checks[size: Int]():
    """Checks if the properties in `StaticTuple` are valid.

    Validity right now is just ensuring the number of elements is > 0.

    Parameters:
      size: The number of elements.
    """
    constrained[size >= 0, "number of elements in `StaticTuple` must be >= 0"]()


@register_passable("trivial")
struct StaticTuple[element_type: AnyTrivialRegType, size: Int](
    Copyable, Defaultable, Movable, Sized
):
    """A statically sized tuple type which contains elements of homogeneous types.

    Parameters:
        element_type: The type of the elements in the tuple.
        size: The size of the tuple.
    """

    alias _mlir_type = __mlir_type[
        `!pop.array<`, size._mlir_value, `, `, Self.element_type, `>`
    ]

    var _mlir_value: Self._mlir_type
    """The underlying storage for the static tuple."""

    @always_inline
    fn __init__(out self):
        """Constructs an empty (undefined) tuple."""
        _static_tuple_construction_checks[size]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline
    fn __init__(out self, *, mlir_value: Self._mlir_type):
        """Constructs from an array type.

        Args:
            mlir_value: Underlying MLIR array type.
        """
        self._mlir_value = mlir_value

    @always_inline
    fn __init__(out self, *, fill: Self.element_type):
        """Constructs a static tuple given a fill value.

        Args:
            fill: The value to fill the tuple with.
        """
        _static_tuple_construction_checks[size]()
        self._mlir_value = __mlir_op.`pop.array.repeat`[
            _type = __mlir_type[
                `!pop.array<`, size._mlir_value, `, `, Self.element_type, `>`
            ]
        ](fill)

    @always_inline
    fn __init__(out self, *elems: Self.element_type):
        """Constructs a static tuple given a set of arguments.

        Args:
            elems: The element types.
        """
        _static_tuple_construction_checks[size]()
        self = Self(values=elems)

    @always_inline
    fn __init__(out self, values: VariadicList[Self.element_type]):
        """Creates a tuple constant using the specified values.

        Args:
            values: The list of values.
        """
        _static_tuple_construction_checks[size]()

        if len(values) == 1:
            return Self(fill=values[0])

        debug_assert(size == len(values), "mismatch in the number of elements")

        self = Self()

        @parameter
        for idx in range(size):
            self.__setitem__[idx](values[idx])

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Returns the length of the array. This is a known constant value.

        Returns:
            The size of the list.
        """
        return size

    @always_inline("nodebug")
    fn __getitem__[index: Int](self) -> Self.element_type:
        """Returns the value of the tuple at the given index.

        Parameters:
            index: The index into the tuple.

        Returns:
            The value at the specified position.
        """
        constrained[index < size]()
        var val = __mlir_op.`pop.array.get`[
            _type = Self.element_type,
            index = index._mlir_value,
        ](self._mlir_value)
        return val

    @always_inline("nodebug")
    fn __getitem__[I: Indexer, //](self, idx: I) -> Self.element_type:
        """Returns the value of the tuple at the given dynamic index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            idx: The index into the tuple.

        Returns:
            The value at the specified position.
        """
        debug_assert(size > index(idx), "index must be within bounds")
        return self._unsafe_ref(index(idx))

    @always_inline("nodebug")
    fn __setitem__[I: Indexer, //](mut self, idx: I, val: Self.element_type):
        """Stores a single value into the tuple at the specified dynamic index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            idx: The index into the tuple.
            val: The value to store.
        """
        debug_assert(size > index(idx), "index must be within bounds")
        self._unsafe_ref(index(idx)) = val

    @always_inline("nodebug")
    fn __setitem__[idx: Int](mut self, val: Self.element_type):
        """Stores a single value into the tuple at the specified index.

        Parameters:
            idx: The index into the tuple.

        Args:
            val: The value to store.
        """
        constrained[idx < size]()

        self._unsafe_ref(idx) = val

    @always_inline("nodebug")
    fn _unsafe_ref(ref self, idx: Int) -> ref [self] Self.element_type:
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer(to=self._mlir_value).address, idx._mlir_value
        )
        return UnsafePointer(ptr)[]

    @always_inline("nodebug")
    fn _replace[idx: Int](self, val: Self.element_type) -> Self:
        """Replaces the value at the specified index.

        Parameters:
            idx: The index into the tuple.

        Args:
            val: The value to store.

        Returns:
            A new tuple with the specified element value replaced.
        """
        constrained[idx < size]()

        var array = __mlir_op.`pop.array.replace`[
            _type = __mlir_type[
                `!pop.array<`, size._mlir_value, `, `, Self.element_type, `>`
            ],
            index = idx._mlir_value,
        ](val, self._mlir_value)

        return Self(mlir_value=array)
