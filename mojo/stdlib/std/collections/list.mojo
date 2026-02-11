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
"""Defines the List type.

These APIs are imported automatically, just like builtins.
"""


from builtin.constrained import _constrained_conforms_to
from builtin.rebind import downcast
import format._utils as fmt
from reflection import get_type_name
from collections._index_normalization import normalize_index
from collections._asan_annotations import (
    __sanitizer_annotate_contiguous_container,
)
from os import abort
from sys import size_of
from sys.intrinsics import _type_is_eq, _type_is_eq_parse_time

from memory import Pointer, memcpy
from builtin.builtin_slice import ContiguousSlice, StridedSlice
from .optional import Optional

# ===-----------------------------------------------------------------------===#
# List
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _ListIter[
    mut: Bool,
    //,
    T: Copyable,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator for List.

    Parameters:
        mut: Whether the reference to the list is mutable.
        T: The type of the elements in the list.
        origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    comptime Element = Self.T  # FIXME(MOCO-2068): shouldn't be needed.

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var index: Int
    var src: Pointer[List[Self.Element], Self.origin]

    @always_inline
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    fn __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        @parameter
        if Self.forward:
            if self.index >= len(self.src[]):
                raise StopIteration()
            self.index += 1
            return self.src[][self.index - 1]
        else:
            if self.index <= 0:
                raise StopIteration()
            self.index -= 1
            return self.src[][self.index]

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len: Int

        @parameter
        if Self.forward:
            iter_len = len(self.src[]) - self.index
        else:
            iter_len = self.index

        return (iter_len, {iter_len})


struct List[T: Copyable](
    Boolable,
    Copyable,
    Defaultable,
    Equatable,
    Iterable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    """A dynamically-allocated and resizable list.

    This is Mojo's primary dynamic array implementation, meaning the list can
    grow and shrink in size at runtime. However, all elements in a `List` must
    be the same type `T`, determined at compile time.

    You can create a `List` in several ways:

    ```mojo
    # Empty list
    var empty_list = List[Int]()

    # With pre-allocated capacity
    var preallocated = List[String](capacity=100)

    # With initial size and fill value
    var filled = List[Float64](length=10, fill=0.0)

    # With initial values and inferred type (Int)
    var numbers = [1, 2, 3, 4, 5]
    ```

    Be aware of the following characteristics:

    - **Type safety**: All elements must be the same type `T`, determined at
      compile time. This is more restrictive than Python's lists but it
      improves performance:

      ```mojo
      var int_list = [1, 2, 3]        # List[Int]
      var str_list = ["a", "b", "c"]  # List[String]
      var mixed = [1, "hello"]        # Error! All elements must be same type
      ```

      However, you can get around this by defining your list type as
      [`Variant`](/mojo/std/utils/variant/Variant). This is a discriminated
      union type, meaning it can store any number of different types that can
      vary at runtime.

    - **Value semantics:** A `List` is value semantic by default, so
      assignment creates a deep copy of all elements:

      ```mojo
      var list1 = [1, 2, 3]
      var list2 = list1.copy()        # Deep copy
      list2.append(4)
      print(list1.__str__())   # => [1, 2, 3]
      print(list2.__str__())   # => [1, 2, 3, 4]
      ```

      This is different from Python, where assignment creates a reference to
      the same list. For more information, read about [value
      semantics](/mojo/manual/values/value-semantics).

    - **Iteration uses immutable references**: When iterating a list, you get
      immutable references to the actual elements, unless you specify `ref`:

      ```mojo
      var numbers = [10, 20, 30]

      # Default behavior creates immutable (read-only) references
      for num in numbers:
          num += 1  # error: expression must be mutable

      # Using `ref` gets mutable (read-write) references
      for ref num in numbers:
          num += 1  # Modifies the original elements
      print(numbers.__str__())  # => [11, 21, 31]
      ```

    - **Out of bounds access**: Accessing elements with invalid indices will
      cause undefined behavior:

      ```mojo
      var my_list = [1, 2, 3]
      print(my_list[5])  # Undefined behavior (out of bounds)
      ```

      For safe access, you should manually check bounds or use methods that
      handle errors gracefully:

      ```mojo
      var my_list = [1, 2, 3]
      if 5 < len(my_list):
          print(my_list[5])  # Safe: check bounds first
      else:
          print("Index out of bounds")

      # Some methods like index() raise exceptions
      try:
          var idx = my_list.index(99)  # Raises ValueError if not found
          print("Found at index:", idx)
      except:
          print("Value not found in list")
      ```

    Examples:

    ```mojo
    var my_list = [10, 20, 30]

    # Add elements
    my_list.append(40)           # [10, 20, 30, 40]
    my_list.insert(1, 15)        # [10, 15, 20, 30, 40]
    my_list.extend([50, 60])     # [10, 15, 20, 30, 40, 50, 60]

    # Access elements
    print(my_list[0])            # 10 (first element)
    print(my_list[-1])           # 60 (last element)
    my_list[1] = 25              # Modify element: [10, 25, 20, 30, 40, 50, 60]

    # Remove elements
    print(my_list.pop())      # Removes and returns last element (60)
    print(my_list.pop(2))     # Removes element at index 2 (20)

    # List properties
    print('len:', len(my_list))          # Current number of elements
    print('cap:', my_list.capacity)      # Current allocated capacity

    # Multiply a list
    var repeated = [1, 2] * 3
    print(repeated.__str__())    # [1, 2, 1, 2, 1, 2]

    # Iterate over a list:
    var fruits = ["apple", "banana", "orange"]

    # Iterate by value (immutable references)
    for fruit in fruits:
        print(fruit)

    # Iterate backwards by value
    for fruit in reversed(fruits):
        print(fruit)

    # Iterate by index
    for i in range(len(fruits)):
        print(i, fruits[i])

    # Concatenate with + and +=
    fruits += ["mango"]
    var more_fruits = fruits + ["grape", "kiwi"]
    print(more_fruits.__str__())
    ```

    Parameters:
        T: The type of elements stored in the list.
    """

    # Fields
    var _data: UnsafePointer[Self.T, MutExternalOrigin]
    """The underlying storage for the list."""
    var _len: Int
    """The number of elements in the list."""
    var capacity: Int
    """The amount of elements that can fit in the list without resizing it."""

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _ListIter[Self.T, iterable_origin, True]
    """The iterator type for this list.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    # asan annotation methods
    fn _annotate_new(self):
        __sanitizer_annotate_contiguous_container(
            beg=self._data.bitcast[NoneType](),
            end=(self._data + self.capacity).bitcast[NoneType](),
            old_mid=(self._data + self.capacity).bitcast[NoneType](),
            new_mid=(self._data + self._len).bitcast[NoneType](),
        )

    fn _annotate_delete(self):
        __sanitizer_annotate_contiguous_container(
            beg=self._data.bitcast[NoneType](),
            end=(self._data + self.capacity).bitcast[NoneType](),
            old_mid=(self._data + self._len).bitcast[NoneType](),
            new_mid=(self._data + self.capacity).bitcast[NoneType](),
        )

    fn _annotate_increase(self, n: Int = 1):
        __sanitizer_annotate_contiguous_container(
            beg=self._data.bitcast[NoneType](),
            end=(self._data + self.capacity).bitcast[NoneType](),
            old_mid=(self._data + self._len).bitcast[NoneType](),
            new_mid=(self._data + self._len + n).bitcast[NoneType](),
        )

    fn _annotate_shrink(self, old_size: Int):
        __sanitizer_annotate_contiguous_container(
            beg=self._data.bitcast[NoneType](),
            end=(self._data + self.capacity).bitcast[NoneType](),
            old_mid=(self._data + old_size).bitcast[NoneType](),
            new_mid=(self._data + self._len).bitcast[NoneType](),
        )

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Constructs an empty list."""
        self._data = {}
        self._len = 0
        self.capacity = 0

    fn __init__(out self, *, capacity: Int):
        """Constructs a list with the given capacity.

        Args:
            capacity: The requested capacity of the list.
        """
        if capacity:
            self._data = alloc[Self.T](capacity)
        else:
            self._data = {}
        self._len = 0
        self.capacity = capacity
        self._annotate_new()

    fn __init__(out self, *, length: Int, fill: Self.T):
        """Constructs a list with the given capacity.

        Args:
            length: The requested length of the list.
            fill: The element to fill each element of the list.
        """
        self = Self()
        self._unchecked_grow(length, fill)

    @always_inline
    fn __init__(out self, var *values: Self.T, __list_literal__: ()):
        """Constructs a list from the given values.

        Args:
            values: The values to populate the list with.
            __list_literal__: Tell Mojo to use this method for list literals.
        """
        self = Self(elements=values^)

    fn __init__(out self, *, var elements: VariadicListMem[Self.T, _]):
        """Constructs a list from the given values.

        Args:
            elements: The values to populate the list with.
        """
        var length = len(elements)

        self = Self(capacity=length)

        self._annotate_increase(length)

        # Transfer all of the elements into the List.
        @parameter
        fn init_elt(idx: Int, var elt: Self.T):
            (self._data + idx).init_pointee_move(elt^)

        elements^.consume_elements[init_elt]()

        # Remember how many elements we have.
        self._len = length

    fn __init__(out self, span: Span[Self.T]):
        """Constructs a list from the a Span of values.

        Args:
            span: The span of values to populate the list with.
        """
        self = Self(capacity=len(span))
        for value in span:
            self.append(value.copy())

    fn __init__[
        IterableType: Iterable,
    ](
        ref iterable: IterableType,
        out self: List[
            downcast[
                IterableType.IteratorType[origin_of(iterable)].Element, Copyable
            ]
        ],
    ) where conforms_to(
        IterableType.IteratorType[origin_of(iterable)].Element, Copyable
    ):
        """Constructs a list from an iterable of values.

        Parameters:
            IterableType: The type of the `iterable` argument.

        Args:
            iterable: The iterable of values to populate the list with.
        """
        var lower, _ = iter(iterable).bounds()
        self = type_of(self)(capacity=lower)
        for var value in iterable:
            self.append(rebind_var[type_of(self).T](value^))

    @always_inline
    fn __init__(out self, *, unsafe_uninit_length: Int):
        """Construct a list with the specified length, with uninitialized
        memory. This is unsafe, as it relies on the caller initializing the
        elements with unsafe operations, not assigning over the uninitialized
        data.

        Args:
            unsafe_uninit_length: The number of elements to allocate.
        """
        self = Self(capacity=unsafe_uninit_length)
        self._annotate_increase(unsafe_uninit_length)
        self._len = unsafe_uninit_length

    fn __copyinit__(out self, existing: Self):
        """Creates a deep copy of the given list.

        Args:
            existing: The list to copy.
        """
        self = Self(capacity=existing.capacity)
        self.extend(Span(existing))

    fn __del__(deinit self):
        """Destroy all elements in the list and free its memory."""

        _constrained_conforms_to[
            conforms_to(Self.T, ImplicitlyDestructible),
            Parent=Self,
            Element = Self.T,
            ParentConformsTo="ImplicitlyDestructible",
        ]()
        comptime TDestructible = downcast[Self.T, ImplicitlyDestructible]

        @parameter
        if not TDestructible.__del__is_trivial:
            for i in range(len(self)):
                (self._data + i).bitcast[TDestructible]().destroy_pointee()
        self._annotate_delete()
        self._data.free()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Checks if two lists are equal.

        Constraints:
            `T` must conform to `Equatable`.

        Args:
            other: The list to compare with.

        Returns:
            True if the lists are equal, False otherwise.

        Examples:

        ```mojo
        var x = [1, 2, 3]
        var y = [1, 2, 3]
        print("x and y are equal" if x == y else "x and y are not equal")
        ```
        """
        _constrained_conforms_to[
            conforms_to(Self.T, Equatable),
            Parent=Self,
            Element = Self.T,
            ParentConformsTo="Equatable",
        ]()

        if len(self) != len(other):
            return False

        var index = 0
        for element in self:
            ref lhs = trait_downcast[Equatable](element)
            ref rhs = trait_downcast[Equatable](other[index])
            if lhs != rhs:
                return False
            index += 1
        return True

    @always_inline
    fn __ne__[
        U: Equatable & Copyable, //
    ](self: List[U, ...], other: List[U, ...]) -> Bool:
        """Checks if two lists are not equal.

        Parameters:
            U: The type of the elements in the list. Must implement the
               trait `Equatable`.

        Args:
            other: The list to compare with.

        Returns:
            True if the lists are not equal, False otherwise.

        Examples:

        ```mojo
        var x = [1, 2, 3]
        var y = [1, 2, 4]
        print("x and y are not equal" if x != y else "x and y are equal")
        ```
        """
        return not (self == other)

    fn __contains__[
        U: Equatable & Copyable, //
    ](self: List[U, ...], value: U) -> Bool:
        """Verify if a given value is present in the list.

        Parameters:
            U: The type of the elements in the list. Must implement the
              trait `Equatable`.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the list, False otherwise.

        Examples:

        ```mojo
        var x = [1, 2, 3]
        print("x contains 3" if 3 in x else "x does not contain 3")
        ```
        """
        for i in self:
            if i == value:
                return True
        return False

    fn __mul__[
        _T: Copyable & ImplicitlyDestructible, //
    ](self: List[_T], x: Int) -> List[_T]:
        """Multiplies the list by x and returns a new list.

        Parameters:
            _T: List element type that supports implicit destruction.

        Args:
            x: The multiplier number.

        Returns:
            The new list.
        """
        # avoid the copy since it would be cleared immediately anyways
        if x == 0:
            return List[_T]()
        var result = self.copy()
        result *= x
        return result^

    fn __imul__[
        _T: Copyable & ImplicitlyDestructible, //
    ](mut self: List[_T], x: Int):
        """Appends the original elements of this list x-1 times or clears it if
        x is <= 0.

        ```mojo
        var a = [1, 2]
        a *= 2 # a = [1, 2, 1, 2]
        ```

        Parameters:
            _T: List element type that supports implicit destruction.

        Args:
            x: The multiplier number.
        """
        if x <= 0 or len(self) == 0:
            self.clear()
            return
        var orig = self.copy()
        self.reserve(len(self) * x)
        for _ in range(x - 1):
            self.extend(Span(orig))

    fn __add__(self, var other: Self) -> Self:
        """Concatenates self with other and returns the result as a new list.

        Args:
            other: List whose elements will be combined with the elements of
                self.

        Returns:
            The newly created list.
        """
        var result = self.copy()
        result.extend(other^)
        return result^

    fn __iadd__(mut self, var other: Self):
        """Appends the elements of other into self.

        Args:
            other: List whose elements will be appended to self.
        """
        self.extend(other^)

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over elements of the list, returning immutable references.

        Returns:
            An iterator of immutable references to the list elements.
        """
        return {0, Pointer(to=self)}

    fn __reversed__(
        ref self,
    ) -> _ListIter[Self.T, origin_of(self), False]:
        """Iterate backwards over the list, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the list elements.
        """
        return _ListIter[forward=False](len(self), Pointer(to=self))

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Gets the number of elements in the list.

        Returns:
            The number of elements in the list.
        """
        return self._len

    fn __bool__(self) -> Bool:
        """Checks whether the list has any elements or not.

        Returns:
            `False` if the list is empty, `True` if there is at least one
            element.
        """
        return len(self) > 0

    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of a `List`.

        Returns:
            A string representation of the list.
        """
        # at least 1 byte per item e.g.: [a, b, c, d] = 4 + 2 * 3 + [] + null
        var l = len(self)
        var output = String(capacity=l + 2 * (l - 1) * Int(l > 1) + 3)
        self.write_to(output)
        return output^

    @no_inline
    fn __repr__(self) -> String:
        """Returns a string representation of a `List`.

        Returns:
            A string representation of the list.
        """
        var string = String()
        self.write_repr_to(string)
        return string^

    fn _write_self_to[
        f: fn(Self.T, mut Some[Writer])
    ](self, mut writer: Some[Writer]):
        fmt.constrained_conforms_to_writable[Self.T, Parent=Self]()

        var iterator = self.__iter__()

        @parameter
        fn iterate(mut w: Some[Writer]) raises StopIteration:
            f(iterator.__next__(), w)

        fmt.write_sequence_to[ElementFn=iterate](writer)
        _ = iterator^

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Write this list to a `Writer`.

        Constraints:
            `T` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """
        self._write_self_to[f = fmt.write_to[Self.T]](writer)

    @no_inline
    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write this list to a `Writer`.

        Constraints:
            `T` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """

        @parameter
        fn write_fields(mut w: Some[Writer]):
            self._write_self_to[f = fmt.write_repr_to[Self.T]](w)

        fmt.FormatStruct(writer, "List").params(
            fmt.TypeNames[Self.T](),
        ).fields[FieldsFn=write_fields]()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn byte_length(self) -> Int:
        """Gets the byte length of the List (`len(self) * size_of[T]()`).

        Returns:
            The byte length of the List (`len(self) * size_of[T]()`).
        """
        return len(self) * size_of[Self.T]()

    @no_inline
    fn _realloc(mut self, new_capacity: Int):
        var new_data = alloc[Self.T](new_capacity)

        @parameter
        if Self.T.__moveinit__is_trivial:
            memcpy(dest=new_data, src=self._data, count=len(self))
        else:
            for i in range(len(self)):
                (new_data + i).init_pointee_move_from(self._data + i)

        if self._data:
            self._annotate_delete()
            self._data.free()
        self._data = new_data
        self.capacity = new_capacity
        self._annotate_new()

    # FIXME: This annotation is needed to support List[Span[x, o]] types with
    # mutable origins.
    @__unsafe_disable_nested_origin_exclusivity
    fn append(mut self, var value: Self.T):
        """Appends a value to this list.

        Args:
            value: The value to append.

        Notes:
            If there is no capacity left, resizes to twice the current capacity.
            Except for 0 capacity where it sets 1.

        Examples:

        ```mojo
        list = [1, 2, 3, 4, 5]
        list.append(6)
        print(list) # [1, 2, 3, 4, 5, 6]
        ```
        """
        if self._len >= self.capacity:
            self._realloc(self.capacity * 2 | Int(self.capacity == 0))
        self._annotate_increase()
        self._unsafe_next_uninit_ptr().init_pointee_move(value^)
        self._len += 1

    fn insert(mut self, i: Int, var value: Self.T):
        """Inserts a value to the list at the given index.
        `a.insert(len(a), value)` is equivalent to `a.append(value)`.

        Args:
            i: The index for the value.
            value: The value to insert.

        Examples:

        ```mojo
        list = ["one", "three"]
        list.insert(1, "two")
        print(list) # ['one', 'two', 'three']
        ```
        """
        debug_assert(i <= len(self), "insert index out of range")

        var normalized_idx = i
        if i < 0:
            normalized_idx = max(len(self) + i, 0)

        var earlier_idx = len(self)
        var later_idx = len(self) - 1
        self.append(value^)

        for _ in range(normalized_idx, len(self) - 1):
            var earlier_ptr = self._data + earlier_idx
            var later_ptr = self._data + later_idx

            var tmp = earlier_ptr.take_pointee()
            earlier_ptr.init_pointee_move_from(later_ptr)
            later_ptr.init_pointee_move(tmp^)

            earlier_idx -= 1
            later_idx -= 1

    fn extend(mut self, var other: List[Self.T, ...]):
        """Extends this list by consuming the elements of `other`.

        Args:
            other: List whose elements will be added in order at the end of this
                list.

        Examples:

        ```mojo
        list = ["one", "two", "three"]
        more = ["four", "five"]
        list.extend(more^) # more's values are consumed
        # print(more)      # Error: use of initialized value
        print(list)        # ['one', 'two', 'three', 'four', 'five']
        ```
        """

        var other_len = len(other)
        var final_size = len(self) + other_len
        self.reserve(final_size)

        var dest_ptr = self._data + self._len
        var src_ptr = other.unsafe_ptr()
        self._annotate_increase(other_len)

        @parameter
        if Self.T.__moveinit__is_trivial:
            memcpy(dest=dest_ptr, src=src_ptr, count=other_len)
        else:
            for _ in range(other_len):
                dest_ptr.init_pointee_move_from(src_ptr)
                src_ptr += 1
                dest_ptr += 1

        # Update the size now since all elements have been moved into this list.
        self._len = final_size
        # The elements of `other` are now consumed, so we mark it as empty so
        # they don't get destroyed when it goes out of scope.
        other._len = 0

    fn extend(mut self, elements: Span[Self.T, _]):
        """Extend this list by copying elements from a `Span`.

        The resulting list will have the length `len(self) + len(elements)`.

        Args:
            elements: The elements to copy into this list.

        Examples:

        ```mojo
        numbers = [1, 2, 3]
        more = [4, 5, 6]
        numbers.extend(Span(more))
        print(numbers.__str__())   # [1, 2, 3, 4, 5, 6]
        ```
        """
        var elements_len = len(elements)
        var new_num_elts = self._len + elements_len
        if new_num_elts > self.capacity:
            # Make sure our capacity at least doubles to avoid O(n^2) behavior.
            self._realloc(max(self.capacity * 2, new_num_elts))

        self._annotate_increase(elements_len)
        var i = self._len
        self._len = new_num_elts

        @parameter
        if Self.T.__copyinit__is_trivial:
            memcpy(
                dest=self.unsafe_ptr() + i,
                src=elements.unsafe_ptr(),
                count=elements_len,
            )
        else:
            for elt in elements:
                UnsafePointer(to=self[i]).init_pointee_copy(elt)
                i += 1

    fn extend[
        dtype: DType, //
    ](mut self: List[Scalar[dtype], ...], value: SIMD[dtype, _]):
        """Extends this list with the elements of a vector.

        Parameters:
            dtype: The DType.

        Args:
            value: The value to append.

        Notes:
            If there is no capacity left, resizes to `len(self) + value.size`.

        Examples:

        ```mojo
        from collections import List

        numbers: List[Int64] = [1, 2]
        more = SIMD[DType.int64, 2](3, 4)
        numbers.extend(more)
        print(numbers) # [SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](2),
                       #  SIMD[DType.int64, 1](3), SIMD[DType.int64, 1](4)]
        ```
        """
        self.reserve(self._len + value.size)
        self._annotate_increase(value.size)
        self._unsafe_next_uninit_ptr().store(value)
        self._len += value.size

    fn extend[
        dtype: DType, //
    ](
        mut self: List[Scalar[dtype], ...],
        value: SIMD[dtype, _],
        *,
        count: Int,
    ):
        """Extends this list with `count` number of elements from a vector.

        Parameters:
            dtype: The DType.

        Args:
            value: The value to append.
            count: The amount of items to append. Must be less than or equal to
                `value.size`.

        Notes:
            If there is no capacity left, resizes to `len(self) + count`.

        Examples:

        ```mojo
        from collections import List

        numbers: List[Int64] = [1, 2]
        more = SIMD[DType.int64, 4](3, 4, 5, 6)
        numbers.extend(more, count=2)
        print(numbers) # [SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](2),
                       #  SIMD[DType.int64, 1](3), SIMD[DType.int64, 1](4)]
        ```
        """
        debug_assert(count <= value.size, "count must be <= value.size")
        self.reserve(self._len + count)
        self._annotate_increase(count)
        var v_ptr = UnsafePointer(to=value).bitcast[Scalar[dtype]]()
        memcpy(dest=self._unsafe_next_uninit_ptr(), src=v_ptr, count=count)
        self._len += count

    fn pop(mut self, i: Int = -1) -> Self.T:
        """Pops a value from the list at the given index.

        Args:
            i: The index of the value to pop.

        Returns:
            The popped value.

        Examples:

        ```mojo
        numbers = ["1", "2", "3", "4", "5"]
        value = numbers.pop(); print(value)   # 5
        print("length", len(numbers))         # length 4
        value = numbers.pop(2); print(value)  # 3
        print(numbers)                        # ['1', '2', '4']
        value = numbers.pop(-2); print(value) # 2, negative index
        ```
        """
        var normalized_idx = normalize_index["List", assert_always=False](
            i, UInt(len(self))
        )

        debug_assert(Int(normalized_idx) < self._len, "pop index out of range")

        var ret_val = (self._data + normalized_idx).take_pointee()
        for j in range(normalized_idx + 1, self._len):
            (self._data + j - 1).init_pointee_move_from(self._data + j)
        self._len -= 1
        self._annotate_shrink(self._len + 1)
        return ret_val^

    fn reserve(mut self, new_capacity: Int):
        """Reserves the requested capacity.

        Args:
            new_capacity: The new capacity.

        Notes:
            If the current capacity is greater or equal, this is a no-op.
            Otherwise, the storage is reallocated and the date is moved.
        """
        if self.capacity >= new_capacity:
            return
        self._realloc(new_capacity)

    fn resize[
        _T: Copyable & ImplicitlyDestructible, //
    ](mut self: List[_T], new_size: Int, value: _T):
        """Resizes the list to the given new size.

        Args:
            new_size: The new size.
            value: The value to use to populate new elements.

        Parameters:
            _T: List element type that supports implicit destruction.

        Notes:
            If the new size is smaller than the current one, elements at the end
            are discarded. If the new size is larger than the current one, the
            list is appended with new values elements up to the requested size.

        Examples:

        ```mojo
        list = ["z", "y", "x", "w"]
        list.resize(3, "v")
        print(list)                  # ['z', 'y', 'x']
        list.resize(6, "v")
        print(list)                  # ['z', 'y', 'x', 'v', 'v', 'v']
        ```
        """
        if new_size <= self._len:
            self.shrink(new_size)
        else:
            self._unchecked_grow(new_size, value)

    fn _unchecked_grow(mut self, new_size: Int, value: Self.T):
        debug_assert(new_size >= self._len)

        self.reserve(new_size)
        self._annotate_increase(new_size - self._len)
        for i in range(self._len, new_size):
            (self._data + i).init_pointee_copy(value)
        self._len = new_size

    fn resize[
        _T: Copyable & ImplicitlyDestructible, //
    ](mut self: List[_T], *, unsafe_uninit_length: Int):
        """Resizes the list to the given new size leaving any new elements
        uninitialized.

        If the new size is smaller than the current one, elements at the end
        are discarded. If the new size is larger than the current one, the
        list is extended and the new elements are left uninitialized.

        Parameters:
            _T: List element type that supports implicit destruction.

        Args:
            unsafe_uninit_length: The new size.

        Examples:

        ```mojo
        list = [1, 2, 3]
        list.resize(unsafe_uninit_length=5) # Indices 3 and 4 are uninitialized memory
        print(len(list))                    # 5
        list[3] = 10; list[4] = 20
        print(list)                         # [1, 2, 3, 10, 20]
        ```
        """
        if unsafe_uninit_length <= self._len:
            self.shrink(unsafe_uninit_length)
        else:
            self.reserve(unsafe_uninit_length)
            self._annotate_increase(unsafe_uninit_length - self._len)
            self._len = unsafe_uninit_length

    fn shrink[
        _T: Copyable & ImplicitlyDestructible, //
    ](mut self: List[_T], new_size: Int):
        """Resizes to the given new size which must be <= the current size.

        Parameters:
            _T: List element type that supports implicit destruction.

        Args:
            new_size: The new size.

        Notes:
            With no new value provided, the new size must be smaller than or
            equal to the current one. Elements at the end are discarded.

        Examples:

        ```mojo
        numbers = [1, 2, 3, 4, 5, 6]
        numbers.shrink(2); print(numbers) # [1, 2]
        # numbers.shrink(8)               # Error: new size is bigger than current
        ```
        """
        if len(self) < new_size:
            abort(
                "You are calling List.resize with a new_size bigger than the"
                " current size. If you want to make the List bigger, provide a"
                " value to fill the new slots with. If not, make sure the new"
                " size is smaller than the current size."
            )

        @parameter
        if not _T.__del__is_trivial:
            for i in range(new_size, len(self)):
                (self._data + i).destroy_pointee()
        var old_size: Int = self._len
        self._len = new_size
        self._annotate_shrink(old_size)
        self.reserve(new_size)

    fn reverse(mut self):
        """Reverses the elements of the list.

        Examples:

        ```mojo
        list = ["o", "l", "l", "e", "H"]
        list.reverse()
        print("".join(list)) # Hello
        ```
        """

        var earlier_idx = 0
        var later_idx = len(self) - 1

        var effective_len = len(self)
        var half_len = effective_len // 2

        for _ in range(half_len):
            var earlier_ptr = self._data + earlier_idx
            var later_ptr = self._data + later_idx

            var tmp = earlier_ptr.take_pointee()
            earlier_ptr.init_pointee_move_from(later_ptr)
            later_ptr.init_pointee_move(tmp^)

            earlier_idx += 1
            later_idx -= 1

    # TODO: Remove explicit self type when issue 1876 is resolved.
    fn index[
        C: Equatable & Copyable, //
    ](
        ref self: List[C, ...],
        value: C,
        start: Int = 0,
        stop: Optional[Int] = None,
    ) raises -> Int:
        """Returns the index of the first occurrence of a value in a list
        restricted by the range given the start and stop bounds.

        Args:
            value: The value to search for.
            start: The starting index of the search, treated as a slice index
                (defaults to 0).
            stop: The ending index of the search, treated as a slice index
                (defaults to None, which means the end of the list).

        Parameters:
            C: The type of the elements in the list. Must implement the
                `Equatable` trait.

        Returns:
            The index of the first occurrence of the value in the list.

        Raises:
            ValueError: If the value is not found in the list.

        Examples:

        ```mojo
        var my_list = [1, 2, 3]
        print(my_list.index(2)) # prints `1`
        ```
        """
        var start_normalized = start

        var stop_normalized: Int
        if stop is None:
            # Default end
            stop_normalized = len(self)
        else:
            stop_normalized = stop.value()

        if start_normalized < 0:
            start_normalized += len(self)
        if stop_normalized < 0:
            stop_normalized += len(self)

        start_normalized = _clip(start_normalized, 0, len(self))
        stop_normalized = _clip(stop_normalized, 0, len(self))

        for i in range(start_normalized, stop_normalized):
            if self[i] == value:
                return i
        raise "ValueError: Given element is not in list"

    fn clear[_T: Copyable & ImplicitlyDestructible, //](mut self: List[_T]):
        """Clears the elements in the list.

        Parameters:
            _T: List element type that supports implicit destruction.

        Examples:

        ```mojo
        list = ["o", "l", "l", "e", "H"]
        print(len(list))  # 5
        list.clear()
        print(len(list))  # 0
        ```
        """
        for i in range(self._len):
            (self._data + i).destroy_pointee()
        var old_size: Int = self._len
        self._len = 0
        self._annotate_shrink(old_size)

    fn steal_data(mut self) -> UnsafePointer[Self.T, MutExternalOrigin]:
        """Take ownership of the underlying pointer from the list.

        Returns:
            The underlying data.

        Examples:

        ```mojo
        from collections import List

        list: List[Int64] = [1, 2, 3, 4]
        ptr = list.steal_data() # list is no longer available
        for idx in range(4):
            print(ptr[idx], end=" ")
        print() # Output: 1 2 3 4
        # Free the pointer data
        ```
        """
        self._annotate_delete()
        var ptr = self._data
        self._data = {}
        self._len = 0
        self.capacity = 0
        return ptr

    fn __getitem__(self, slice: StridedSlice) -> Self:
        """Gets the sequence of elements at the specified positions.

        Args:
            slice: A slice that specifies positions of the new list.

        Returns:
            A new list containing the list at the specified slice.
        """
        var start, end, step = slice.indices(len(self))
        var r = range(start, end, step)

        if not len(r):
            return Self()

        var res = Self(capacity=len(r))
        for i in r:
            res.append(self[i].copy())

        return res^

    fn __getitem__[
        origin: Origin, //
    ](ref[origin] self, slice: ContiguousSlice) -> Span[Self.T, origin]:
        """Gets the sequence of elements at the specified positions.

        Parameters:
            origin: The origin of `List`.

        Args:
            slice: A slice the specifies the positions of the new list.

        Returns:
            A span over the specified slice.
        """
        var start, end = slice.indices(len(self))
        return Span[Self.T, origin](
            ptr=self.unsafe_ptr() + start, length=end - start
        )

    fn __getitem__[I: Indexer, //](ref self, idx: I) -> ref[self] Self.T:
        """Gets the list element at the given index.

        Args:
            idx: The index of the element.

        Parameters:
            I: A type that can be used as an index.

        Returns:
            A reference to the element at the given index.
        """

        var normalized_idx = normalize_index["List", assert_always=False](
            idx, UInt(len(self))
        )
        return (self._data + normalized_idx)[]

    @always_inline
    fn unsafe_get(ref self, idx: Int) -> ref[self] Self.T:
        """Get a reference to an element of self without checking index bounds.

        Args:
            idx: The index of the element to get.

        Returns:
            A reference to the element at the given index.

        Notes:
            Users should consider using `__getitem__` instead of this method as
            it is unsafe. If an index is out of bounds, this method will not
            abort, it will be considered undefined behavior.

            Note that there is no wraparound for negative indices, caution is
            advised. Using negative indices is considered undefined behavior.
            Never use `my_list.unsafe_get(-1)` to get the last element of the
            list. Instead, do `my_list.unsafe_get(len(my_list) - 1)`.
        """
        debug_assert(
            0 <= idx < len(self),
            (
                "The index provided must be within the range [0, len(List) -1]"
                " when using List.unsafe_get()"
            ),
        )
        return (self._data + idx)[]

    @always_inline
    fn unsafe_set[
        _T: Copyable & ImplicitlyDestructible
    ](mut self: List[_T], idx: Int, var value: _T):
        """Write a value to a given location without checking index bounds.

        Parameters:
            _T: List element type that supports implicit destruction.

        Args:
            idx: The index of the element to set.
            value: The value to set.

        Notes:
            Users should consider using `my_list[idx] = value` instead of this
            method as it is unsafe. If an index is out of bounds, this method
            will not abort, it will be considered undefined behavior.

            Note that there is no wraparound for negative indices, caution is
            advised. Using negative indices is considered undefined behavior.
            Never use `my_list.unsafe_set(-1, value)` to set the last element of
            the list. Instead, do `my_list.unsafe_set(len(my_list) - 1, value)`.
        """
        debug_assert(
            0 <= idx < len(self),
            (
                "The index provided must be within the range [0, len(List) -1]"
                " when using List.unsafe_set()"
            ),
        )
        (self._data + idx).destroy_pointee()
        (self._data + idx).init_pointee_move(value^)

    fn count[
        _T: Equatable & Copyable, //
    ](self: List[_T, ...], value: _T) -> Int:
        """Counts the number of occurrences of a value in the list.

        Parameters:
            _T: The type of the elements in the list. Must implement the
                trait `Equatable`.

        Args:
            value: The value to count.

        Returns:
            The number of occurrences of the value in the list.

        Examples:

        ```mojo
        list = ["a", "b", "c", "b", "b", "a", "c"]
        print(list.count("b")) # 3
        ```
        """
        var count = 0
        for elem in self:
            if elem == value:
                count += 1
        return count

    fn swap_elements(mut self, elt_idx_1: Int, elt_idx_2: Int):
        """Swaps elements at the specified indexes if they are different.

        Args:
            elt_idx_1: The index of one element.
            elt_idx_2: The index of the other element.

        Examples:

        ```mojo
        var my_list = [1, 2, 3]
        my_list.swap_elements(0, 2)
        print(my_list.__str__()) # 3, 2, 1
        ```

        Notes:
            This is useful because `swap(my_list[i], my_list[j])` cannot be
            supported by Mojo, because a mutable alias may be formed.
        """
        debug_assert(
            0 <= elt_idx_1 < len(self) and 0 <= elt_idx_2 < len(self),
            (
                "The indices provided to swap_elements must be within the range"
                " [0, len(List)-1]"
            ),
        )
        var ptr = self._data
        (ptr + elt_idx_1).swap_pointees(ptr + elt_idx_2)

    fn unsafe_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> UnsafePointer[
        Self.T, origin, address_space=address_space
    ]:
        """Retrieves a pointer to the underlying memory.

        Parameters:
            origin: The origin of the `List`.
            address_space: The `AddressSpace` of the `List`.

        Returns:
            The pointer to the underlying memory.
        """
        return (
            self._data.unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
            .address_space_cast[address_space]()
        )

    @always_inline
    fn _unsafe_next_uninit_ptr(
        ref self,
    ) -> UnsafePointer[Self.T, origin_of(self)]:
        """Retrieves a pointer to the next uninitialized element position.

        Safety:

        - This pointer MUST not be used to read or write memory beyond the
        allocated capacity of this list.
        - This pointer may not be used to initialize non-contiguous elements.
        - Ensure that `List._len` is updated to reflect the new number of
          initialized elements, otherwise elements may be unexpectedly
          overwritten or not destroyed correctly.

        Notes:
            This returns a pointer that points to the element position immediately
            after the last initialized element. This is equivalent to
            `list.unsafe_ptr() + len(list)`.
        """
        debug_assert(
            self.capacity > 0 and self.capacity > self._len,
            (
                "safety violation: Insufficient capacity to retrieve pointer to"
                " next uninitialized element"
            ),
        )

        # self.unsafe_ptr() + self._len won't work because .unsafe_ptr()
        # takes a ref that might mutate self
        var length = self._len
        return self.unsafe_ptr() + length


fn _clip(value: Int, start: Int, end: Int) -> Int:
    return max(start, min(value, end))
