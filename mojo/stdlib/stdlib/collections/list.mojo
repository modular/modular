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
"""Defines the List type.

These APIs are imported automatically, just like builtins.
"""


from os import abort
from sys import sizeof
from sys.intrinsics import _type_is_eq

from memory import Pointer, Span, UnsafePointer, memcpy

from .optional import Optional

# ===-----------------------------------------------------------------------===#
# List
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _ListIter[
    list_mutability: Bool, //,
    T: Copyable & Movable,
    hint_trivial_type: Bool,
    list_origin: Origin[list_mutability],
    forward: Bool = True,
](Copyable, Movable):
    """Iterator for List.

    Parameters:
        list_mutability: Whether the reference to the list is mutable.
        T: The type of the elements in the list.
        hint_trivial_type: Set to `True` if the type `T` is trivial, this is not
            mandatory, but it helps performance. Will go away in the future.
        list_origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    alias list_type = List[T, hint_trivial_type]

    var index: Int
    var src: Pointer[Self.list_type, list_origin]

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> ref [list_origin] T:
        @parameter
        if forward:
            self.index += 1
            return self.src[][self.index - 1]
        else:
            self.index -= 1
            return self.src[][self.index]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src[]) - self.index
        else:
            return self.index


struct List[T: Copyable & Movable, hint_trivial_type: Bool = False](
    Boolable,
    Copyable,
    Movable,
    ExplicitlyCopyable,
    Sized,
):
    """The `List` type is a dynamically-allocated list.

    Parameters:
        T: The type of the elements.
        hint_trivial_type: A hint to the compiler that the type T is trivial.
            It's not mandatory, but if set, it allows some optimizations.

    Notes:
        It supports pushing and popping from the back resizing the underlying
        storage as needed.  When it is deallocated, it frees its memory.
    """

    # Fields
    var data: UnsafePointer[T]
    """The underlying storage for the list."""
    var _len: Int
    """The number of elements in the list."""
    var capacity: Int
    """The amount of elements that can fit in the list without resizing it."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Constructs an empty list."""
        self.data = UnsafePointer[T]()
        self._len = 0
        self.capacity = 0

    fn copy(self) -> Self:
        """Creates a deep copy of the given list.

        Returns:
            A copy of the value.
        """
        var copy = Self(capacity=self.capacity)
        for ref e in self:
            copy.append(e)
        return copy^

    fn __init__(out self, *, capacity: Int):
        """Constructs a list with the given capacity.

        Args:
            capacity: The requested capacity of the list.
        """
        if capacity:
            self.data = UnsafePointer[T].alloc(capacity)
        else:
            self.data = UnsafePointer[T]()
        self._len = 0
        self.capacity = capacity

    fn __init__(out self, *, length: UInt, fill: T):
        """Constructs a list with the given capacity.

        Args:
            length: The requested length of the list.
            fill: The element to fill each element of the list.
        """
        self = Self()
        self.resize(length, fill)

    @always_inline
    fn __init__(out self, owned *values: T, __list_literal__: () = ()):
        """Constructs a list from the given values.

        Args:
            values: The values to populate the list with.
            __list_literal__: Tell Mojo to use this method for list literals.
        """
        self = Self(elements=values^)

    fn __init__(out self, *, owned elements: VariadicListMem[T, _]):
        """Constructs a list from the given values.

        Args:
            elements: The values to populate the list with.
        """
        var length = len(elements)

        self = Self(capacity=length)

        for i in range(length):
            var src = UnsafePointer(to=elements[i])
            var dest = self.data + i

            src.move_pointee_into(dest)

        # Do not destroy the elements when their backing storage goes away.
        __disable_del elements

        self._len = length

    fn __init__(out self, span: Span[T]):
        """Constructs a list from the a Span of values.

        Args:
            span: The span of values to populate the list with.
        """
        self = Self(capacity=len(span))
        for ref value in span:
            self.append(value)

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
        self._len = unsafe_uninit_length

    fn __copyinit__(out self, existing: Self):
        """Creates a deepcopy of the given list.

        Args:
            existing: The list to copy.
        """
        self = Self(capacity=existing.capacity)
        for i in range(len(existing)):
            self.append(existing[i])

    fn __del__(owned self):
        """Destroy all elements in the list and free its memory."""

        @parameter
        if not hint_trivial_type:
            for i in range(len(self)):
                (self.data + i).destroy_pointee()
        self.data.free()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __eq__[
        U: EqualityComparable & Copyable & Movable, //
    ](self: List[U, *_], other: List[U, *_]) -> Bool:
        """Checks if two lists are equal.

        Parameters:
            U: The type of the elements in the list. Must implement the
               trait `EqualityComparable`.

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
        if len(self) != len(other):
            return False
        var index = 0
        for ref element in self:
            if element != other[index]:
                return False
            index += 1
        return True

    @always_inline
    fn __ne__[
        U: EqualityComparable & Copyable & Movable, //
    ](self: List[U, *_], other: List[U, *_]) -> Bool:
        """Checks if two lists are not equal.

        Parameters:
            U: The type of the elements in the list. Must implement the
               trait `EqualityComparable`.

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
        U: EqualityComparable & Copyable & Movable, //
    ](self: List[U, *_], value: U) -> Bool:
        """Verify if a given value is present in the list.

        Parameters:
            U: The type of the elements in the list. Must implement the
              trait `EqualityComparable`.

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
        for ref i in self:
            if i == value:
                return True
        return False

    fn __mul__(self, x: Int) -> Self:
        """Multiplies the list by x and returns a new list.

        Args:
            x: The multiplier number.

        Returns:
            The new list.
        """
        # avoid the copy since it would be cleared immediately anyways
        if x == 0:
            return Self()
        var result = self.copy()
        result *= x
        return result^

    fn __imul__(mut self, x: Int):
        """Appends the original elements of this list x-1 times or clears it if
        x is <= 0.

        ```mojo
        var a = [1, 2]
        a *= 2 # a = [1, 2, 1, 2]
        ```

        Args:
            x: The multiplier number.
        """
        if x <= 0 or len(self) == 0:
            self.clear()
            return
        var orig = self.copy()
        self.reserve(len(self) * x)
        for _ in range(x - 1):
            self.extend(orig)

    fn __add__(self, owned other: Self) -> Self:
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

    fn __iadd__(mut self, owned other: Self):
        """Appends the elements of other into self.

        Args:
            other: List whose elements will be appended to self.
        """
        self.extend(other^)

    fn __iter__(ref self) -> _ListIter[T, hint_trivial_type, __origin_of(self)]:
        """Iterate over elements of the list, returning immutable references.

        Returns:
            An iterator of immutable references to the list elements.
        """
        return _ListIter(0, Pointer(to=self))

    fn __reversed__(
        ref self,
    ) -> _ListIter[T, hint_trivial_type, __origin_of(self), False]:
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
    fn __str__[
        U: Representable & Copyable & Movable, //
    ](self: List[U, *_]) -> String:
        """Returns a string representation of a `List`.

        Parameters:
            U: The type of the elements in the list. Must implement the
              trait `Representable`.

        Returns:
            A string representation of the list.

        Notes:
            Note that since we can't condition methods on a trait yet,
            the way to call this method is a bit special. Here is an example
            below:

            ```mojo
            var my_list = [1, 2, 3]
            print(my_list.__str__())
            ```

            When the compiler supports conditional methods, then a simple
            `String(my_list)` will be enough.
        """
        # at least 1 byte per item e.g.: [a, b, c, d] = 4 + 2 * 3 + [] + null
        var l = len(self)
        var output = String(capacity=l + 2 * (l - 1) * Int(l > 1) + 3)
        self.write_to(output)
        return output^

    @no_inline
    fn write_to[
        W: Writer, U: Representable & Copyable & Movable, //
    ](self: List[U, *_], mut writer: W):
        """Write `my_list.__str__()` to a `Writer`.

        Parameters:
            W: A type conforming to the Writable trait.
            U: The type of the List elements. Must have the trait
                `Representable`.

        Args:
            writer: The object to write to.
        """
        writer.write("[")
        for i in range(len(self)):
            writer.write(repr(self[i]))
            if i < len(self) - 1:
                writer.write(", ")
        writer.write("]")

    @no_inline
    fn __repr__[
        U: Representable & Copyable & Movable, //
    ](self: List[U, *_]) -> String:
        """Returns a string representation of a `List`.

        Parameters:
            U: The type of the elements in the list. Must implement the
              trait `Representable`.

        Returns:
            A string representation of the list.

        Notes:
            Note that since we can't condition methods on a trait yet, the way
            to call this method is a bit special. Here is an example below:

            ```mojo
            var my_list = [1, 2, 3]
            print(my_list.__repr__())
            ```

            When the compiler supports conditional methods, then a simple
            `repr(my_list)` will be enough.
        """
        return self.__str__()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn byte_length(self) -> Int:
        """Gets the byte length of the List (`len(self) * sizeof[T]()`).

        Returns:
            The byte length of the List (`len(self) * sizeof[T]()`).
        """
        return len(self) * sizeof[T]()

    @no_inline
    fn _realloc(mut self, new_capacity: Int):
        var new_data = UnsafePointer[T].alloc(new_capacity)

        @parameter
        if hint_trivial_type:
            memcpy(new_data, self.data, len(self))
        else:
            for i in range(len(self)):
                (self.data + i).move_pointee_into(new_data + i)

        if self.data:
            self.data.free()
        self.data = new_data
        self.capacity = new_capacity

    fn append(mut self, owned value: T):
        """Appends a value to this list.

        Args:
            value: The value to append.

        Notes:
            If there is no capacity left, resizes to twice the current capacity.
            Except for 0 capacity where it sets 1.
        """
        if self._len >= self.capacity:
            self._realloc(self.capacity * 2 | Int(self.capacity == 0))
        self._unsafe_next_uninit_ptr().init_pointee_move(value^)
        self._len += 1

    fn append(mut self, elements: Span[T, _]):
        """Appends elements to this list.

        Args:
            elements: The elements to append.
        """
        var elements_len = len(elements)
        var new_num_elts = self._len + elements_len
        if new_num_elts > self.capacity:
            # Make sure our capacity at least doubles to avoid O(n^2) behavior.
            self._realloc(max(self.capacity * 2, new_num_elts))

        var i = self._len
        self._len = new_num_elts

        @parameter
        if hint_trivial_type:
            memcpy(self.data + i, elements.unsafe_ptr(), elements_len)
        else:
            for ref elt in elements:
                UnsafePointer(to=self[i]).init_pointee_copy(elt)
                i += 1

    fn insert(mut self, i: Int, owned value: T):
        """Inserts a value to the list at the given index.
        `a.insert(len(a), value)` is equivalent to `a.append(value)`.

        Args:
            i: The index for the value.
            value: The value to insert.
        """
        debug_assert(i <= len(self), "insert index out of range")

        var normalized_idx = i
        if i < 0:
            normalized_idx = max(0, len(self) + i)

        var earlier_idx = len(self)
        var later_idx = len(self) - 1
        self.append(value^)

        for _ in range(normalized_idx, len(self) - 1):
            var earlier_ptr = self.data + earlier_idx
            var later_ptr = self.data + later_idx

            var tmp = earlier_ptr.take_pointee()
            later_ptr.move_pointee_into(earlier_ptr)
            later_ptr.init_pointee_move(tmp^)

            earlier_idx -= 1
            later_idx -= 1

    fn extend(mut self, owned other: List[T, *_]):
        """Extends this list by consuming the elements of `other`.

        Args:
            other: List whose elements will be added in order at the end of this
                list.
        """

        var other_len = len(other)
        var final_size = len(self) + other_len
        self.reserve(final_size)

        var dest_ptr = self.data + self._len
        var src_ptr = other.unsafe_ptr()

        @parameter
        if hint_trivial_type:
            memcpy(dest_ptr, src_ptr, other_len)
        else:
            for _ in range(other_len):
                # This (TODO: optimistically) moves an element directly from the
                # `other` list into this list using a single `T.__moveinit()__`
                # call, without moving into an intermediate temporary value
                # (avoiding an extra redundant move constructor call).
                src_ptr.move_pointee_into(dest_ptr)
                src_ptr += 1
                dest_ptr += 1

        # Update the size now since all elements have been moved into this list.
        self._len = final_size
        # The elements of `other` are now consumed, so we mark it as empty so
        # they don't get destroyed when it goes out of scope.
        other._len = 0

    fn extend[
        D: DType, //
    ](mut self: List[Scalar[D], *_, **_], value: SIMD[D, _]):
        """Extends this list with the elements of a vector.

        Parameters:
            D: The DType.

        Args:
            value: The value to append.

        Notes:
            If there is no capacity left, resizes to `len(self) + value.size`.
        """
        self.reserve(self._len + value.size)
        self._unsafe_next_uninit_ptr().store(value)
        self._len += value.size

    fn extend[
        D: DType, //
    ](mut self: List[Scalar[D], *_, **_], value: SIMD[D, _], *, count: Int):
        """Extends this list with `count` number of elements from a vector.

        Parameters:
            D: The DType.

        Args:
            value: The value to append.
            count: The ammount of items to append. Must be less than or equal to
                `value.size`.

        Notes:
            If there is no capacity left, resizes to `len(self) + count`.
        """
        debug_assert(count <= value.size, "count must be <= value.size")
        self.reserve(self._len + count)
        var v_ptr = UnsafePointer(to=value).bitcast[Scalar[D]]()
        memcpy(self._unsafe_next_uninit_ptr(), v_ptr, count)
        self._len += count

    fn extend[
        D: DType, //
    ](mut self: List[Scalar[D], *_, **_], value: Span[Scalar[D]]):
        """Extends this list with the elements of a `Span`.

        Parameters:
            D: The DType.

        Args:
            value: The value to append.

        Notes:
            If there is no capacity left, resizes to `len(self) + len(value)`.
        """
        self.reserve(self._len + len(value))
        memcpy(self._unsafe_next_uninit_ptr(), value.unsafe_ptr(), len(value))
        self._len += len(value)

    fn pop(mut self, i: Int = -1) -> T:
        """Pops a value from the list at the given index.

        Args:
            i: The index of the value to pop.

        Returns:
            The popped value.
        """
        debug_assert(-self._len <= i < self._len, "pop index out of range")

        var normalized_idx = i
        if i < 0:
            normalized_idx += self._len

        var ret_val = (self.data + normalized_idx).take_pointee()
        for j in range(normalized_idx + 1, self._len):
            (self.data + j).move_pointee_into(self.data + j - 1)
        self._len -= 1

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

    fn resize(mut self, new_size: Int, value: T):
        """Resizes the list to the given new size.

        Args:
            new_size: The new size.
            value: The value to use to populate new elements.

        Notes:
            If the new size is smaller than the current one, elements at the end
            are discarded. If the new size is larger than the current one, the
            list is appended with new values elements up to the requested size.
        """
        if new_size <= self._len:
            self.shrink(new_size)
        else:
            self.reserve(new_size)
            for i in range(self._len, new_size):
                (self.data + i).init_pointee_copy(value)
            self._len = new_size

    fn resize(mut self, *, unsafe_uninit_length: Int):
        """Resizes the list to the given new size leaving any new elements
        uninitialized.

        If the new size is smaller than the current one, elements at the end
        are discarded. If the new size is larger than the current one, the
        list is extended and the new elements are left uninitialized.

        Args:
            unsafe_uninit_length: The new size.
        """
        if unsafe_uninit_length <= self._len:
            self.shrink(unsafe_uninit_length)
        else:
            self.reserve(unsafe_uninit_length)
            self._len = unsafe_uninit_length

    fn shrink(mut self, new_size: Int):
        """Resizes to the given new size which must be <= the current size.

        Args:
            new_size: The new size.

        Notes:
            With no new value provided, the new size must be smaller than or
            equal to the current one. Elements at the end are discarded.
        """
        if len(self) < new_size:
            abort(
                "You are calling List.resize with a new_size bigger than the"
                " current size. If you want to make the List bigger, provide a"
                " value to fill the new slots with. If not, make sure the new"
                " size is smaller than the current size."
            )

        @parameter
        if not hint_trivial_type:
            for i in range(new_size, len(self)):
                (self.data + i).destroy_pointee()
        self._len = new_size
        self.reserve(new_size)

    fn reverse(mut self):
        """Reverses the elements of the list."""

        var earlier_idx = 0
        var later_idx = len(self) - 1

        var effective_len = len(self)
        var half_len = effective_len // 2

        for _ in range(half_len):
            var earlier_ptr = self.data + earlier_idx
            var later_ptr = self.data + later_idx

            var tmp = earlier_ptr.take_pointee()
            later_ptr.move_pointee_into(earlier_ptr)
            later_ptr.init_pointee_move(tmp^)

            earlier_idx += 1
            later_idx -= 1

    # TODO: Remove explicit self type when issue 1876 is resolved.
    fn index[
        C: EqualityComparable & Copyable & Movable, //
    ](
        ref self: List[C, *_],
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
                `EqualityComparable` trait.

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

    fn _binary_search_index[
        dtype: DType, //,
    ](self: List[Scalar[dtype], **_], needle: Scalar[dtype]) -> Optional[UInt]:
        """Finds the index of `needle` with binary search.

        Args:
            needle: The value to binary search for.

        Returns:
            Returns None if `needle` is not present, or if `self` was not
            sorted.

        Notes:
            This function will return an unspecified index if `self` is not
            sorted in ascending order.
        """
        var cursor = UInt(0)
        var b = self.data
        var length = len(self)
        while length > 1:
            var half = length >> 1
            length -= half
            cursor += Int(b[cursor + half - 1] < needle) * half

        return Optional(cursor) if b[cursor] == needle else None

    fn clear(mut self):
        """Clears the elements in the list."""
        for i in range(self._len):
            (self.data + i).destroy_pointee()
        self._len = 0

    fn steal_data(mut self) -> UnsafePointer[T]:
        """Take ownership of the underlying pointer from the list.

        Returns:
            The underlying data.
        """
        var ptr = self.data
        self.data = UnsafePointer[T]()
        self._len = 0
        self.capacity = 0
        return ptr

    fn __getitem__(self, slice: Slice) -> Self:
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
            res.append(self[i])

        return res^

    fn __getitem__[I: Indexer](ref self, idx: I) -> ref [self] T:
        """Gets the list element at the given index.

        Args:
            idx: The index of the element.

        Parameters:
            I: A type that can be used as an index.

        Returns:
            A reference to the element at the given index.
        """

        @parameter
        if _type_is_eq[I, UInt]():
            return (self.data + idx)[]
        else:
            var normalized_idx = Int(idx)
            debug_assert(
                -self._len <= normalized_idx < self._len,
                "index: ",
                normalized_idx,
                " is out of bounds for `List` of length: ",
                self._len,
            )
            if normalized_idx < 0:
                normalized_idx += len(self)

            return (self.data + normalized_idx)[]

    @always_inline
    fn unsafe_get(ref self, idx: Int) -> ref [self] Self.T:
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
        return (self.data + idx)[]

    @always_inline
    fn unsafe_set(mut self, idx: Int, owned value: T):
        """Write a value to a given location without checking index bounds.

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
        (self.data + idx).destroy_pointee()
        (self.data + idx).init_pointee_move(value^)

    fn count[
        T: EqualityComparable & Copyable & Movable, //
    ](self: List[T, *_], value: T) -> Int:
        """Counts the number of occurrences of a value in the list.

        Parameters:
            T: The type of the elements in the list. Must implement the
                trait `EqualityComparable`.

        Args:
            value: The value to count.

        Returns:
            The number of occurrences of the value in the list.
        """
        var count = 0
        for ref elem in self:
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
        if elt_idx_1 != elt_idx_2:
            swap((self.data + elt_idx_1)[], (self.data + elt_idx_2)[])

    fn unsafe_ptr(
        ref self,
    ) -> UnsafePointer[
        T,
        mut = Origin(__origin_of(self)).mut,
        origin = __origin_of(self),
    ]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The pointer to the underlying memory.
        """
        return self.data.origin_cast[
            mut = Origin(__origin_of(self)).mut, origin = __origin_of(self)
        ]()

    @always_inline
    fn _unsafe_next_uninit_ptr(
        ref self,
    ) -> UnsafePointer[
        T,
        mut = Origin(__origin_of(self)).mut,
        origin = __origin_of(self),
    ]:
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

    fn _cast_hint_trivial_type[
        hint_trivial_type: Bool
    ](owned self) -> List[T, hint_trivial_type]:
        var result = List[T, hint_trivial_type]()
        result.data = self.data
        result._len = self._len
        result.capacity = self.capacity

        # We stole the elements, don't destroy them.
        __disable_del self

        return result^


fn _clip(value: Int, start: Int, end: Int) -> Int:
    return max(start, min(value, end))
