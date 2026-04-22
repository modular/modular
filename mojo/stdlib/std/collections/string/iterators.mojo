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
"""Implements the iterators and related utilities for efficient string
operations."""

from std.collections.string._utf8 import (
    _utf8_first_byte_sequence_length,
    _is_utf8_continuation_byte,
)


struct CodepointSliceIter[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator, Sized):
    """Iterator for `StringSlice` over substring slices containing a single
    Unicode codepoint.

    Parameters:
        mut: Whether the slice is mutable.
        origin: The origin of the underlying string data.
        forward: The iteration direction. `False` is backwards.

    The `forward` parameter only controls the behavior of the `__next__()`
    method used for normal iteration. Calls to `next()` will always take an
    element from the front of the iterator, and calls to `next_back()` will
    always take an element from the end.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type for this codepoint iterator.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime Element = StringSlice[Self.origin]
    """The element type yielded by iteration."""

    var _slice: StringSlice[Self.origin]

    # Note:
    #   Marked private since `StringSlice.codepoints()` is the intended public
    #   way to construct this type.
    @doc_hidden
    def __init__(out self, str_slice: StringSlice[Self.origin]):
        self._slice = str_slice

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the `StringSlice` yielding individual characters.

        Returns:
            An iterator over the characters in the string slice.
        """
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> StringSlice[Self.origin]:
        """Get the next codepoint in the underlying string slice.

        This returns the next single-codepoint substring slice encoded in the
        underlying string, and advances the iterator state.

        If `forward` is set to `False`, this will return the next codepoint
        from the end of the string.

        This function will abort if this iterator has been exhausted.

        Returns:
            The next character in the string.

        Raises:
            `StopIteration` if the iterator has been exhausted.
        """

        # NOTE: This intentionally check if the length *in bytes* is greater
        # than zero, because checking the codepoint length requires a linear
        # scan of the string, which is needlessly expensive for this purpose.
        if self._slice.byte_length() <= 0:
            raise StopIteration()

        comptime if Self.forward:
            return self.next().value()
        else:
            return self.next_back().value()

    @always_inline
    def __len__(self) -> Int:
        """Returns the remaining length of this iterator in `Codepoint`s.

        The value returned from this method indicates the number of subsequent
        calls to `next()` that will return a value.

        Returns:
            Number of codepoints remaining in this iterator.
        """
        return self._slice.count_codepoints()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def peek_next(self) -> Optional[StringSlice[Self.origin]]:
        """Check what the next single-codepoint slice in this iterator is,
        without advancing the iterator state.

        Repeated calls to this method will return the same value.

        Returns:
            The next codepoint slice in the underlying string, or None if the
            string is empty.

        **Examples:**

        `peek_next()` does not advance the iterator, so repeated calls will
        return the same value:

        ```mojo
        from std.collections.string import Codepoint
        from std.testing import assert_equal

        var input = StringSlice("123")
        var iter = input.codepoint_slices()

        assert_equal(iter.peek_next().value(), "1")
        assert_equal(iter.peek_next().value(), "1")
        assert_equal(iter.peek_next().value(), "1")

        # A call to `next()` return the same value as `peek_next()` had,
        # but also advance the iterator.
        assert_equal(iter.next().value(), "1")

        # Later `peek_next()` calls will return the _new_ next character:
        assert_equal(iter.peek_next().value(), "2")
        ```
        """
        if self._slice.byte_length() > 0:
            # SAFETY: Will not read out of bounds because `_slice` is guaranteed
            #   to contain valid UTF-8.
            var curr_ptr = self._slice.unsafe_ptr()
            var byte_len = _utf8_first_byte_sequence_length(curr_ptr[])
            return StringSlice[Self.origin](ptr=curr_ptr, length=byte_len)
        else:
            return None

    def peek_back(mut self) -> Optional[StringSlice[Self.origin]]:
        """Check what the last single-codepoint slice in this iterator is,
        without advancing the iterator state.

        Repeated calls to this method will return the same value.

        Returns:
            The last codepoint slice in the underlying string, or None if the
            string is empty.

        **Examples:**

        `peek_back()` does not advance the iterator, so repeated calls will
        return the same value:

        ```mojo
        from std.collections.string import Codepoint
        from std.testing import assert_equal

        var input = StringSlice("123")
        var iter = input.codepoint_slices()

        # Repeated calls to `peek_back()` return the same value.
        assert_equal(iter.peek_back().value(), "3")
        assert_equal(iter.peek_back().value(), "3")
        assert_equal(iter.peek_back().value(), "3")

        # A call to `next_back()` return the same value as `peek_back()` had,
        # but also advance the iterator.
        assert_equal(iter.next_back().value(), "3")

        # Later `peek_back()` calls will return the _new_ next character:
        assert_equal(iter.peek_back().value(), "2")
        ```
        """
        if self._slice.byte_length() > 0:
            var byte_len = 1
            var back_ptr = (
                self._slice.unsafe_ptr() + self._slice.byte_length() - 1
            )
            # SAFETY:
            #   Guaranteed not to go out of bounds because UTF-8
            #   guarantees there is always a "start" byte eventually before any
            #   continuation bytes.
            while _is_utf8_continuation_byte(back_ptr[]):
                byte_len += 1
                back_ptr -= 1

            return StringSlice[Self.origin](ptr=back_ptr, length=byte_len)
        else:
            return None

    def next(mut self) -> Optional[StringSlice[Self.origin]]:
        """Get the next codepoint slice in the underlying string slice, or None
        if the iterator is empty.

        This returns the next single-codepoint substring encoded in the
        underlying string, and advances the iterator state.

        Returns:
            A character if the string is not empty, otherwise None.
        """
        var result: Optional[StringSlice[Self.origin]] = self.peek_next()

        if result:
            # SAFETY: We just checked that `result` holds a value
            var slice_len = result.unsafe_value().byte_length()
            # Advance the pointer in _slice.
            self._slice._slice._data += slice_len
            # Decrement the byte-length of _slice.
            self._slice._slice._len -= slice_len

        return result

    def next_back(mut self) -> Optional[StringSlice[Self.origin]]:
        """Get the last single-codepoint slice in this iterator is, or None
        if the iterator is empty.

        This returns the last codepoint slice in this iterator, and advances
        the iterator state.

        Returns:
            The last codepoint slice in the underlying string, or None if the
            string is empty.
        """
        var result: Optional[StringSlice[Self.origin]] = self.peek_back()

        if result:
            # SAFETY: We just checked that `result` holds a value
            var slice_len = result.unsafe_value().byte_length()
            # Decrement the byte-length of _slice.
            self._slice._slice._len -= slice_len

        return result


struct CodepointsIter[mut: Bool, //, origin: Origin[mut=mut]](
    ImplicitlyCopyable, Iterable, Iterator, Sized
):
    """Iterator over the `Codepoint`s in a string slice, constructed by
    `StringSlice.codepoints()`.

    Parameters:
        mut: Mutability of the underlying string data.
        origin: Origin of the underlying string data.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type for this codepoint iterator.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime Element = Codepoint
    """The element type yielded by iteration."""

    var _slice: StringSlice[Self.origin]
    """String slice containing the bytes that have not been read yet.

    When this iterator advances, the pointer in `_slice` is advanced by the
    byte length of each read character, and the slice length is decremented by
    the same amount.
    """

    # Note:
    #   Marked private since `StringSlice.codepoints()` is the intended public
    #   way to construct this type.
    @doc_hidden
    def __init__(out self, str_slice: StringSlice[Self.origin]):
        self._slice = str_slice

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(mut self) raises StopIteration -> Codepoint:
        """Get the next codepoint in the underlying string slice.

        This returns the next `Codepoint` encoded in the underlying string, and
        advances the iterator state.

        Returns:
            The next character in the string.

        Raises:
            StopIteration: If the iterator is exhausted.
        """
        if self._slice.byte_length() <= 0:
            raise StopIteration()
        return self.next().value()

    @always_inline
    def __len__(self) -> Int:
        """Returns the remaining length of this iterator in `Codepoint`s.

        The value returned from this method indicates the number of subsequent
        calls to `next()` that will return a value.

        Returns:
            Number of codepoints remaining in this iterator.
        """
        return self._slice.count_codepoints()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def peek_next(self) -> Optional[Codepoint]:
        """Check what the next codepoint in this iterator is, without advancing
        the iterator state.

        Repeated calls to this method will return the same value.

        Returns:
            The next character in the underlying string, or None if the string
            is empty.

        # Examples

        `peek_next()` does not advance the iterator, so repeated calls will
        return the same value:

        ```mojo
        from std.collections.string import Codepoint
        from std.testing import assert_equal

        var input = StringSlice("123")
        var iter = input.codepoints()

        assert_equal(iter.peek_next().value(), Codepoint.ord("1"))
        assert_equal(iter.peek_next().value(), Codepoint.ord("1"))
        assert_equal(iter.peek_next().value(), Codepoint.ord("1"))

        # A call to `next()` return the same value as `peek_next()` had,
        # but also advance the iterator.
        assert_equal(iter.next().value(), Codepoint.ord("1"))

        # Later `peek_next()` calls will return the _new_ next character:
        assert_equal(iter.peek_next().value(), Codepoint.ord("2"))
        ```
        """
        if self._slice.byte_length() > 0:
            # SAFETY: Will not read out of bounds because `_slice` is guaranteed
            #   to contain valid UTF-8.
            codepoint, _ = Codepoint.unsafe_decode_utf8_codepoint(
                self._slice._slice
            )
            return codepoint
        else:
            return None

    def next(mut self) -> Optional[Codepoint]:
        """Get the next codepoint in the underlying string slice, or None if
        the iterator is empty.

        This returns the next `Codepoint` encoded in the underlying string, and
        advances the iterator state.

        Returns:
            A character if the string is not empty, otherwise None.
        """
        var result: Optional[Codepoint] = self.peek_next()

        if result:
            # SAFETY: We just checked that `result` holds a value
            var char_len = result.unsafe_value().utf8_byte_length()
            # Advance the pointer in _slice.
            self._slice._slice._data += char_len
            # Decrement the byte-length of _slice.
            self._slice._slice._len -= char_len

        return result
