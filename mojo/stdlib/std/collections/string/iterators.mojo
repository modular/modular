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
from std.collections.string._grapheme_break import (
    _GraphemeBreakState,
    _is_grapheme_break,
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


struct GraphemeSliceIter[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
](ImplicitlyCopyable, Iterable, Iterator, Sized):
    """Iterator over grapheme clusters in a string, yielding each cluster as a
    `StringSlice`.

    A grapheme cluster is what a user would typically think of as a single
    "character" on screen. This includes combining character sequences, emoji
    with modifiers, flag sequences, and other multi-codepoint grapheme clusters
    as defined by UAX #29.

    Parameters:
        mut: Whether the slice is mutable.
        origin: The origin of the underlying string data.

    Note: Only forward iteration is supported. Backward grapheme iteration
    would require a different algorithm since the UAX #29 state machine is
    inherently forward-scanning.

    Note: `len()` is an O(n) operation that must scan all remaining bytes
    to count grapheme boundaries. Avoid calling it in a loop; prefer
    iterating with `for g in s.graphemes()` or calling `next()` until
    `None`.

    # TODO: Add a SIMD fast path for ASCII runs — every ASCII byte is its
    # own grapheme, so chunks of `< 0x80` bytes can be counted directly
    # without entering the state machine.
    # TODO: Add `next_back()` for reverse grapheme iteration (needed for
    # cursor movement, backspace, rtrim). Approach: back up a bounded
    # number of codepoints to a safe restart point, then scan forward.

    Example:

    ```mojo
    %# from testing import assert_equal
    var text = String("cafe\\u{0301}")  # "café" with combining accent
    var count = 0
    for grapheme in text.graphemes():
        count += 1
    # count == 4: c, a, f, e + combining acute (2 codepoints, 1 grapheme)
    assert_equal(count, 4)
    ```
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime Element = StringSlice[Self.origin]
    """The element type yielded by iteration."""

    var _slice: StringSlice[Self.origin]
    """Remaining bytes to iterate over."""

    var _state: _GraphemeBreakState
    """UAX #29 segmentation state."""

    var _state_primed: Bool
    """True if the state machine already processed the first codepoint of
    the next grapheme (from the previous break detection)."""

    @doc_hidden
    def __init__(out self, str_slice: StringSlice[Self.origin]):
        """Construct from a string slice.

        Args:
            str_slice: The string slice to iterate over.
        """
        self._slice = str_slice
        self._state = _GraphemeBreakState()
        self._state_primed = False

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Return an iterator over grapheme clusters.

        Returns:
            A copy of this iterator.
        """
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> StringSlice[Self.origin]:
        """Get the next grapheme cluster.

        Returns:
            The next grapheme cluster as a `StringSlice`.

        Raises:
            `StopIteration` if the iterator has been exhausted.
        """
        if self._slice.byte_length() <= 0:
            raise StopIteration()
        return self.next().value()

    def __len__(self) -> Int:
        """Return the number of remaining grapheme clusters.

        This is an O(n) operation that scans all remaining bytes to count
        grapheme cluster boundaries.

        Returns:
            The number of grapheme clusters remaining.
        """
        # The state machine reports a break at start-of-text (GB1) via its
        # initial `prev_gbp = GBP_CONTROL`, so every grapheme cluster --
        # including the first -- is signalled by a `True` return.
        var count = 0
        var state = _GraphemeBreakState()
        var remaining = self._slice

        while remaining.byte_length() > 0:
            var cp, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
                remaining._slice
            )
            if _is_grapheme_break(state, cp.to_u32()):
                count += 1

            remaining._slice._data += num_bytes
            remaining._slice._len -= num_bytes

        return count

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def next(mut self) -> Optional[StringSlice[Self.origin]]:
        """Get the next grapheme cluster, or `None` if exhausted.

        Returns:
            The next grapheme cluster as a `StringSlice`, or `None`.
        """
        if self._slice.byte_length() <= 0:
            return None

        var start_ptr = self._slice.unsafe_ptr()
        var total_bytes = self._slice.byte_length()
        var consumed = 0

        # Decode the first codepoint of this grapheme cluster.
        var cp, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
            self._slice._slice
        )

        if not self._state_primed:
            # First call, or state is not yet primed: feed this codepoint
            # to the state machine to establish the initial state.
            _ = _is_grapheme_break(self._state, cp.to_u32())
        # else: state was already updated for this codepoint when the
        # previous next() detected the break. Skip re-feeding.

        consumed += num_bytes

        # Continue consuming codepoints until we hit a break.
        var found_break = False
        while consumed < total_bytes:
            var remaining = Span[Byte, Self.origin](
                ptr=self._slice.unsafe_ptr() + consumed,
                length=total_bytes - consumed,
            )
            cp, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(remaining)

            if _is_grapheme_break(self._state, cp.to_u32()):
                # Found a break — the grapheme ends before this codepoint.
                # The state machine has already been updated with this
                # codepoint, so mark as primed for the next call.
                found_break = True
                break

            consumed += num_bytes

        self._state_primed = found_break

        # Advance the slice past this grapheme cluster.
        self._slice._slice._data += consumed
        self._slice._slice._len -= consumed

        return StringSlice[Self.origin](ptr=start_ptr, length=consumed)
